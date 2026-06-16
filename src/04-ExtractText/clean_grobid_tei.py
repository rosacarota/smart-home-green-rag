from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass, field
from collections import Counter
import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET


TEI_DIR = Path("data/articles/extracted_grobid_tei")

# Canonical outputs for the RAG pipeline.
CLEANED_BLOCKS_DIR = Path("data/articles/cleaned_blocks")
SUPPLEMENTARY_BLOCKS_DIR = Path("data/articles/supplementary_blocks")
TABLE_BLOCKS_DIR = Path("data/articles/table_blocks")

# Diagnostic outputs.
REPORT_DIR = Path("data/articles/cleaning_reports")
DISCARDED_BLOCKS_DIR = Path("data/articles/discarded_blocks")
REVIEW_BLOCKS_DIR = Path("data/articles/review_blocks")

TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NAMESPACE}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# Administrative sections that are not useful for the RAG knowledge base.
EXCLUDED_SECTION_TITLES = {
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "author contributions",
    "authors contributions",
    "funding",
    "funding information",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "declaration of interests",
    "data availability",
    "data availability statement",
    "ethical approval",
    "ethics statement",
    "informed consent",
    "supplementary material",
    "supplemental material",
}

# Useful, but usually not part of the main narrative flow.
SUPPLEMENTARY_SECTION_TITLES = {
    "glossary",
    "highlights",
    "key points",
    "key messages",
    "research highlights",
    "outstanding questions",
    "research questions",
    "trends",
    "sidebar",
    "case study",
}

EXCLUDED_CONTAINER_MARKERS = {
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgement",
    "funding",
    "conflict",
    "competing-interests",
    "data-availability",
    "ethics",
    "supplementary-material",
}

SUPPLEMENTARY_CONTAINER_MARKERS = {
    "box",
    "boxed-text",
    "sidebar",
    "panel",
    "glossary",
    "highlights",
    "key-points",
    "questions",
    "trends",
    "case-study",
}

# Inline elements ignored when extracting prose.
INLINE_SKIPPED_ELEMENT_NAMES = {
    "figure",
    "table",
    "graphic",
    "figDesc",
    "fw",
    "note",
    "listBibl",
    "biblStruct",
}

PUBLISHER_BOILERPLATE_PATTERNS = [
    re.compile(
        r"Contents lists available at.*?"
        r"journal homepage:\s*(?:https?://)?\S+",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"©\s*\d{4}.*?all rights reserved\.?",
        flags=re.IGNORECASE,
    ),
]

FOOTER_PATTERN = re.compile(
    r"^.{0,100}\b(?:vol\.?|volume)\s*\d+.*"
    r"\b(?:no\.?|issue)\s*\d+.*\b\d{1,4}$",
    flags=re.IGNORECASE,
)

CAPTION_PREFIX_PATTERN = re.compile(
    r"^(?:figure|fig\.?|table|scheme|chart|plate)\s+"
    r"(?:\d+|[ivxlcdm]+)[\s.:)-]",
    flags=re.IGNORECASE,
)

DEFINITION_PATTERN = re.compile(
    r"(?:^|[.;]\s+)"
    r"[A-Z][A-Za-z0-9 ()/\-]{1,70}:"
)

EMBEDDED_SUPPLEMENTARY_PATTERN = re.compile(
    r"\b(?:Glossary|Highlights|Key Points|Key Messages|"
    r"Outstanding Questions|Research Questions|Trends)\b"
)

TABLE_LEAKAGE_MARKERS = {
    "technique used for activity detection",
    "papers 1",
    "comparison of smart home",
}


@dataclass
class Block:
    kind: str
    text: str
    category: str
    reason: str = ""
    level: int = 0
    section_path: list[str] = field(default_factory=list)
    source_tag: str = ""
    source_type: str = ""
    coords: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class AbstractCandidate:
    text: str
    score: float
    word_count: int
    language: str = ""


@dataclass
class CleaningResult:
    main_blocks: list[Block]
    supplementary_blocks: list[Block]
    table_blocks: list[Block]
    discarded_blocks: list[Block]
    review_blocks: list[Block]
    report: dict


def normalize_article_name(article: str) -> str:
    article = article.strip()

    if article.lower().endswith(".xml"):
        article = article[:-4]

    if article.endswith(".grobid.tei"):
        article = article.removesuffix(".grobid.tei")
    elif article.endswith(".tei"):
        article = article.removesuffix(".tei")

    if article.startswith("article_"):
        return article

    return f"article_{article}"


def get_article_id(tei_path: Path) -> str:
    filename = tei_path.name

    for suffix in (
        ".grobid.tei.xml",
        ".tei.xml",
        ".xml",
    ):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]

    return tei_path.stem


def find_tei_file(article: str) -> Path:
    article_id = normalize_article_name(article)

    candidates = [
        TEI_DIR / f"{article_id}.grobid.tei.xml",
        TEI_DIR / f"{article_id}.tei.xml",
        TEI_DIR / f"{article_id}.xml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matching_files = sorted(TEI_DIR.glob(f"{article_id}*.tei.xml"))

    if matching_files:
        return matching_files[0]

    raise FileNotFoundError(f"TEI file not found for article: {article_id}")


def find_all_tei_files() -> list[Path]:
    tei_files = sorted(TEI_DIR.glob("article_*.tei.xml"))

    if not tei_files:
        raise FileNotFoundError(f"No TEI files found in: {TEI_DIR}")

    return tei_files


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", maxsplit=1)[1]

    return tag


def normalize_text(text: str) -> str:
    # Normalize Unicode variants without rewriting the scientific content.
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphens, zero-width characters, and byte-order marks.
    text = text.replace("\u00ad", "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Collapse PDF line breaks and repeated spaces.
    text = re.sub(r"\s+", " ", text).strip()

    for pattern in PUBLISHER_BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)

    # Correct spacing introduced by removed inline elements.
    text = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_heading(heading: str) -> str:
    heading = unicodedata.normalize("NFKC", heading).casefold()
    heading = re.sub(r"[^\w]+", " ", heading, flags=re.UNICODE)
    return re.sub(r"\s+", " ", heading).strip()


def extract_element_text(element: ET.Element) -> str:
    parts: list[str] = []

    if element.text:
        parts.append(element.text)

    for child in element:
        child_name = local_name(child.tag)
        child_type = child.attrib.get("type", "").lower()

        skip_child = child_name in INLINE_SKIPPED_ELEMENT_NAMES

        # Bibliographic references are usually not useful inside the chunk text.
        if child_name == "ref" and child_type == "bibr":
            skip_child = True

        if child_name == "lb":
            parts.append(" ")
        elif child_name == "s" and not skip_child:
            sentence_text = extract_element_text(child)
            if sentence_text:
                parts.append(sentence_text)
                parts.append(" ")
        elif not skip_child:
            parts.append(extract_element_text(child))

        if child.tail:
            parts.append(child.tail)

    return normalize_text("".join(parts))


def heading_matches(heading: str, candidates: set[str]) -> bool:
    normalized_heading = normalize_heading(heading)

    for candidate in candidates:
        if normalized_heading == candidate:
            return True

        if normalized_heading.startswith(f"{candidate} "):
            return True

    return False


def heading_equals(heading: str, candidates: set[str]) -> bool:
    return normalize_heading(heading) in candidates


def get_container_attributes(element: ET.Element) -> str:
    values = [
        element.attrib.get("type", ""),
        element.attrib.get("subtype", ""),
        element.attrib.get("place", ""),
        element.attrib.get("rend", ""),
    ]

    return normalize_heading(" ".join(values))


def find_direct_heading(element: ET.Element) -> str:
    for child in element:
        if local_name(child.tag) == "head":
            return extract_element_text(child)

    return ""


def classify_container(
    element: ET.Element,
    inherited_category: str,
) -> tuple[str, str]:
    heading = find_direct_heading(element)
    attributes = get_container_attributes(element)

    if heading and heading_matches(heading, EXCLUDED_SECTION_TITLES):
        return "discard", "excluded_section"

    if any(marker in attributes for marker in EXCLUDED_CONTAINER_MARKERS):
        return "discard", "excluded_container_type"

    if heading:
        normalized_heading = normalize_heading(heading)

        if (
            normalized_heading.startswith("box ")
            or normalized_heading.startswith("case study ")
        ):
            return "supplementary", "boxed_content"

        if heading_equals(heading, SUPPLEMENTARY_SECTION_TITLES):
            return "supplementary", "supplementary_section"

    if any(marker in attributes for marker in SUPPLEMENTARY_CONTAINER_MARKERS):
        return "supplementary", "supplementary_container_type"

    return inherited_category, "inherited"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def numeric_ratio(text: str) -> float:
    words = text.split()

    if not words:
        return 0.0

    numeric_words = sum(any(character.isdigit() for character in word) for word in words)
    return numeric_words / len(words)


def parse_coords(coords: str) -> list[tuple[int, float, float, float, float]]:
    parsed: list[tuple[int, float, float, float, float]] = []

    if not coords:
        return parsed

    for item in coords.split(";"):
        parts = item.split(",")

        if len(parts) < 5:
            continue

        try:
            page = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            parsed.append((page, x, y, width, height))
        except ValueError:
            continue

    return parsed


def coord_pages(coords: str) -> list[int]:
    pages = sorted({page for page, *_ in parse_coords(coords)})
    return pages


def small_font_ratio(coords: str, threshold: float = 7.0) -> float:
    parsed_coords = parse_coords(coords)

    if not parsed_coords:
        return 0.0

    small_count = sum(1 for *_, height in parsed_coords if height < threshold)
    return small_count / len(parsed_coords)


def looks_like_definition_block(text: str) -> bool:
    definitions = DEFINITION_PATTERN.findall(text)
    words = max(word_count(text), 1)
    colon_density = text.count(":") / words

    return len(definitions) >= 3 or (
        text.count(":") >= 4 and colon_density >= 0.02
    )


def looks_like_question_block(text: str) -> bool:
    questions = text.count("?")
    return questions >= 3 and word_count(text) <= 700


def looks_like_caption(text: str) -> bool:
    normalized_text = text.casefold().strip()

    if CAPTION_PREFIX_PATTERN.match(text):
        return True

    credit_markers = (
        "photo credit",
        "image credit",
        "figure credit",
        "data from",
    )

    return word_count(text) <= 160 and any(
        marker in normalized_text for marker in credit_markers
    )


def looks_like_chart_text(text: str) -> bool:
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)

    return (
        word_count(text) <= 140
        and len(years) >= 4
        and numeric_ratio(text) >= 0.30
    )


def looks_like_table_leakage_sentence(text: str, coords: str) -> bool:
    normalized_text = normalize_heading(text)
    words = word_count(text)
    ratio = small_font_ratio(coords)

    has_table_marker = any(
        marker in normalized_text for marker in TABLE_LEAKAGE_MARKERS
    )

    numbered_category_count = len(
        re.findall(r"\b\d+\s+[A-Z][A-Za-z\- ]{3,60}", text)
    )

    many_references = text.count(";") >= 12
    high_numeric_density = words <= 180 and numeric_ratio(text) >= 0.35

    if has_table_marker:
        return True

    if ratio >= 0.85 and (
        numbered_category_count >= 2
        or many_references
        or high_numeric_density
    ):
        return True

    if ratio >= 0.50 and (
        numbered_category_count >= 3
        or many_references
        or high_numeric_density
    ):
        return True

    return False


def is_noise_paragraph(text: str) -> bool:
    if not text:
        return True

    if re.fullmatch(r"\d{1,4}", text):
        return True

    if FOOTER_PATTERN.fullmatch(text):
        return True

    normalized_text = text.casefold()

    short_noise_markers = (
        "all rights reserved",
        "journal homepage",
        "available online",
        "contents lists available",
    )

    if word_count(text) <= 45 and any(
        marker in normalized_text for marker in short_noise_markers
    ):
        return True

    if re.fullmatch(r"(?:https?://)?doi\.org/\S+", text):
        return True

    return False


def detect_block_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    words = word_count(text)

    if words >= 1200:
        warnings.append("very_long_block")

    if words >= 80 and text[-1:] not in ".!?;:":
        warnings.append("missing_terminal_punctuation")

    if numeric_ratio(text) >= 0.25:
        warnings.append("high_numeric_density")

    embedded_match = EMBEDDED_SUPPLEMENTARY_PATTERN.search(text)

    if embedded_match and embedded_match.start() > 30:
        warnings.append("possible_embedded_sidebar")

    return warnings


def classify_text_block(
    text: str,
    inherited_category: str,
) -> tuple[str, str, list[str]]:
    warnings = detect_block_warnings(text)

    if inherited_category == "discard":
        return "discard", "discarded_parent", warnings

    if is_noise_paragraph(text):
        return "discard", "boilerplate_or_noise", warnings

    if looks_like_caption(text):
        return "discard", "figure_or_table_caption", warnings

    if looks_like_chart_text(text):
        return "discard", "chart_text", warnings

    if inherited_category == "supplementary":
        return "supplementary", "supplementary_parent", warnings

    if looks_like_definition_block(text):
        return "supplementary", "definition_block", warnings

    if looks_like_question_block(text):
        return "supplementary", "question_block", warnings

    return "main", "main_content", warnings


def extract_clean_text_from_sentences(
    element: ET.Element,
    section_path: list[str],
    level: int,
    source_kind: str,
    discarded_blocks: list[Block],
) -> tuple[str, list[str]]:
    sentence_elements = element.findall(".//tei:s", NS)

    if not sentence_elements:
        return extract_element_text(element), []

    kept_sentences: list[str] = []
    warnings: list[str] = []

    for sentence_index, sentence in enumerate(sentence_elements, start=1):
        sentence_text = extract_element_text(sentence)

        if not sentence_text:
            continue

        sentence_coords = sentence.attrib.get("coords", "")

        if looks_like_table_leakage_sentence(sentence_text, sentence_coords):
            warnings.append("removed_possible_table_leakage_sentence")
            discarded_blocks.append(
                Block(
                    kind="sentence",
                    text=sentence_text,
                    category="discard",
                    reason="possible_table_leakage_sentence",
                    level=level,
                    section_path=list(section_path),
                    source_tag=local_name(sentence.tag),
                    source_type=source_kind,
                    coords=sentence_coords,
                    warnings=[f"sentence_index:{sentence_index}"],
                )
            )
            continue

        kept_sentences.append(sentence_text)

    return normalize_text(" ".join(kept_sentences)), sorted(set(warnings))


def find_document_title(root: ET.Element) -> str:
    title_paths = [
        ".//tei:teiHeader//tei:titleStmt/tei:title[@type='main']",
        ".//tei:teiHeader//tei:titleStmt/tei:title",
        ".//tei:teiHeader//tei:analytic/tei:title",
    ]

    for title_path in title_paths:
        title_element = root.find(title_path, NS)

        if title_element is None:
            continue

        title = extract_element_text(title_element)

        if title:
            return title

    return ""


def collect_abstract_elements(root: ET.Element) -> list[ET.Element]:
    elements = list(root.findall(".//tei:teiHeader//tei:abstract", NS))

    for div_element in root.findall(".//tei:text/tei:front//tei:div", NS):
        attributes = get_container_attributes(div_element)

        if "abstract" in attributes:
            elements.append(div_element)

    unique_elements: list[ET.Element] = []
    seen_ids: set[int] = set()

    for element in elements:
        element_id = id(element)

        if element_id not in seen_ids:
            seen_ids.add(element_id)
            unique_elements.append(element)

    return unique_elements


def extract_abstract_candidate(element: ET.Element) -> str:
    paragraph_elements = element.findall(".//tei:p", NS)

    if not paragraph_elements:
        return extract_element_text(element)

    paragraphs = [extract_element_text(paragraph) for paragraph in paragraph_elements]
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if paragraph and not is_noise_paragraph(paragraph)
    ]

    return "\n\n".join(paragraphs)


def score_abstract_candidate(text: str, language: str) -> float:
    words = word_count(text)
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", text))
    normalized_text = text.casefold()
    score = 0.0

    if 50 <= words <= 600:
        score += 3.0
    elif 30 <= words <= 900:
        score += 1.0
    else:
        score -= 2.0

    if sentence_count >= 3:
        score += 2.0

    abstract_markers = (
        "in this paper",
        "in this study",
        "we present",
        "we review",
        "we investigate",
        "we examine",
        "our study",
        "our analysis",
    )

    if any(marker in normalized_text for marker in abstract_markers):
        score += 4.0

    if language.casefold().startswith("en"):
        score += 0.5

    if looks_like_definition_block(text):
        score -= 7.0

    if looks_like_caption(text) or looks_like_chart_text(text):
        score -= 6.0

    if looks_like_question_block(text):
        score -= 3.0

    return score


def select_abstract(root: ET.Element) -> tuple[str, list[AbstractCandidate], list[str]]:
    candidates: list[AbstractCandidate] = []
    warnings: list[str] = []

    for element in collect_abstract_elements(root):
        text = extract_abstract_candidate(element)

        if not text:
            continue

        language = element.attrib.get(XML_LANG, "")
        candidates.append(
            AbstractCandidate(
                text=text,
                score=score_abstract_candidate(text, language),
                word_count=word_count(text),
                language=language,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    if not candidates:
        warnings.append("abstract_not_found")
        return "", candidates, warnings

    if len(candidates) > 1:
        score_difference = candidates[0].score - candidates[1].score

        if score_difference <= 1.5:
            warnings.append("ambiguous_abstract_selection")

    selected = candidates[0]

    if selected.score < 0:
        warnings.append("low_confidence_abstract")

    return selected.text, candidates, warnings


def append_text_block(
    element: ET.Element,
    kind: str,
    inherited_category: str,
    section_path: list[str],
    level: int,
    blocks: list[Block],
    discarded_blocks: list[Block],
    list_prefix: str = "",
) -> None:
    text, extraction_warnings = extract_clean_text_from_sentences(
        element=element,
        section_path=section_path,
        level=level,
        source_kind=kind,
        discarded_blocks=discarded_blocks,
    )

    if not text:
        return

    if list_prefix:
        text = f"{list_prefix}{text}"

    category, reason, warnings = classify_text_block(text, inherited_category)
    warnings = sorted(set([*warnings, *extraction_warnings]))

    block = Block(
        kind=kind,
        text=text,
        category=category,
        reason=reason,
        level=level,
        section_path=list(section_path),
        source_tag=local_name(element.tag),
        source_type=element.attrib.get("type", ""),
        coords=element.attrib.get("coords", ""),
        warnings=warnings,
    )

    if category == "discard":
        discarded_blocks.append(block)
    else:
        blocks.append(block)


def append_discarded_container(
    element: ET.Element,
    reason: str,
    level: int,
    section_path: list[str],
    discarded_blocks: list[Block],
    kind: str = "container",
) -> None:
    discarded_text = extract_element_text(element)

    if not discarded_text:
        return

    discarded_blocks.append(
        Block(
            kind=kind,
            text=discarded_text,
            category="discard",
            reason=reason,
            level=level,
            section_path=list(section_path),
            source_tag=local_name(element.tag),
            source_type=element.attrib.get("type", ""),
            coords=element.attrib.get("coords", ""),
        )
    )


def walk_container(
    parent: ET.Element,
    blocks: list[Block],
    table_blocks: list[Block],
    discarded_blocks: list[Block],
    inherited_category: str,
    heading_level: int,
    section_path: list[str],
) -> None:
    container_category, container_reason = classify_container(parent, inherited_category)

    current_category = container_category
    current_section_path = list(section_path)

    if current_category == "discard":
        append_discarded_container(
            element=parent,
            reason=container_reason,
            level=heading_level,
            section_path=section_path,
            discarded_blocks=discarded_blocks,
        )
        return

    for child in parent:
        child_name = local_name(child.tag)

        if child_name == "head":
            heading = extract_element_text(child)

            if not heading:
                continue

            heading_category, heading_reason = classify_container(parent, current_category)

            if heading_matches(heading, EXCLUDED_SECTION_TITLES):
                current_category = "discard"
                continue

            normalized_heading = normalize_heading(heading)

            if (
                normalized_heading.startswith("box ")
                or normalized_heading.startswith("case study ")
                or heading_equals(heading, SUPPLEMENTARY_SECTION_TITLES)
            ):
                current_category = "supplementary"
                heading_reason = "supplementary_heading"
            else:
                current_category = heading_category

            current_section_path = [*section_path, heading]

            blocks.append(
                Block(
                    kind="heading",
                    text=heading,
                    category=current_category,
                    reason=heading_reason,
                    level=min(max(heading_level, 1), 6),
                    section_path=list(current_section_path),
                    source_tag=child_name,
                    source_type=child.attrib.get("type", ""),
                    coords=child.attrib.get("coords", ""),
                )
            )
            continue

        if child_name == "div":
            walk_container(
                parent=child,
                blocks=blocks,
                table_blocks=table_blocks,
                discarded_blocks=discarded_blocks,
                inherited_category=current_category,
                heading_level=min(heading_level + 1, 6),
                section_path=current_section_path,
            )
            continue

        if child_name in {"p", "ab"}:
            append_text_block(
                element=child,
                kind="paragraph",
                inherited_category=current_category,
                section_path=current_section_path,
                level=heading_level,
                blocks=blocks,
                discarded_blocks=discarded_blocks,
            )
            continue

        if child_name == "list":
            list_type = child.attrib.get("type", "").casefold()
            items = child.findall("./tei:item", NS)

            for index, item in enumerate(items, start=1):
                prefix = f"{index}. " if list_type in {"ordered", "numbered"} else "- "
                append_text_block(
                    element=item,
                    kind="list_item",
                    inherited_category=current_category,
                    section_path=current_section_path,
                    level=heading_level,
                    blocks=blocks,
                    discarded_blocks=discarded_blocks,
                    list_prefix=prefix,
                )
            continue

        if child_name == "quote":
            append_text_block(
                element=child,
                kind="quote",
                inherited_category=current_category,
                section_path=current_section_path,
                level=heading_level,
                blocks=blocks,
                discarded_blocks=discarded_blocks,
            )
            continue

        if child_name == "formula":
            append_text_block(
                element=child,
                kind="formula",
                inherited_category=current_category,
                section_path=current_section_path,
                level=heading_level,
                blocks=blocks,
                discarded_blocks=discarded_blocks,
            )
            continue

        if child_name == "note":
            walk_container(
                parent=child,
                blocks=blocks,
                table_blocks=table_blocks,
                discarded_blocks=discarded_blocks,
                inherited_category="supplementary",
                heading_level=min(heading_level + 1, 6),
                section_path=current_section_path,
            )
            continue

        if child_name == "figure":
            figure_type = child.attrib.get("type", "").casefold()
            figure_text = extract_element_text(child)

            if figure_type == "table":
                if figure_text:
                    table_blocks.append(
                        Block(
                            kind="table",
                            text=figure_text,
                            category="table",
                            reason="grobid_table_figure",
                            level=heading_level,
                            section_path=list(current_section_path),
                            source_tag=child_name,
                            source_type=figure_type,
                            coords=child.attrib.get("coords", ""),
                            warnings=detect_block_warnings(figure_text),
                        )
                    )
            else:
                append_discarded_container(
                    element=child,
                    reason="figure_content",
                    level=heading_level,
                    section_path=current_section_path,
                    discarded_blocks=discarded_blocks,
                    kind="figure",
                )
            continue

        if child_name in {
            "table",
            "graphic",
            "figDesc",
            "fw",
            "listBibl",
            "biblStruct",
        }:
            append_discarded_container(
                element=child,
                reason=f"excluded_{child_name}",
                level=heading_level,
                section_path=current_section_path,
                discarded_blocks=discarded_blocks,
                kind=child_name,
            )
            continue

        # Preserve paragraphs nested inside publisher-specific wrapper tags.
        if list(child):
            walk_container(
                parent=child,
                blocks=blocks,
                table_blocks=table_blocks,
                discarded_blocks=discarded_blocks,
                inherited_category=current_category,
                heading_level=heading_level,
                section_path=current_section_path,
            )


def parse_body(root: ET.Element) -> tuple[list[Block], list[Block], list[Block], int]:
    body = root.find(".//tei:text/tei:body", NS)

    if body is None:
        raise ValueError("The TEI document does not contain a body element.")

    blocks: list[Block] = []
    table_blocks: list[Block] = []
    discarded_blocks: list[Block] = []
    raw_body_text = normalize_text(" ".join(body.itertext()))

    walk_container(
        parent=body,
        blocks=blocks,
        table_blocks=table_blocks,
        discarded_blocks=discarded_blocks,
        inherited_category="main",
        heading_level=1,
        section_path=[],
    )

    return blocks, table_blocks, discarded_blocks, len(raw_body_text)


def text_fingerprint(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\W+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def remove_duplicate_blocks(
    blocks: list[Block],
    abstract_text: str,
    discarded_blocks: list[Block],
) -> list[Block]:
    seen: set[str] = set()

    if abstract_text:
        seen.add(text_fingerprint(abstract_text))

    unique_blocks: list[Block] = []

    for block in blocks:
        if block.kind == "heading":
            unique_blocks.append(block)
            continue

        fingerprint = text_fingerprint(block.text)

        if len(fingerprint) >= 80 and fingerprint in seen:
            duplicate = Block(**asdict(block))
            duplicate.category = "discard"
            duplicate.reason = "duplicate_content"
            discarded_blocks.append(duplicate)
            continue

        if len(fingerprint) >= 80:
            seen.add(fingerprint)

        unique_blocks.append(block)

    return unique_blocks


def block_to_record(
    article_id: str,
    source_title: str,
    source_tei: Path,
    block: Block,
    block_index: int,
) -> dict:
    record = asdict(block)
    record.update(
        {
            "article_id": article_id,
            "source_title": source_title,
            "source_tei": str(source_tei),
            "block_index": block_index,
            "section_title": block.section_path[-1] if block.section_path else None,
            "section_depth": len(block.section_path),
            "word_count": word_count(block.text),
            "text_characters": len(block.text),
            "pages": coord_pages(block.coords),
        }
    )
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        if path.exists():
            path.unlink()
        return

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")


def build_records(
    article_id: str,
    source_title: str,
    source_tei: Path,
    blocks: list[Block],
) -> list[dict]:
    return [
        block_to_record(
            article_id=article_id,
            source_title=source_title,
            source_tei=source_tei,
            block=block,
            block_index=index,
        )
        for index, block in enumerate(blocks, start=1)
    ]


def clean_tei_file(tei_path: Path) -> Path:
    article_id = get_article_id(tei_path)
    main_output_path = CLEANED_BLOCKS_DIR / f"{article_id}.jsonl"
    supplementary_output_path = SUPPLEMENTARY_BLOCKS_DIR / f"{article_id}.jsonl"
    table_output_path = TABLE_BLOCKS_DIR / f"{article_id}.jsonl"
    report_path = REPORT_DIR / f"{article_id}.json"
    discarded_path = DISCARDED_BLOCKS_DIR / f"{article_id}.jsonl"
    review_path = REVIEW_BLOCKS_DIR / f"{article_id}.jsonl"

    print(f"Cleaning TEI file: {tei_path}")

    try:
        tree = ET.parse(tei_path)
    except ET.ParseError as error:
        raise ValueError(f"Invalid TEI XML file: {tei_path}") from error

    root = tree.getroot()
    title = find_document_title(root)
    document_warnings: list[str] = []

    if not title:
        title = article_id
        document_warnings.append("title_not_found")

    abstract_text, abstract_candidates, abstract_warnings = select_abstract(root)
    document_warnings.extend(abstract_warnings)

    blocks, table_blocks, discarded_blocks, raw_body_length = parse_body(root)
    blocks = remove_duplicate_blocks(blocks, abstract_text, discarded_blocks)

    if abstract_text:
        abstract_block = Block(
            kind="abstract",
            text=abstract_text,
            category="main",
            reason="selected_abstract",
            level=1,
            section_path=["Abstract"],
            source_tag="abstract",
            source_type="",
            coords="",
            warnings=[],
        )
        blocks = [abstract_block, *blocks]
    else:
        document_warnings.append("abstract_not_selected")

    main_blocks = [block for block in blocks if block.category == "main"]
    supplementary_blocks = [
        block for block in blocks if block.category == "supplementary"
    ]
    review_blocks = [
        block
        for block in [*main_blocks, *supplementary_blocks, *table_blocks]
        if block.warnings
    ]

    non_heading_main_blocks = [
        block for block in main_blocks if block.kind != "heading"
    ]

    if not non_heading_main_blocks:
        raise ValueError(f"No useful main content extracted from: {tei_path}")

    main_text_length = sum(len(block.text) for block in non_heading_main_blocks)
    cleaned_ratio = main_text_length / raw_body_length if raw_body_length else 0.0

    if cleaned_ratio < 0.25:
        document_warnings.append("low_main_text_ratio")

    if review_blocks:
        document_warnings.append("manual_review_recommended")

    main_records = build_records(article_id, title, tei_path, main_blocks)
    supplementary_records = build_records(
        article_id, title, tei_path, supplementary_blocks
    )
    table_records = build_records(article_id, title, tei_path, table_blocks)
    discarded_records = build_records(article_id, title, tei_path, discarded_blocks)
    review_records = build_records(article_id, title, tei_path, review_blocks)

    write_jsonl(main_output_path, main_records)
    write_jsonl(supplementary_output_path, supplementary_records)
    write_jsonl(table_output_path, table_records)
    write_jsonl(discarded_path, discarded_records)
    write_jsonl(review_path, review_records)

    report = {
        "article_id": article_id,
        "source_tei": str(tei_path),
        "main_blocks_output": str(main_output_path),
        "supplementary_blocks_output": (
            str(supplementary_output_path) if supplementary_records else None
        ),
        "table_blocks_output": str(table_output_path) if table_records else None,
        "discarded_blocks_output": str(discarded_path) if discarded_records else None,
        "review_blocks_output": str(review_path) if review_records else None,
        "title_found": title != article_id,
        "source_title": title,
        "abstract_candidates": [
            {
                "score": candidate.score,
                "word_count": candidate.word_count,
                "language": candidate.language,
                "preview": candidate.text[:300],
            }
            for candidate in abstract_candidates
        ],
        "selected_abstract_preview": abstract_text[:500],
        "counts": {
            "main_blocks": len(main_blocks),
            "main_text_blocks": len(non_heading_main_blocks),
            "supplementary_blocks": len(supplementary_blocks),
            "table_blocks": len(table_blocks),
            "discarded_blocks": len(discarded_blocks),
            "review_blocks": len(review_blocks),
            "block_kinds": dict(Counter(block.kind for block in blocks)),
            "discard_reasons": dict(
                Counter(block.reason for block in discarded_blocks)
            ),
            "warnings": dict(
                Counter(
                    warning
                    for block in [*main_blocks, *supplementary_blocks, *table_blocks]
                    for warning in block.warnings
                )
            ),
        },
        "quality": {
            "raw_body_characters": raw_body_length,
            "main_text_characters": main_text_length,
            "main_text_ratio": round(cleaned_ratio, 4),
        },
        "warnings": sorted(set(document_warnings)),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Main JSONL saved to: {main_output_path}")

    if supplementary_records:
        print(f"Supplementary JSONL saved to: {supplementary_output_path}")

    if table_records:
        print(f"Table JSONL saved to: {table_output_path}")

    if discarded_records:
        print(f"Discarded JSONL saved to: {discarded_path}")

    if review_records:
        print(f"Review JSONL saved to: {review_path}")

    print(f"Cleaning report saved to: {report_path}")
    print(
        "Cleaning summary: "
        f"{len(main_blocks)} main blocks, "
        f"{len(supplementary_blocks)} supplementary blocks, "
        f"{len(table_blocks)} table blocks, "
        f"{len(discarded_blocks)} discarded blocks, "
        f"{len(review_blocks)} review blocks."
    )

    return main_output_path


def clean_all_tei_files() -> None:
    tei_files = find_all_tei_files()
    successful_files = 0
    failed_files: list[tuple[Path, str]] = []

    print(f"TEI files found: {len(tei_files)}")

    for tei_path in tei_files:
        try:
            clean_tei_file(tei_path)
            successful_files += 1
        except (ValueError, OSError) as error:
            failed_files.append((tei_path, str(error)))

    print(f"Files cleaned successfully: {successful_files}/{len(tei_files)}")

    if failed_files:
        print("Files with errors:")

        for tei_path, error_message in failed_files:
            print(f"- {tei_path.name}: {error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GROBID TEI XML files into structured JSONL blocks "
            "for RAG ingestion. No Markdown output is generated."
        )
    )

    selection_group = parser.add_mutually_exclusive_group(required=True)

    selection_group.add_argument(
        "--article",
        type=str,
        help="Article number or name, for example: 2 or article_2.",
    )

    selection_group.add_argument(
        "--all",
        action="store_true",
        help="Clean all available TEI files.",
    )

    args = parser.parse_args()

    if args.article:
        clean_tei_file(find_tei_file(args.article))
    else:
        clean_all_tei_files()


if __name__ == "__main__":
    main()
