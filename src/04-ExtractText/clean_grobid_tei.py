from pathlib import Path
from dataclasses import asdict, dataclass, field
from collections import Counter
import argparse
import json
import re
import unicodedata
import xml.etree.ElementTree as ET


TEI_DIR = Path("data/articles/extracted_grobid_tei")
MAIN_OUTPUT_DIR = Path("data/articles/cleaned_markdown")
SUPPLEMENTARY_OUTPUT_DIR = Path("data/articles/supplementary_markdown")
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

# These sections can contain useful information, but they should not interrupt
# the main reading flow. They are written to a separate Markdown file.
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

# Inline elements are ignored when extracting the text of a paragraph.
# Standalone notes and boxed figures are handled separately as supplementary.
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

    raise FileNotFoundError(
        f"TEI file not found for article: {article_id}"
    )


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


def extract_element_text(element: ET.Element) -> str:
    parts: list[str] = []

    if element.text:
        parts.append(element.text)

    for child in element:
        child_name = local_name(child.tag)
        child_type = child.attrib.get("type", "").lower()

        skip_child = child_name in INLINE_SKIPPED_ELEMENT_NAMES

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

def normalize_heading(heading: str) -> str:
    heading = unicodedata.normalize("NFKC", heading).casefold()
    heading = re.sub(r"[^\w]+", " ", heading, flags=re.UNICODE)
    return re.sub(r"\s+", " ", heading).strip()


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

    if any(
        marker in attributes
        for marker in SUPPLEMENTARY_CONTAINER_MARKERS
    ):
        return "supplementary", "supplementary_container_type"

    return inherited_category, "inherited"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))


def numeric_ratio(text: str) -> float:
    words = text.split()

    if not words:
        return 0.0

    numeric_words = sum(
        any(character.isdigit() for character in word)
        for word in words
    )

    return numeric_words / len(words)


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

    return (
        word_count(text) <= 160
        and any(marker in normalized_text for marker in credit_markers)
    )


def looks_like_chart_text(text: str) -> bool:
    years = re.findall(r"\b(?:19|20)\d{2}\b", text)

    return (
        word_count(text) <= 140
        and len(years) >= 4
        and numeric_ratio(text) >= 0.30
    )


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
        marker in normalized_text
        for marker in short_noise_markers
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

    paragraphs = [
        extract_element_text(paragraph)
        for paragraph in paragraph_elements
    ]
    paragraphs = [
        paragraph
        for paragraph in paragraphs
        if paragraph and not is_noise_paragraph(paragraph)
    ]

    return "\n\n".join(paragraphs)


def score_abstract_candidate(
    text: str,
    language: str,
) -> float:
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


def select_abstract(
    root: ET.Element,
) -> tuple[str, list[AbstractCandidate], list[str]]:
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
    text = extract_element_text(element)

    if not text:
        return

    if list_prefix:
        text = f"{list_prefix}{text}"

    category, reason, warnings = classify_text_block(
        text,
        inherited_category,
    )

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


def walk_container(
    parent: ET.Element,
    blocks: list[Block],
    discarded_blocks: list[Block],
    inherited_category: str,
    heading_level: int,
    section_path: list[str],
) -> None:
    container_category, container_reason = classify_container(
        parent,
        inherited_category,
    )

    current_category = container_category
    current_section_path = list(section_path)

    if current_category == "discard":
        discarded_text = extract_element_text(parent)

        if discarded_text:
            discarded_blocks.append(
                Block(
                    kind="container",
                    text=discarded_text,
                    category="discard",
                    reason=container_reason,
                    level=heading_level,
                    section_path=list(section_path),
                    source_tag=local_name(parent.tag),
                    source_type=parent.attrib.get("type", ""),
                    coords=parent.attrib.get("coords", ""),
                )
            )

        return

    for child in parent:
        child_name = local_name(child.tag)

        if child_name == "head":
            heading = extract_element_text(child)

            if not heading:
                continue

            heading_category, heading_reason = classify_container(
                parent,
                current_category,
            )

            if heading_matches(heading, EXCLUDED_SECTION_TITLES):
                current_category = "discard"
                continue

            normalized_heading = normalize_heading(heading)

            if (
                normalized_heading.startswith("box ")
                or normalized_heading.startswith("case study ")
                or heading_equals(
                    heading,
                    SUPPLEMENTARY_SECTION_TITLES,
                )
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
                    level=min(max(heading_level, 2), 6),
                    section_path=list(current_section_path),
                    source_tag=child_name,
                    source_type=child.attrib.get("type", ""),
                    coords=child.attrib.get("coords", ""),
                )
            )
            continue

        if child_name == "div":
            walk_container(
                child,
                blocks,
                discarded_blocks,
                current_category,
                min(heading_level + 1, 6),
                current_section_path,
            )
            continue

        if child_name in {"p", "ab"}:
            append_text_block(
                child,
                "paragraph",
                current_category,
                current_section_path,
                heading_level,
                blocks,
                discarded_blocks,
            )
            continue

        if child_name == "list":
            list_type = child.attrib.get("type", "").casefold()
            items = child.findall("./tei:item", NS)

            for index, item in enumerate(items, start=1):
                prefix = (
                    f"{index}. "
                    if list_type in {"ordered", "numbered"}
                    else "- "
                )
                append_text_block(
                    item,
                    "list_item",
                    current_category,
                    current_section_path,
                    heading_level,
                    blocks,
                    discarded_blocks,
                    list_prefix=prefix,
                )
            continue

        if child_name == "quote":
            append_text_block(
                child,
                "quote",
                current_category,
                current_section_path,
                heading_level,
                blocks,
                discarded_blocks,
            )
            continue

        if child_name == "formula":
            append_text_block(
                child,
                "formula",
                current_category,
                current_section_path,
                heading_level,
                blocks,
                discarded_blocks,
            )
            continue

        if child_name == "note":
            walk_container(
                child,
                blocks,
                discarded_blocks,
                "supplementary",
                min(heading_level + 1, 6),
                current_section_path,
            )
            continue

        if child_name == "figure":
            figure_category, _ = classify_container(
                child,
                current_category,
            )

            if figure_category == "supplementary":
                walk_container(
                    child,
                    blocks,
                    discarded_blocks,
                    "supplementary",
                    min(heading_level + 1, 6),
                    current_section_path,
                )
            else:
                figure_text = extract_element_text(child)

                if figure_text:
                    discarded_blocks.append(
                        Block(
                            kind="figure",
                            text=figure_text,
                            category="discard",
                            reason="figure_content",
                            level=heading_level,
                            section_path=list(current_section_path),
                            source_tag=child_name,
                            source_type=child.attrib.get("type", ""),
                            coords=child.attrib.get("coords", ""),
                        )
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
            discarded_text = extract_element_text(child)

            if discarded_text:
                discarded_blocks.append(
                    Block(
                        kind=child_name,
                        text=discarded_text,
                        category="discard",
                        reason=f"excluded_{child_name}",
                        level=heading_level,
                        section_path=list(current_section_path),
                        source_tag=child_name,
                        source_type=child.attrib.get("type", ""),
                        coords=child.attrib.get("coords", ""),
                    )
                )
            continue

        # Preserve paragraphs nested inside publisher-specific wrapper tags.
        if list(child):
            walk_container(
                child,
                blocks,
                discarded_blocks,
                current_category,
                heading_level,
                current_section_path,
            )


def parse_body(
    root: ET.Element,
) -> tuple[list[Block], list[Block], int]:
    body = root.find(".//tei:text/tei:body", NS)

    if body is None:
        raise ValueError(
            "The TEI document does not contain a body element."
        )

    blocks: list[Block] = []
    discarded_blocks: list[Block] = []
    raw_body_text = normalize_text(" ".join(body.itertext()))

    walk_container(
        body,
        blocks,
        discarded_blocks,
        inherited_category="main",
        heading_level=2,
        section_path=[],
    )

    return blocks, discarded_blocks, len(raw_body_text)


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


def render_blocks(
    blocks: list[Block],
    category: str,
) -> str:
    lines: list[str] = []

    for block in blocks:
        if block.category != category:
            continue

        if block.kind == "heading":
            lines.extend(
                [
                    f"{'#' * block.level} {block.text}",
                    "",
                ]
            )
        elif block.kind == "quote":
            lines.extend([f"> {block.text}", ""])
        else:
            lines.extend([block.text, ""])

    markdown = "\n".join(lines)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def write_jsonl(path: Path, blocks: list[Block]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not blocks:
        if path.exists():
            path.unlink()
        return

    with path.open("w", encoding="utf-8") as file:
        for block in blocks:
            json.dump(asdict(block), file, ensure_ascii=False)
            file.write("\n")


def clean_tei_file(tei_path: Path) -> Path:
    article_id = get_article_id(tei_path)
    main_output_path = MAIN_OUTPUT_DIR / f"{article_id}.md"
    supplementary_output_path = (
        SUPPLEMENTARY_OUTPUT_DIR / f"{article_id}.md"
    )
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
    abstract_text, abstract_candidates, document_warnings = select_abstract(
        root
    )
    blocks, discarded_blocks, raw_body_length = parse_body(root)
    blocks = remove_duplicate_blocks(
        blocks,
        abstract_text,
        discarded_blocks,
    )

    main_blocks = [block for block in blocks if block.category == "main"]
    supplementary_blocks = [
        block for block in blocks
        if block.category == "supplementary"
    ]
    review_blocks = [block for block in blocks if block.warnings]

    main_body_markdown = render_blocks(blocks, "main")
    supplementary_markdown = render_blocks(blocks, "supplementary")

    main_lines: list[str] = []

    if title:
        main_lines.extend([f"# {title}", ""])
    else:
        document_warnings.append("title_not_found")

    if abstract_text:
        main_lines.extend(["## Abstract", "", abstract_text, ""])

    if main_body_markdown:
        main_lines.append(main_body_markdown)

    main_markdown = "\n".join(main_lines)
    main_markdown = re.sub(r"\n{3,}", "\n\n", main_markdown).strip()

    if not main_markdown:
        raise ValueError(f"No useful main content extracted from: {tei_path}")

    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_output_path.write_text(
        main_markdown + "\n",
        encoding="utf-8",
    )

    SUPPLEMENTARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if supplementary_markdown:
        supplementary_document = "\n".join(
            [
                f"# {title or article_id} — Supplementary Content",
                "",
                supplementary_markdown,
            ]
        ).strip()
        supplementary_output_path.write_text(
            supplementary_document + "\n",
            encoding="utf-8",
        )
    elif supplementary_output_path.exists():
        supplementary_output_path.unlink()

    write_jsonl(discarded_path, discarded_blocks)
    write_jsonl(review_path, review_blocks)

    main_text_length = sum(
        len(block.text)
        for block in main_blocks
        if block.kind != "heading"
    )
    cleaned_ratio = (
        main_text_length / raw_body_length
        if raw_body_length
        else 0.0
    )

    if not abstract_text:
        document_warnings.append("abstract_not_selected")

    if not any(block.kind == "paragraph" for block in main_blocks):
        document_warnings.append("no_main_paragraphs")

    if cleaned_ratio < 0.25:
        document_warnings.append("low_main_text_ratio")

    if review_blocks:
        document_warnings.append("manual_review_recommended")

    report = {
        "article_id": article_id,
        "source_tei": str(tei_path),
        "main_output": str(main_output_path),
        "supplementary_output": (
            str(supplementary_output_path)
            if supplementary_markdown
            else None
        ),
        "title_found": bool(title),
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
            "supplementary_blocks": len(supplementary_blocks),
            "discarded_blocks": len(discarded_blocks),
            "review_blocks": len(review_blocks),
            "block_kinds": dict(Counter(block.kind for block in blocks)),
            "discard_reasons": dict(
                Counter(block.reason for block in discarded_blocks)
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

    print(f"Main Markdown saved to: {main_output_path}")

    if supplementary_markdown:
        print(
            "Supplementary Markdown saved to: "
            f"{supplementary_output_path}"
        )

    print(f"Cleaning report saved to: {report_path}")
    print(
        "Cleaning summary: "
        f"{len(main_blocks)} main blocks, "
        f"{len(supplementary_blocks)} supplementary blocks, "
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

    print(
        "Files cleaned successfully: "
        f"{successful_files}/{len(tei_files)}"
    )

    if failed_files:
        print("Files with errors:")

        for tei_path, error_message in failed_files:
            print(f"- {tei_path.name}: {error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GROBID TEI XML files into cleaned main and "
            "supplementary Markdown documents."
        )
    )

    selection_group = parser.add_mutually_exclusive_group(required=True)

    selection_group.add_argument(
        "--article",
        type=str,
        help=(
            "Article number or name, for example: 2 or article_2."
        ),
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
