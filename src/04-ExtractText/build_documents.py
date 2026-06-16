from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

from llama_index.core import Document


BLOCKS_DIR = Path("data/articles/cleaned_blocks")
METADATA_DIR = Path("data/articles/articles-metadata")
DOCUMENTS_PREVIEW_DIR = Path("data/articles/documents_preview")

GREEN_CATEGORIES = [
    "lighting_efficiency",
    "occupancy_based_control",
    "hvac_optimization",
    "appliance_energy_management",
    "environmental_awareness",
    "comfort_security",
    "water_saving",
    "scheduling_optimization",
    "standby_reduction",
]

# Metadata that should not be inserted into the text sent to the
# embedding model.
EXCLUDED_EMBED_METADATA_KEYS = [
    "article_id",
    "original_article_id",
    "source_url",
    "article_summary",
    "source_type",
    "source_file",
    "source_blocks_file",
    "content_type",
    "block_count",
    "section_count",
    "section_titles",
    "source_pages",
    *GREEN_CATEGORIES,
]

# Metadata that should not be inserted into prompts sent to the LLM.
# The source title, year, and URL remain available.
EXCLUDED_LLM_METADATA_KEYS = [
    "article_summary",
    "source_file",
    "source_blocks_file",
    "block_count",
    "section_count",
    "section_titles",
    "source_pages",
    *GREEN_CATEGORIES,
]


Block = dict[str, Any]


def normalize_binary_value(value: Any) -> int:
    """
    Convert common binary representations to 0 or 1.
    """
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return 1 if value != 0 else 0

    if isinstance(value, float):
        return 1 if value != 0.0 else 0

    if isinstance(value, str):
        normalized_value = value.strip().casefold()

        if normalized_value in {
            "1",
            "true",
            "yes",
            "y",
            "present",
        }:
            return 1

    return 0


def make_metadata_vector_store_safe(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert nested metadata values into strings.

    Many vector stores handle primitive metadata values more reliably
    than nested dictionaries or lists.
    """
    normalized_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (dict, list, tuple, set)):
            normalized_metadata[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            normalized_metadata[key] = value

    return normalized_metadata


def read_jsonl(path: Path) -> list[Block]:
    """
    Read a JSONL file containing cleaned TEI blocks.
    """
    blocks: list[Block] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                block = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error

            if not isinstance(block, dict):
                raise ValueError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )

            blocks.append(block)

    if not blocks:
        raise ValueError(f"No blocks found in: {path}")

    return blocks


def normalize_section_path(value: Any) -> list[str]:
    """
    Normalize section_path values from the cleaned block JSONL.
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(">") if part.strip()]

    return []


def get_block_sort_key(block: Block) -> tuple[int, int]:
    """
    Sort blocks by their stable indexes when available.
    """
    block_index = block.get("block_index")
    source_block_index = block.get("source_block_index")

    try:
        primary = int(block_index)
    except (TypeError, ValueError):
        primary = 10**9

    try:
        secondary = int(source_block_index)
    except (TypeError, ValueError):
        secondary = 10**9

    return primary, secondary


def build_article_text_from_blocks(blocks: list[Block]) -> str:
    """
    Build a readable article-level text from cleaned TEI blocks.

    The text is not Markdown-dependent. Section labels are injected from
    section_path so that the article context remains visible even though
    Markdown files are no longer part of the pipeline.
    """
    sorted_blocks = sorted(blocks, key=get_block_sort_key)
    lines: list[str] = []
    current_section_path: tuple[str, ...] | None = None

    for block in sorted_blocks:
        kind = str(block.get("kind", "")).casefold()
        text = str(block.get("text", "")).strip()

        if not text:
            continue

        # We inject section labels from section_path when the first real
        # content block of a section is encountered. This avoids depending
        # on Markdown headings or heading blocks.
        section_path = tuple(
            normalize_section_path(block.get("section_path"))
        )

        if kind == "heading":
            continue

        if section_path != current_section_path:
            if section_path:
                lines.append("")
                lines.append("Section: " + " > ".join(section_path))
                lines.append("")
            current_section_path = section_path

        if kind == "quote":
            lines.append(f"Quote: {text}")
        elif kind == "formula":
            lines.append(f"Formula: {text}")
        else:
            lines.append(text)

        lines.append("")

    article_text = "\n".join(lines).strip()

    if not article_text:
        raise ValueError("The cleaned blocks did not contain usable text.")

    return article_text


def collect_section_titles(blocks: list[Block]) -> list[str]:
    """
    Collect unique section titles from the cleaned blocks.
    """
    titles: list[str] = []

    for block in sorted(blocks, key=get_block_sort_key):
        section_title = block.get("section_title")

        if not section_title:
            section_path = normalize_section_path(
                block.get("section_path")
            )
            section_title = section_path[-1] if section_path else ""

        section_title = str(section_title).strip()

        if section_title and section_title not in titles:
            titles.append(section_title)

    return titles


def collect_source_pages(blocks: list[Block]) -> list[int]:
    """
    Collect unique source pages from block metadata when available.
    """
    pages: set[int] = set()

    for block in blocks:
        value = block.get("pages", block.get("source_pages", []))

        if isinstance(value, int):
            pages.add(value)
            continue

        if isinstance(value, str):
            try:
                pages.add(int(value))
            except ValueError:
                continue
            continue

        if isinstance(value, list):
            for item in value:
                try:
                    pages.add(int(item))
                except (TypeError, ValueError):
                    continue

    return sorted(pages)


def infer_source_title(
    blocks: list[Block],
    metadata: dict[str, Any],
) -> str:
    """
    Infer the article title from metadata or cleaned blocks.
    """
    if metadata.get("source_title"):
        return str(metadata["source_title"])

    if metadata.get("title"):
        return str(metadata["title"])

    for block in blocks:
        source_title = block.get("source_title")

        if source_title:
            return str(source_title)

    return ""


def load_metadata(
    article_id: str,
    blocks: list[Block],
    blocks_path: Path,
) -> dict[str, Any]:
    metadata_path = METADATA_DIR / f"{article_id}.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Metadata must be a JSON object: {metadata_path}"
        )

    # Some metadata files store the green labels in a nested object.
    nested_green_categories = metadata.pop(
        "normalized_green_categories_one_hot",
        {},
    )

    if not isinstance(nested_green_categories, dict):
        nested_green_categories = {}

    for category in GREEN_CATEGORIES:
        value = metadata.get(
            category,
            nested_green_categories.get(category, 0),
        )

        metadata[category] = normalize_binary_value(value)

    # Use the filename as the canonical article identifier.
    original_article_id = metadata.get("article_id")

    if original_article_id and original_article_id != article_id:
        metadata["original_article_id"] = str(original_article_id)

    metadata["article_id"] = article_id
    metadata["content_type"] = "main"
    metadata["source_blocks_file"] = str(blocks_path)
    metadata["source_file"] = str(blocks_path)
    metadata["block_count"] = len(blocks)

    source_title = infer_source_title(
        blocks=blocks,
        metadata=metadata,
    )

    if source_title:
        metadata["source_title"] = source_title

    section_titles = collect_section_titles(blocks)
    metadata["section_titles"] = section_titles
    metadata["section_count"] = len(section_titles)

    source_pages = collect_source_pages(blocks)

    if source_pages:
        metadata["source_pages"] = source_pages

    # Convert the year to an integer when possible.
    year = metadata.get("year")

    if year is not None and str(year).isdigit():
        metadata["year"] = int(year)

    return make_metadata_vector_store_safe(metadata)


def find_cleaned_block_files(
    article: str | None = None,
) -> list[Path]:
    if article:
        article = article.strip()

        if article.lower().endswith(".jsonl"):
            article = article[:-6]

        if not article.startswith("article_"):
            article = f"article_{article}"

        path = BLOCKS_DIR / f"{article}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                f"Cleaned blocks file not found: {path}"
            )

        return [path]

    block_files = sorted(BLOCKS_DIR.glob("article_*.jsonl"))

    if not block_files:
        raise FileNotFoundError(
            f"No cleaned JSONL block files found in: {BLOCKS_DIR}"
        )

    return block_files


def build_documents(
    article: str | None = None,
) -> list[Document]:
    documents: list[Document] = []

    for blocks_path in find_cleaned_block_files(article=article):
        article_id = blocks_path.stem
        blocks = read_jsonl(blocks_path)
        article_text = build_article_text_from_blocks(blocks)

        metadata = load_metadata(
            article_id=article_id,
            blocks=blocks,
            blocks_path=blocks_path,
        )

        excluded_embed_keys = [
            key
            for key in EXCLUDED_EMBED_METADATA_KEYS
            if key in metadata
        ]

        excluded_llm_keys = [
            key
            for key in EXCLUDED_LLM_METADATA_KEYS
            if key in metadata
        ]

        document = Document(
            text=article_text,
            metadata=metadata,
            id_=article_id,
            excluded_embed_metadata_keys=excluded_embed_keys,
            excluded_llm_metadata_keys=excluded_llm_keys,
        )

        documents.append(document)

    return documents


def save_documents_preview(
    documents: list[Document],
) -> None:
    preview = []

    for document in documents:
        preview.append(
            {
                "document_id": document.id_,
                "metadata": document.metadata,
                "excluded_embed_metadata_keys": (
                    document.excluded_embed_metadata_keys
                ),
                "excluded_llm_metadata_keys": (
                    document.excluded_llm_metadata_keys
                ),
                "text_preview": document.text[:2000],
                "text_characters": len(document.text),
                "word_count": len(document.text.split()),
            }
        )

    output_path = DOCUMENTS_PREVIEW_DIR / "documents_preview.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(preview, file, indent=2, ensure_ascii=False)

    print(f"Documents preview saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build LlamaIndex Document objects from cleaned TEI JSONL "
            "blocks. Markdown files are not used."
        )
    )

    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help="Article number or name, for example: 2 or article_2.",
    )

    args = parser.parse_args()

    documents = build_documents(article=args.article)
    save_documents_preview(documents)

    print(f"Documents created: {len(documents)}")

    if documents:
        first_document = documents[0]

        print("\nFirst document ID:")
        print(first_document.id_)

        print("\nFirst document metadata:")
        print(
            json.dumps(
                first_document.metadata,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nFirst document text preview:")
        print(first_document.text[:1000])


if __name__ == "__main__":
    main()
