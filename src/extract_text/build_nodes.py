from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import json
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode


CLEANED_BLOCKS_DIR = Path("data/articles/cleaned_blocks")

OUTPUT_DIR = Path("data/articles/nodes_preview")
OUTPUT_PATH = OUTPUT_DIR / "section_sentence_nodes_preview.json"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

SECTION_PATH_SEPARATOR = " > "
SECTION_TEXT_SEPARATOR = "\n\n"

TEXT_BLOCK_KINDS = {
    "abstract",
    "paragraph",
    "list_item",
    "quote",
    "formula",
}

# These metadata are useful for provenance/filtering, but should not be
# injected into the embedding text or into the LLM context as prose.
SECTION_OPERATIONAL_METADATA_KEYS = [
    "article_id",
    "section_id",
    "section_index",
    "section_depth",
    "source_tei",
    "source_block_start",
    "source_block_end",
    "source_block_indices",
    "source_pages",
    "first_page",
    "last_page",
]

CHUNK_OPERATIONAL_METADATA_KEYS = [
    "chunk_index",
    "chunk_index_in_article",
    "chunk_index_in_section",
    "total_chunks_in_article",
    "total_chunks_in_section",
    "chunking_strategy",
    "chunk_size_tokens",
    "chunk_overlap_tokens",
    "previous_chunk_id",
    "next_chunk_id",
]


def merge_unique(
    current_values: list[str],
    new_values: list[str],
) -> list[str]:
    """
    Merge two lists without introducing duplicates.
    """
    result = list(current_values)

    for value in new_values:
        if value not in result:
            result.append(value)

    return result


def normalize_article_name(article: str) -> str:
    article = article.strip()

    if article.lower().endswith(".jsonl"):
        article = article[:-6]

    if article.startswith("article_"):
        return article

    return f"article_{article}"


def get_node_text(node) -> str:
    """
    Return node text without metadata.
    """
    return node.get_content(
        metadata_mode=MetadataMode.NONE,
    ).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file and return a list of dictionaries.
    """
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )

            records.append(record)

    return records


def find_cleaned_block_files(
    article: str | None = None,
    input_dir: Path = CLEANED_BLOCKS_DIR,
) -> list[Path]:
    """
    Return cleaned block JSONL files produced by clean_grobid_tei.py.
    """
    if article:
        article_id = normalize_article_name(article)
        path = input_dir / f"{article_id}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                f"Cleaned blocks file not found: {path}"
            )

        return [path]

    files = sorted(input_dir.glob("article_*.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No cleaned block JSONL files found in: {input_dir}"
        )

    return files


def load_cleaned_blocks(
    article: str | None = None,
    input_dir: Path = CLEANED_BLOCKS_DIR,
) -> list[dict[str, Any]]:
    """
    Load main cleaned blocks from data/articles/cleaned_blocks/*.jsonl.
    """
    records: list[dict[str, Any]] = []

    for path in find_cleaned_block_files(
        article=article,
        input_dir=input_dir,
    ):
        records.extend(read_jsonl(path))

    records.sort(
        key=lambda record: (
            str(record.get("article_id", "")),
            int(record.get("block_index", 0) or 0),
        )
    )

    return records


def as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default

    return str(value).strip()


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_section_path(record: dict[str, Any]) -> list[str]:
    section_path = record.get("section_path")

    if isinstance(section_path, list):
        parts = [as_text(part) for part in section_path]
        parts = [part for part in parts if part]

        if parts:
            return parts

    section_title = as_text(record.get("section_title"))

    if section_title:
        return [section_title]

    return ["Untitled section"]


def build_header_path(
    source_title: str,
    section_path: list[str],
) -> str:
    header_parts = [source_title]

    for section_part in section_path:
        if section_part and section_part not in header_parts:
            header_parts.append(section_part)

    return SECTION_PATH_SEPARATOR.join(header_parts)


def collect_pages(records: list[dict[str, Any]]) -> list[int]:
    pages: set[int] = set()

    for record in records:
        record_pages = record.get("pages", [])

        if not isinstance(record_pages, list):
            continue

        for page in record_pages:
            try:
                pages.add(int(page))
            except (TypeError, ValueError):
                continue

    return sorted(pages)


def build_section_text(records: list[dict[str, Any]]) -> str:
    """
    Build the text of a section from non-heading cleaned blocks.
    """
    text_parts: list[str] = []

    for record in records:
        kind = as_text(record.get("kind"))

        if kind not in TEXT_BLOCK_KINDS:
            continue

        text = as_text(record.get("text"))

        if text:
            text_parts.append(text)

    return SECTION_TEXT_SEPARATOR.join(text_parts).strip()


def group_blocks_by_section(
    records: list[dict[str, Any]],
) -> list[tuple[tuple[str, tuple[str, ...]], list[dict[str, Any]]]]:
    """
    Group cleaned block records by article and section path.

    The order is determined by the first block index of each section, so the
    original article flow is preserved.
    """
    grouped: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    first_position: dict[tuple[str, tuple[str, ...]], tuple[str, int]] = {}

    for record in records:
        article_id = as_text(record.get("article_id"), "unknown_article")
        section_path = normalize_section_path(record)
        key = (article_id, tuple(section_path))

        grouped[key].append(record)

        position = (
            article_id,
            as_int(record.get("block_index"), default=0),
        )

        if key not in first_position or position < first_position[key]:
            first_position[key] = position

    return sorted(
        grouped.items(),
        key=lambda item: first_position[item[0]],
    )


def build_section_documents(
    records: list[dict[str, Any]] | None = None,
    article: str | None = None,
    input_dir: Path = CLEANED_BLOCKS_DIR,
) -> list[Document]:
    """
    Build one LlamaIndex Document for each cleaned TEI section.

    This replaces the old MarkdownNodeParser step. The input is now the JSONL
    produced by clean_grobid_tei.py.
    """
    if records is None:
        records = load_cleaned_blocks(
            article=article,
            input_dir=input_dir,
        )

    section_documents: list[Document] = []
    article_section_counters: dict[str, int] = defaultdict(int)
    skipped_empty_sections = 0

    for (article_id, section_path_tuple), section_records in group_blocks_by_section(records):
        section_text = build_section_text(section_records)

        if not section_text:
            skipped_empty_sections += 1
            continue

        section_path = list(section_path_tuple)
        section_title = section_path[-1] if section_path else "Untitled section"
        source_title = as_text(
            section_records[0].get("source_title"),
            default=article_id,
        )
        source_tei = as_text(section_records[0].get("source_tei"))
        section_depth = len(section_path)
        header_path = build_header_path(
            source_title=source_title,
            section_path=section_path,
        )

        article_section_counters[article_id] += 1
        section_index = article_section_counters[article_id]
        section_id = f"{article_id}__section_{section_index:03d}"

        block_indices = [
            as_int(record.get("block_index"), default=0)
            for record in section_records
            if as_int(record.get("block_index"), default=0) > 0
        ]
        pages = collect_pages(section_records)

        metadata: dict[str, Any] = {
            "article_id": article_id,
            "source_title": source_title,
            "source_tei": source_tei,
            "section_id": section_id,
            "section_index": section_index,
            "section_title": section_title,
            "section_path": SECTION_PATH_SEPARATOR.join(section_path),
            "section_depth": section_depth,
            "header_path": header_path,
            "source_block_start": min(block_indices) if block_indices else 0,
            "source_block_end": max(block_indices) if block_indices else 0,
            "source_block_indices": ",".join(str(index) for index in block_indices),
            "source_pages": ",".join(str(page) for page in pages),
        }

        if pages:
            metadata["first_page"] = pages[0]
            metadata["last_page"] = pages[-1]

        section_document = Document(
            text=section_text,
            metadata=metadata,
            id_=section_id,
            excluded_embed_metadata_keys=SECTION_OPERATIONAL_METADATA_KEYS,
            excluded_llm_metadata_keys=SECTION_OPERATIONAL_METADATA_KEYS,
        )

        section_documents.append(section_document)

    print(f"Empty sections skipped: {skipped_empty_sections}")

    return section_documents


def build_nodes(
    records: list[dict[str, Any]] | None = None,
    article: str | None = None,
    input_dir: Path = CLEANED_BLOCKS_DIR,
):
    """
    Split cleaned TEI sections into sentence-aware chunks.
    """
    section_documents = build_section_documents(
        records=records,
        article=article,
        input_dir=input_dir,
    )

    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        include_metadata=True,
        include_prev_next_rel=True,
    )

    nodes = sentence_splitter.get_nodes_from_documents(
        section_documents,
        show_progress=True,
    )

    article_chunk_counters: dict[str, int] = defaultdict(int)
    section_chunk_counters: dict[str, int] = defaultdict(int)
    nodes_by_article: dict[str, list] = defaultdict(list)
    nodes_by_section: dict[str, list] = defaultdict(list)

    operational_metadata_keys = [
        *CHUNK_OPERATIONAL_METADATA_KEYS,
    ]

    for global_index, node in enumerate(nodes, start=1):
        article_id = as_text(
            node.metadata.get("article_id"),
            "unknown_article",
        )
        section_id = as_text(
            node.metadata.get("section_id"),
            "unknown_section",
        )

        article_chunk_counters[article_id] += 1
        section_chunk_counters[section_id] += 1

        chunk_index_in_article = article_chunk_counters[article_id]
        chunk_index_in_section = section_chunk_counters[section_id]

        node.metadata.update(
            {
                "chunk_index": global_index,
                "chunk_index_in_article": chunk_index_in_article,
                "chunk_index_in_section": chunk_index_in_section,
                "chunking_strategy": "tei_jsonl_section_sentence",
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
            }
        )

        node.id_ = f"{section_id}__chunk_{chunk_index_in_section:03d}"

        node.excluded_embed_metadata_keys = merge_unique(
            list(node.excluded_embed_metadata_keys),
            operational_metadata_keys,
        )
        node.excluded_llm_metadata_keys = merge_unique(
            list(node.excluded_llm_metadata_keys),
            operational_metadata_keys,
        )

        nodes_by_article[article_id].append(node)
        nodes_by_section[section_id].append(node)

    for article_id, article_nodes in nodes_by_article.items():
        total_chunks_in_article = len(article_nodes)

        for index, node in enumerate(article_nodes):
            previous_chunk_id = article_nodes[index - 1].node_id if index > 0 else ""
            next_chunk_id = (
                article_nodes[index + 1].node_id
                if index < total_chunks_in_article - 1
                else ""
            )

            node.metadata.update(
                {
                    "total_chunks_in_article": total_chunks_in_article,
                    "previous_chunk_id": previous_chunk_id,
                    "next_chunk_id": next_chunk_id,
                }
            )

    for section_id, section_nodes in nodes_by_section.items():
        total_chunks_in_section = len(section_nodes)

        for node in section_nodes:
            node.metadata["total_chunks_in_section"] = total_chunks_in_section

    return nodes, section_documents


def save_nodes_preview(
    nodes,
    section_documents: list[Document],
    output_path: Path = OUTPUT_PATH,
) -> None:
    """
    Save all generated nodes for manual inspection.
    """
    preview = []

    for node_index, node in enumerate(nodes, start=1):
        node_text = get_node_text(node)

        preview.append(
            {
                "node_index": node_index,
                "node_id": node.node_id,
                "article_id": node.metadata.get("article_id"),
                "source_title": node.metadata.get("source_title"),
                "section_id": node.metadata.get("section_id"),
                "section_title": node.metadata.get("section_title"),
                "section_path": node.metadata.get("section_path"),
                "header_path": node.metadata.get("header_path"),
                "source_pages": node.metadata.get("source_pages"),
                "source_block_start": node.metadata.get("source_block_start"),
                "source_block_end": node.metadata.get("source_block_end"),
                "chunk_index_in_article": node.metadata.get("chunk_index_in_article"),
                "chunk_index_in_section": node.metadata.get("chunk_index_in_section"),
                "total_chunks_in_article": node.metadata.get("total_chunks_in_article"),
                "total_chunks_in_section": node.metadata.get("total_chunks_in_section"),
                "previous_chunk_id": node.metadata.get("previous_chunk_id"),
                "next_chunk_id": node.metadata.get("next_chunk_id"),
                "metadata": node.metadata,
                "text": node_text,
                "text_characters": len(node_text),
                "word_count": len(node_text.split()),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            preview,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Section documents created: {len(section_documents)}")
    print(f"Section-sentence nodes created: {len(nodes)}")
    print(f"Nodes preview saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build sentence-aware nodes from cleaned TEI JSONL blocks."
        )
    )

    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help="Article number or name, for example: 2 or article_2.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=CLEANED_BLOCKS_DIR,
        help="Directory containing cleaned_blocks/*.jsonl files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the nodes preview JSON file.",
    )

    args = parser.parse_args()

    records = load_cleaned_blocks(
        article=args.article,
        input_dir=args.input_dir,
    )

    print(f"Cleaned blocks loaded: {len(records)}")

    nodes, section_documents = build_nodes(
        records=records,
        input_dir=args.input_dir,
    )

    save_nodes_preview(
        nodes=nodes,
        section_documents=section_documents,
        output_path=args.output,
    )

    if nodes:
        first_node = nodes[0]

        print("\nFirst node metadata:")
        print(
            json.dumps(
                first_node.metadata,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nFirst node text:")
        print(get_node_text(first_node))


if __name__ == "__main__":
    main()
