from pathlib import Path
from typing import Any
import json

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


ARTICLE_NODES_PREVIEW_PATH = Path(
    "data/articles/nodes_preview/section_sentence_nodes_preview.json"
)
DOCUMENTS_PREVIEW_PATH = Path(
    "data/articles/documents_preview/documents_preview.json"
)
RULE_NODES_PATH = Path(
    "data/dataset-rules/final_dataset/rule_nodes.jsonl"
)

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"
REPORTS_DIR = KNOWLEDGE_BASE_DIR / "reports"
INDEX_REPORT_PATH = REPORTS_DIR / "vector_indexes_report.json"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

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

ARTICLE_EXCLUDED_EMBED_METADATA_KEYS = [
    "article_id",
    "source_tei",
    "source_block_indices",
    "source_block_start",
    "source_block_end",
    "previous_chunk_id",
    "next_chunk_id",
]

ARTICLE_EXCLUDED_LLM_METADATA_KEYS = [
    "source_tei",
    "source_block_indices",
]

RULE_EXCLUDED_EMBED_METADATA_KEYS = [
    "sample_id",
    "rule_id",
    "if_then_rule",
    "original_green_category",
]

RULE_EXCLUDED_LLM_METADATA_KEYS = [
    "sample_id",
    "rule_id",
]


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error

            if isinstance(record, dict):
                records.append(record)

    return records


def normalize_article_id(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    if text.startswith("ARTICLE_"):
        suffix = text.removeprefix("ARTICLE_").lstrip("0") or "0"
        return f"article_{suffix}"

    return text


def make_metadata_vector_store_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (dict, list, tuple, set)):
            safe_metadata[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            safe_metadata[key] = value

    return safe_metadata


def load_document_metadata_map(
    documents_preview_path: Path = DOCUMENTS_PREVIEW_PATH,
) -> dict[str, dict[str, Any]]:
    if not documents_preview_path.exists():
        print(
            f"Warning: documents preview not found, article category metadata will be limited: {documents_preview_path}"
        )
        return {}

    documents = read_json(documents_preview_path)

    if not isinstance(documents, list):
        raise ValueError(
            f"Documents preview must be a list: {documents_preview_path}"
        )

    metadata_map: dict[str, dict[str, Any]] = {}

    for document in documents:
        if not isinstance(document, dict):
            continue

        document_id = normalize_article_id(document.get("document_id"))
        metadata = document.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        article_id = normalize_article_id(
            metadata.get("article_id") or document_id
        )

        if article_id:
            metadata_map[article_id] = metadata

    return metadata_map


def get_article_category_metadata(
    article_id: str,
    document_metadata_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    document_metadata = document_metadata_map.get(article_id, {})
    category_metadata: dict[str, Any] = {}
    active_categories: list[str] = []

    nested_categories = document_metadata.get(
        "normalized_green_categories_one_hot",
        {},
    )

    if not isinstance(nested_categories, dict):
        nested_categories = {}

    for category in GREEN_CATEGORIES:
        value = document_metadata.get(category, nested_categories.get(category, 0))

        try:
            normalized_value = 1 if int(value) != 0 else 0
        except (TypeError, ValueError):
            normalized_value = 1 if str(value).strip().casefold() in {
                "1",
                "true",
                "yes",
                "present",
            } else 0

        category_metadata[category] = normalized_value

        if normalized_value == 1:
            active_categories.append(category)

    category_metadata["green_categories"] = ", ".join(active_categories)
    category_metadata["dominant_green_category"] = (
        active_categories[0] if active_categories else ""
    )

    if document_metadata.get("article_summary"):
        category_metadata["article_summary"] = document_metadata["article_summary"]

    if document_metadata.get("year"):
        category_metadata["year"] = document_metadata["year"]

    if document_metadata.get("source_url"):
        category_metadata["source_url"] = document_metadata["source_url"]

    return category_metadata


def build_article_nodes() -> list[TextNode]:
    article_records = read_json(ARTICLE_NODES_PREVIEW_PATH)

    if not isinstance(article_records, list):
        raise ValueError(
            f"Article nodes preview must be a list: {ARTICLE_NODES_PREVIEW_PATH}"
        )

    document_metadata_map = load_document_metadata_map()
    nodes: list[TextNode] = []

    for record in article_records:
        if not isinstance(record, dict):
            continue

        text = str(record.get("text") or "").strip()
        node_id = str(record.get("node_id") or "").strip()
        article_id = normalize_article_id(record.get("article_id"))

        if not text or not node_id:
            continue

        metadata = {}

        if isinstance(record.get("metadata"), dict):
            metadata.update(record["metadata"])

        metadata.update(
            {
                "content_type": "article_chunk",
                "article_id": article_id,
                "node_id": node_id,
                "source_title": record.get("source_title") or metadata.get("source_title", ""),
                "section_title": record.get("section_title") or metadata.get("section_title", ""),
                "section_path": record.get("section_path") or metadata.get("section_path", ""),
                "header_path": record.get("header_path") or metadata.get("header_path", ""),
                "chunk_index_in_article": record.get("chunk_index_in_article"),
                "chunk_index_in_section": record.get("chunk_index_in_section"),
                "word_count": record.get("word_count"),
            }
        )

        metadata.update(
            get_article_category_metadata(
                article_id=article_id,
                document_metadata_map=document_metadata_map,
            )
        )

        safe_metadata = make_metadata_vector_store_safe(metadata)

        excluded_embed_keys = [
            key for key in ARTICLE_EXCLUDED_EMBED_METADATA_KEYS if key in safe_metadata
        ]
        excluded_llm_keys = [
            key for key in ARTICLE_EXCLUDED_LLM_METADATA_KEYS if key in safe_metadata
        ]

        node = TextNode(
            id_=node_id,
            text=text,
            metadata=safe_metadata,
            excluded_embed_metadata_keys=excluded_embed_keys,
            excluded_llm_metadata_keys=excluded_llm_keys,
        )

        nodes.append(node)

    if not nodes:
        raise ValueError("No article nodes were created.")

    return nodes


def build_rule_nodes() -> list[TextNode]:
    rule_records = read_jsonl(RULE_NODES_PATH)
    nodes: list[TextNode] = []

    for record in rule_records:
        text = str(record.get("text") or "").strip()
        node_id = str(record.get("node_id") or "").strip()
        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        if not text or not node_id:
            continue

        safe_metadata = make_metadata_vector_store_safe(metadata)
        safe_metadata["content_type"] = "green_rule"
        safe_metadata["node_id"] = node_id

        excluded_embed_keys = [
            key for key in RULE_EXCLUDED_EMBED_METADATA_KEYS if key in safe_metadata
        ]
        excluded_llm_keys = [
            key for key in RULE_EXCLUDED_LLM_METADATA_KEYS if key in safe_metadata
        ]

        node = TextNode(
            id_=node_id,
            text=text,
            metadata=safe_metadata,
            excluded_embed_metadata_keys=excluded_embed_keys,
            excluded_llm_metadata_keys=excluded_llm_keys,
        )

        nodes.append(node)

    if not nodes:
        raise ValueError("No rule nodes were created.")

    return nodes


def persist_index(nodes: list[TextNode], persist_dir: Path, embed_model: HuggingFaceEmbedding) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)

    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=str(persist_dir))


def count_categories(nodes: list[TextNode]) -> dict[str, int]:
    counts = {category: 0 for category in GREEN_CATEGORIES}

    for node in nodes:
        for category in GREEN_CATEGORIES:
            value = node.metadata.get(category, 0)
            try:
                if int(value) == 1:
                    counts[category] += 1
            except (TypeError, ValueError):
                continue

    return counts


def main() -> None:
    print(f"Embedding model: {EMBED_MODEL_NAME}")

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )
    Settings.embed_model = embed_model

    print("\nLoading article nodes...")
    article_nodes = build_article_nodes()
    print(f"Article nodes: {len(article_nodes)}")

    print("\nLoading rule nodes...")
    rule_nodes = build_rule_nodes()
    print(f"Rule nodes: {len(rule_nodes)}")

    print("\nBuilding article index...")
    persist_index(
        nodes=article_nodes,
        persist_dir=ARTICLE_INDEX_DIR,
        embed_model=embed_model,
    )
    print(f"Article index saved to: {ARTICLE_INDEX_DIR}")

    print("\nBuilding rule index...")
    persist_index(
        nodes=rule_nodes,
        persist_dir=RULE_INDEX_DIR,
        embed_model=embed_model,
    )
    print(f"Rule index saved to: {RULE_INDEX_DIR}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "embedding_model": EMBED_MODEL_NAME,
        "article_nodes_path": str(ARTICLE_NODES_PREVIEW_PATH),
        "documents_preview_path": str(DOCUMENTS_PREVIEW_PATH),
        "rule_nodes_path": str(RULE_NODES_PATH),
        "article_index_dir": str(ARTICLE_INDEX_DIR),
        "rule_index_dir": str(RULE_INDEX_DIR),
        "article_nodes_indexed": len(article_nodes),
        "rule_nodes_indexed": len(rule_nodes),
        "article_category_counts": count_categories(article_nodes),
        "rule_category_counts": count_categories(rule_nodes),
    }

    with INDEX_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nVector indexes report saved to: {INDEX_REPORT_PATH}")


if __name__ == "__main__":
    main()
