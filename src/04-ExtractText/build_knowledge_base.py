from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import shutil
import subprocess
import sys

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import build_documents
import build_nodes
import build_rules_nodes


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"
REPORTS_DIR = KNOWLEDGE_BASE_DIR / "reports"
REPORT_PATH = REPORTS_DIR / "knowledge_base_build_report.json"

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

DEFAULT_QUERY = (
    "IF You exit from home THEN Turn off lights."
)


def run_step(command: list[str]) -> None:
    print("\n" + "=" * 90)
    print("Running:", " ".join(command))
    print("=" * 90)
    subprocess.run(command, check=True)


def normalize_article_id(value: Any) -> str:
    text = str(value or "").strip()

    if not text:
        return ""

    if text.startswith("ARTICLE_"):
        suffix = text.removeprefix("ARTICLE_").lstrip("0") or "0"
        return f"article_{suffix}"

    return text


def normalize_binary_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return 1 if value != 0 else 0

    if isinstance(value, str):
        normalized = value.strip().casefold()
        return 1 if normalized in {"1", "true", "yes", "y", "present"} else 0

    return 0


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


def merge_unique(current_values: list[str], new_values: list[str]) -> list[str]:
    result = list(current_values)

    for value in new_values:
        if value not in result:
            result.append(value)

    return result


def get_node_text(node: TextNode) -> str:
    return node.get_content(metadata_mode=MetadataMode.NONE).strip()


def build_document_metadata_map(article: str | None = None) -> dict[str, dict[str, Any]]:
    """
    Build article-level metadata in memory from cleaned blocks.

    No documents_preview.json is read here. The documents are created only to
    reuse their metadata, especially the green one-hot categories.
    """
    documents = build_documents.build_documents(article=article)
    metadata_map: dict[str, dict[str, Any]] = {}

    for document in documents:
        metadata = dict(document.metadata)
        article_id = normalize_article_id(metadata.get("article_id") or document.id_)

        if article_id:
            metadata_map[article_id] = metadata

    print(f"Article documents built in memory: {len(documents)}")
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
        normalized_value = normalize_binary_value(value)
        category_metadata[category] = normalized_value

        if normalized_value == 1:
            active_categories.append(category)

    category_metadata["green_categories"] = ", ".join(active_categories)
    category_metadata["dominant_green_category"] = (
        active_categories[0] if active_categories else ""
    )

    optional_keys = [
        "article_summary",
        "year",
        "source_url",
        "source_title",
        "source_type",
        "original_article_id",
    ]

    for key in optional_keys:
        if document_metadata.get(key) is not None:
            category_metadata[key] = document_metadata[key]

    return category_metadata


def enrich_article_nodes(
    article_nodes: list[TextNode],
    document_metadata_map: dict[str, dict[str, Any]],
) -> list[TextNode]:
    """
    Add article-level category metadata to article chunks in memory.
    """
    enriched_nodes: list[TextNode] = []

    for node in article_nodes:
        metadata = dict(node.metadata)
        article_id = normalize_article_id(metadata.get("article_id"))

        metadata.update(
            {
                "content_type": "article_chunk",
                "article_id": article_id,
                "node_id": node.node_id,
                "source_title": metadata.get("source_title", ""),
                "section_title": metadata.get("section_title", ""),
                "section_path": metadata.get("section_path", ""),
                "header_path": metadata.get("header_path", ""),
            }
        )

        metadata.update(
            get_article_category_metadata(
                article_id=article_id,
                document_metadata_map=document_metadata_map,
            )
        )

        node.metadata = make_metadata_vector_store_safe(metadata)

        node.excluded_embed_metadata_keys = merge_unique(
            list(node.excluded_embed_metadata_keys),
            [key for key in ARTICLE_EXCLUDED_EMBED_METADATA_KEYS if key in node.metadata],
        )
        node.excluded_llm_metadata_keys = merge_unique(
            list(node.excluded_llm_metadata_keys),
            [key for key in ARTICLE_EXCLUDED_LLM_METADATA_KEYS if key in node.metadata],
        )

        enriched_nodes.append(node)

    return enriched_nodes


def build_article_nodes_in_memory(article: str | None = None) -> list[TextNode]:
    """
    Build article nodes directly from cleaned_blocks JSONL.

    No nodes_preview JSON is read here.
    """
    document_metadata_map = build_document_metadata_map(article=article)

    records = build_nodes.load_cleaned_blocks(article=article)
    print(f"Cleaned article blocks loaded: {len(records)}")

    article_nodes, section_documents = build_nodes.build_nodes(records=records)
    print(f"Section documents built in memory: {len(section_documents)}")
    print(f"Article nodes built in memory: {len(article_nodes)}")

    return enrich_article_nodes(
        article_nodes=list(article_nodes),
        document_metadata_map=document_metadata_map,
    )


def prepare_rule_nodes(rule_nodes: list[TextNode]) -> list[TextNode]:
    prepared_nodes: list[TextNode] = []

    for node in rule_nodes:
        metadata = make_metadata_vector_store_safe(dict(node.metadata))
        metadata["content_type"] = "green_rule"
        metadata["node_id"] = node.node_id
        node.metadata = metadata

        node.excluded_embed_metadata_keys = merge_unique(
            list(node.excluded_embed_metadata_keys),
            [key for key in RULE_EXCLUDED_EMBED_METADATA_KEYS if key in node.metadata],
        )
        node.excluded_llm_metadata_keys = merge_unique(
            list(node.excluded_llm_metadata_keys),
            [key for key in RULE_EXCLUDED_LLM_METADATA_KEYS if key in node.metadata],
        )

        prepared_nodes.append(node)

    return prepared_nodes


def build_rule_nodes_in_memory() -> list[TextNode]:
    """
    Build rule nodes directly from the normalized rules JSON.

    No rule_nodes.jsonl is read or written here.
    """
    rule_nodes = build_rules_nodes.build_rule_nodes()
    rule_nodes = prepare_rule_nodes(rule_nodes)
    print(f"Rule nodes built in memory: {len(rule_nodes)}")
    return rule_nodes


def reset_persist_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def build_index(
    nodes: list[TextNode],
    persist_dir: Path,
    embed_model: HuggingFaceEmbedding,
    persist: bool,
    overwrite: bool,
) -> VectorStoreIndex:
    if not nodes:
        raise ValueError(f"No nodes provided for index: {persist_dir}")

    print(f"Building index with {len(nodes)} nodes...")
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        show_progress=True,
    )

    if persist:
        reset_persist_dir(persist_dir, overwrite=overwrite)
        index.storage_context.persist(persist_dir=str(persist_dir))
        print(f"Index persisted to: {persist_dir}")
    else:
        print("Index kept only in memory. It will disappear when the script ends.")

    return index


def count_categories(nodes: list[TextNode]) -> dict[str, int]:
    counts = {category: 0 for category in GREEN_CATEGORIES}

    for node in nodes:
        for category in GREEN_CATEGORIES:
            if normalize_binary_value(node.metadata.get(category, 0)) == 1:
                counts[category] += 1

    return counts


def write_report(
    article_nodes: list[TextNode],
    rule_nodes: list[TextNode],
    persist: bool,
    report_path: Path = REPORT_PATH,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "embedding_model": EMBED_MODEL_NAME,
        "article_nodes_indexed": len(article_nodes),
        "rule_nodes_indexed": len(rule_nodes),
        "article_index_dir": str(ARTICLE_INDEX_DIR) if persist else None,
        "rule_index_dir": str(RULE_INDEX_DIR) if persist else None,
        "article_category_counts": count_categories(article_nodes),
        "rule_category_counts": count_categories(rule_nodes),
        "intermediate_previews_used": False,
        "documents_built_in_memory": True,
        "article_nodes_built_in_memory": True,
        "rule_nodes_built_in_memory": True,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Build report saved to: {report_path}")


def print_article_results(results) -> None:
    print("\n" + "=" * 90)
    print("ARTICLE RETRIEVAL RESULTS")
    print("=" * 90)

    for rank, result in enumerate(results, start=1):
        node = result.node
        metadata = node.metadata
        text = get_node_text(node)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {result.score}")
        print(f"Article: {metadata.get('source_title', '')}")
        print(f"Article ID: {metadata.get('article_id', '')}")
        print(f"Section: {metadata.get('section_title', '')}")
        print(f"Green categories: {metadata.get('green_categories', '')}")
        print(f"Chunk ID: {node.node_id}")
        print("\nText preview:")
        print(text[:900])


def print_rule_results(results) -> None:
    print("\n" + "=" * 90)
    print("RULE RETRIEVAL RESULTS")
    print("=" * 90)

    for rank, result in enumerate(results, start=1):
        node = result.node
        metadata = node.metadata
        text = get_node_text(node)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {result.score}")
        print(f"Sample ID: {metadata.get('sample_id', '')}")
        print(f"Topic: {metadata.get('llm_topic_name', '')}")
        print(f"Green category: {metadata.get('green_category', '')}")
        print(f"Rule ID: {node.node_id}")
        print("\nRule text:")
        print(text[:900])


def run_retrieval_test(
    article_index: VectorStoreIndex,
    rule_index: VectorStoreIndex,
    query: str,
    top_k_articles: int,
    top_k_rules: int,
) -> None:
    print("\n" + "=" * 90)
    print("RUNNING IN-MEMORY RETRIEVAL TEST")
    print("=" * 90)
    print(f"Query: {query}")

    article_retriever = article_index.as_retriever(
        similarity_top_k=top_k_articles,
    )
    rule_retriever = rule_index.as_retriever(
        similarity_top_k=top_k_rules,
    )

    rule_results = rule_retriever.retrieve(query)
    article_results = article_retriever.retrieve(query)

    print_rule_results(rule_results)
    print_article_results(article_results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the article and rule knowledge base in one run. "
            "Documents, article nodes, and rule nodes are created in memory; "
            "only the final vector indexes are persisted by default."
        )
    )

    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help="Optional article number/name for a quick single-article build, e.g. 2 or article_2.",
    )
    parser.add_argument(
        "--run-extraction",
        action="store_true",
        help="Run extract_grobid.py before building the KB. Requires GROBID running.",
    )
    parser.add_argument(
        "--run-cleaning",
        action="store_true",
        help="Run clean_grobid_tei.py before building the KB.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Keep indexes only in RAM. Useful for testing, but the KB disappears when the script ends.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing persisted index folders before writing the new ones.",
    )
    parser.add_argument(
        "--test-retrieval",
        action="store_true",
        help="Run a retrieval test immediately using the in-memory indexes.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Rule or natural-language query for the optional retrieval test.",
    )
    parser.add_argument(
        "--top-k-articles",
        type=int,
        default=5,
        help="Number of article chunks to retrieve in the optional test.",
    )
    parser.add_argument(
        "--top-k-rules",
        type=int,
        default=5,
        help="Number of similar rules to retrieve in the optional test.",
    )

    args = parser.parse_args()

    if args.run_extraction:
        if args.article:
            run_step([sys.executable, "src/04-ExtractText/extract_grobid.py", "--article", args.article])
        else:
            run_step([sys.executable, "src/04-ExtractText/extract_grobid.py", "--all"])

    if args.run_cleaning:
        if args.article:
            run_step([sys.executable, "src/04-ExtractText/clean_grobid_tei.py", "--article", args.article])
        else:
            run_step([sys.executable, "src/04-ExtractText/clean_grobid_tei.py", "--all"])

    print("\n" + "=" * 90)
    print("BUILDING KNOWLEDGE BASE IN MEMORY")
    print("=" * 90)
    print(f"Embedding model: {EMBED_MODEL_NAME}")

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )
    Settings.embed_model = embed_model

    print("\nBuilding article nodes directly in memory...")
    article_nodes = build_article_nodes_in_memory(article=args.article)

    print("\nBuilding rule nodes directly in memory...")
    rule_nodes = build_rule_nodes_in_memory()

    persist = not args.no_persist

    print("\nBuilding article vector index...")
    article_index = build_index(
        nodes=article_nodes,
        persist_dir=ARTICLE_INDEX_DIR,
        embed_model=embed_model,
        persist=persist,
        overwrite=args.overwrite,
    )

    print("\nBuilding rule vector index...")
    rule_index = build_index(
        nodes=rule_nodes,
        persist_dir=RULE_INDEX_DIR,
        embed_model=embed_model,
        persist=persist,
        overwrite=args.overwrite,
    )

    if persist:
        write_report(
            article_nodes=article_nodes,
            rule_nodes=rule_nodes,
            persist=persist,
        )

    if args.test_retrieval:
        run_retrieval_test(
            article_index=article_index,
            rule_index=rule_index,
            query=args.query,
            top_k_articles=args.top_k_articles,
            top_k_rules=args.top_k_rules,
        )

    print("\n" + "=" * 90)
    print("KNOWLEDGE BASE BUILD COMPLETED")
    print("=" * 90)

    if persist:
        print(f"Article index: {ARTICLE_INDEX_DIR}")
        print(f"Rule index: {RULE_INDEX_DIR}")
    else:
        print("Indexes were not persisted. They existed only during this script run.")


if __name__ == "__main__":
    main()
