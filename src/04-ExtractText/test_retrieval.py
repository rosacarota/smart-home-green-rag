from pathlib import Path
import argparse

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"

EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"

DEFAULT_QUERY = (
    "IF You exit home THEN Turn off lights."
)


def load_index(persist_dir: Path, embed_model: HuggingFaceEmbedding):
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Index directory not found: {persist_dir}. Run build_vector_indexes.py first."
        )

    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(
        persist_dir=str(persist_dir),
    )

    return load_index_from_storage(
        storage_context=storage_context,
        embed_model=embed_model,
    )


def print_article_results(results) -> None:
    print("\n" + "=" * 90)
    print("ARTICLE RETRIEVAL RESULTS")
    print("=" * 90)

    for rank, result in enumerate(results, start=1):
        node = result.node
        metadata = node.metadata
        text = node.get_content(metadata_mode="none").strip()

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
        text = node.get_content(metadata_mode="none").strip()

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {result.score}")
        print(f"Sample ID: {metadata.get('sample_id', '')}")
        print(f"Topic: {metadata.get('llm_topic_name', '')}")
        print(f"Green category: {metadata.get('green_category', '')}")
        print(f"Rule ID: {node.node_id}")
        print("\nRule text:")
        print(text[:900])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test retrieval over article and green-rule vector indexes."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Input trigger-action rule or natural-language query.",
    )
    parser.add_argument(
        "--top-k-articles",
        type=int,
        default=5,
        help="Number of article chunks to retrieve.",
    )
    parser.add_argument(
        "--top-k-rules",
        type=int,
        default=5,
        help="Number of similar rules to retrieve.",
    )

    args = parser.parse_args()

    print(f"Embedding model: {EMBED_MODEL_NAME}")
    print(f"Query: {args.query}")

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )
    Settings.embed_model = embed_model

    print("\nLoading article index...")
    article_index = load_index(
        persist_dir=ARTICLE_INDEX_DIR,
        embed_model=embed_model,
    )

    print("Loading rule index...")
    rule_index = load_index(
        persist_dir=RULE_INDEX_DIR,
        embed_model=embed_model,
    )

    article_retriever = article_index.as_retriever(
        similarity_top_k=args.top_k_articles,
    )
    rule_retriever = rule_index.as_retriever(
        similarity_top_k=args.top_k_rules,
    )

    article_results = article_retriever.retrieve(args.query)
    rule_results = rule_retriever.retrieve(args.query)

    print_rule_results(rule_results)
    print_article_results(article_results)

    print("\n" + "=" * 90)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()
