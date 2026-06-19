from __future__ import annotations

from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore

from rag_pipeline.clients import configure_llamaindex
from rag_pipeline.config import (
    ARTICLE_INDEX_DIR,
    RULE_INDEX_DIR,
    TOP_K_ARTICLES,
    TOP_K_RULES,
)


def validate_index_dir(index_dir: Path, index_name: str) -> None:
    """
    Check that the persisted index folder exists before trying to load it.
    """
    if not index_dir.exists():
        raise FileNotFoundError(
            f"{index_name} index directory not found: {index_dir}\n"
            "Build the knowledge base first, for example:\n"
            "python build_knowledge_base.py --overwrite"
        )


def load_persisted_index(index_dir: Path, index_name: str):
    """
    Load a persisted LlamaIndex index from disk.
    """
    validate_index_dir(index_dir, index_name=index_name)

    storage_context = StorageContext.from_defaults(
        persist_dir=str(index_dir)
    )

    return load_index_from_storage(storage_context)


def retrieve_context(
    query: str,
    top_k_rules: int = TOP_K_RULES,
    top_k_articles: int = TOP_K_ARTICLES,
) -> dict[str, list[NodeWithScore]]:
    """
    Retrieve relevant context from both indexes.

    rule_index:
        retrieves similar trigger-action rules.

    article_index:
        retrieves relevant article chunks about green/smart-home concepts.
    """
    configure_llamaindex()

    rule_index = load_persisted_index(
        RULE_INDEX_DIR,
        index_name="Rule",
    )

    article_index = load_persisted_index(
        ARTICLE_INDEX_DIR,
        index_name="Article",
    )

    rule_retriever = rule_index.as_retriever(
        similarity_top_k=top_k_rules,
    )

    article_retriever = article_index.as_retriever(
        similarity_top_k=top_k_articles,
    )

    rule_results = rule_retriever.retrieve(query)
    article_results = article_retriever.retrieve(query)

    return {
        "rules": rule_results,
        "articles": article_results,
    }   