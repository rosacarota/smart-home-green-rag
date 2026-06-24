from __future__ import annotations

from typing import Any

from llama_index.core.schema import MetadataMode, NodeWithScore

from generation.rag_pipeline.config import TEXT_PREVIEW_CHARS


def get_node_text(result: NodeWithScore) -> str:
    """
    Return only the node text, without injecting metadata into the content.
    """
    return result.node.get_content(
        metadata_mode=MetadataMode.NONE
    ).strip()


def result_to_dict(
    result: NodeWithScore,
    rank: int,
    max_preview_chars: int = TEXT_PREVIEW_CHARS,
) -> dict[str, Any]:
    """
    Convert one retrieved node into a JSON-serializable dictionary.
    """
    node = result.node
    metadata = dict(node.metadata)
    text = get_node_text(result)

    return {
        "rank": rank,
        "score": result.score,
        "node_id": node.node_id,
        "content_type": metadata.get("content_type", ""),

        # Rule metadata
        "rule_id": metadata.get("rule_id", ""),
        "llm_topic_name": metadata.get("llm_topic_name", ""),
        "green_category": metadata.get("green_category", ""),

        # Article metadata
        "article_id": metadata.get("article_id", ""),
        "source_title": metadata.get("source_title", ""),
        "section_title": metadata.get("section_title", ""),
        "section_path": metadata.get("section_path", ""),
        "green_categories": metadata.get("green_categories", ""),
        "dominant_green_category": metadata.get("dominant_green_category", ""),

        # Text
        "text_length": len(text),
        "text": text,
        "text_preview": text[:max_preview_chars],
    }