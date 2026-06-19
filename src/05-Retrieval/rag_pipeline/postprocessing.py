from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any
import re

from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

from rag_pipeline.config import (
    FINAL_ARTICLES_K,
    FINAL_RULES_K,
    MIN_ARTICLE_CHARS,
    RERANK_MODEL_NAME,
    RULE_DEDUP_SIMILARITY_THRESHOLD,
)
from rag_pipeline.node_utils import (
    get_node_text,
    result_to_dict,
)


# =============================================================================
# RULE POSTPROCESSING
# =============================================================================

def normalize_text_for_dedup(text: str) -> str:
    """
    Normalize text to make rule deduplication more robust.

    Example:
        "Turn off Hue lights." -> "turn off hue lights"
    """
    text = text.casefold()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_rule_line(text: str) -> str:
    """
    Extract only the actual rule from a rule node.

    Example node text:
        Green category: lighting_efficiency
        Topic: automated_light_control
        Rule: IF You exit an area from Location THEN Turn Hue lights off.

    Output:
        IF You exit an area from Location THEN Turn Hue lights off.
    """
    for line in text.splitlines():
        normalized_line = line.strip()

        if normalized_line.casefold().startswith("rule:"):
            return normalized_line.split(":", maxsplit=1)[1].strip()

    return text.strip()


def are_texts_similar(text_a: str, text_b: str) -> bool:
    """
    Detect near-duplicate rules using normalized text similarity.
    """
    normalized_a = normalize_text_for_dedup(text_a)
    normalized_b = normalize_text_for_dedup(text_b)

    similarity = SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()

    return similarity >= RULE_DEDUP_SIMILARITY_THRESHOLD


def postprocess_rules(
    rule_results: list[NodeWithScore],
) -> dict[str, list[dict[str, Any]]]:
    """
    Deduplicate retrieved rules and keep the final top-k rules.

    Input:
        raw retrieved rule nodes from rule_index.

    Output:
        {
            "selected": [...],
            "discarded": [...]
        }
    """
    unique_rules: list[tuple[NodeWithScore, int, str]] = []
    discarded: list[dict[str, Any]] = []

    for rank, result in enumerate(rule_results, start=1):
        text = get_node_text(result)
        rule_text = extract_rule_line(text)

        is_duplicate = False
        duplicate_of_rank = None

        for _, selected_rank, selected_rule_text in unique_rules:
            if are_texts_similar(rule_text, selected_rule_text):
                is_duplicate = True
                duplicate_of_rank = selected_rank
                break

        if is_duplicate:
            item = result_to_dict(result, rank=rank)
            item["original_rank"] = rank
            item["original_score"] = item.pop("score")
            item["rerank_score"] = None
            item["discard_reason"] = "duplicate_rule"
            item["duplicate_of_original_rank"] = duplicate_of_rank
            discarded.append(item)
        else:
            unique_rules.append((result, rank, rule_text))

    selected: list[dict[str, Any]] = []

    for final_rank, (result, original_rank, _) in enumerate(
        unique_rules[:FINAL_RULES_K],
        start=1,
    ):
        item = result_to_dict(result, rank=final_rank)
        item["original_rank"] = original_rank
        item["original_score"] = item.pop("score")
        item["rerank_score"] = None
        item["final_rank"] = final_rank
        selected.append(item)

    for result, original_rank, _ in unique_rules[FINAL_RULES_K:]:
        item = result_to_dict(result, rank=original_rank)
        item["original_rank"] = original_rank
        item["original_score"] = item.pop("score")
        item["rerank_score"] = None
        item["discard_reason"] = "beyond_final_rules_k_after_dedup"
        discarded.append(item)

    return {
        "selected": selected,
        "discarded": discarded,
    }


# =============================================================================
# ARTICLE POSTPROCESSING
# =============================================================================

def remove_short_article_chunks(
    article_results: list[NodeWithScore],
) -> tuple[list[tuple[NodeWithScore, int]], list[dict[str, Any]]]:
    """
    Remove article chunks that are too short to be useful as generation context.

    This is a general filter, not specific to a domain.
    """
    kept: list[tuple[NodeWithScore, int]] = []
    discarded: list[dict[str, Any]] = []

    for rank, result in enumerate(article_results, start=1):
        text = get_node_text(result)

        if len(text) < MIN_ARTICLE_CHARS:
            item = result_to_dict(result, rank=rank)
            item["original_rank"] = rank
            item["original_score"] = item.pop("score")
            item["rerank_score"] = None
            item["discard_reason"] = "too_short_article_chunk"
            discarded.append(item)
        else:
            kept.append((result, rank))

    return kept, discarded


def rerank_articles(
    query: str,
    article_candidates: list[tuple[NodeWithScore, int]],
) -> list[tuple[NodeWithScore, int, float]]:
    """
    Rerank article chunks using a cross-encoder.

    The cross-encoder receives pairs:
        (query, article_chunk_text)

    and returns a relevance score for each pair.
    """
    if not article_candidates:
        return []

    reranker = CrossEncoder(RERANK_MODEL_NAME)

    pairs = [
        (query, get_node_text(result))
        for result, _ in article_candidates
    ]

    rerank_scores = reranker.predict(pairs)

    reranked = [
        (result, original_rank, float(rerank_score))
        for (result, original_rank), rerank_score in zip(
            article_candidates,
            rerank_scores,
        )
    ]

    reranked.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return reranked


def postprocess_articles(
    query: str,
    article_results: list[NodeWithScore],
) -> dict[str, list[dict[str, Any]]]:
    """
    Remove short chunks, rerank the remaining article chunks,
    and keep the final top-k article chunks.
    """
    article_candidates, discarded = remove_short_article_chunks(
        article_results
    )

    reranked_articles = rerank_articles(
        query=query,
        article_candidates=article_candidates,
    )

    selected: list[dict[str, Any]] = []

    for final_rank, (result, original_rank, rerank_score) in enumerate(
        reranked_articles[:FINAL_ARTICLES_K],
        start=1,
    ):
        item = result_to_dict(result, rank=final_rank)
        item["original_rank"] = original_rank
        item["original_score"] = item.pop("score")
        item["rerank_rank"] = final_rank
        item["rerank_score"] = rerank_score
        item["final_rank"] = final_rank
        selected.append(item)

    for rerank_rank, (result, original_rank, rerank_score) in enumerate(
        reranked_articles[FINAL_ARTICLES_K:],
        start=FINAL_ARTICLES_K + 1,
    ):
        item = result_to_dict(result, rank=original_rank)
        item["original_rank"] = original_rank
        item["original_score"] = item.pop("score")
        item["rerank_rank"] = rerank_rank
        item["rerank_score"] = rerank_score
        item["discard_reason"] = "beyond_final_articles_k_after_rerank"
        discarded.append(item)

    return {
        "selected": selected,
        "discarded": discarded,
    }


# =============================================================================
# FULL POSTPROCESSING PIPELINE
# =============================================================================

def postprocess_context(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
) -> dict[str, Any]:
    """
    Apply the full postprocessing pipeline.

    Rules:
        - textual deduplication
        - keep final top-k rules

    Articles:
        - remove short chunks
        - rerank with cross-encoder
        - keep final top-k article chunks
    """
    processed_rules = postprocess_rules(
        retrieved["rules"]
    )

    processed_articles = postprocess_articles(
        query=query,
        article_results=retrieved["articles"],
    )

    return {
        "rules": processed_rules,
        "articles": processed_articles,
    }