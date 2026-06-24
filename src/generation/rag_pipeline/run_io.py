from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json

from llama_index.core.schema import NodeWithScore

from generation.rag_pipeline.config import (
    ARTICLE_INDEX_DIR,
    EMBED_MODEL_NAME,
    FINAL_ARTICLES_K,
    FINAL_RULES_K,
    GENERATION_OUTPUT_FILENAME,
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    MIN_ARTICLE_CHARS,
    OLLAMA_CLOUD_HOST,
    POSTPROCESSED_OUTPUT_FILENAME,
    RERANK_MODEL_NAME,
    RETRIEVAL_OUTPUT_FILENAME,
    RETRIEVAL_RUNS_DIR,
    RULE_DEDUP_SIMILARITY_THRESHOLD,
    RULE_INDEX_DIR,
    RUN_DIR_PATTERN,
    RUN_ID_WIDTH,
    TOP_K_ARTICLES,
    TOP_K_RULES,
    ECO_METRIC_OUTPUT_FILENAME,
)
from generation.rag_pipeline.node_utils import result_to_dict


# =============================================================================
# RUN MANAGEMENT
# =============================================================================

def get_existing_run_numbers(runs_dir: Path) -> list[int]:
    """
    Return all numeric run IDs already present in the runs directory.

    Expected folder names:
        run_001
        run_002
        run_003
    """
    if not runs_dir.exists():
        return []

    run_numbers: list[int] = []

    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue

        match = RUN_DIR_PATTERN.match(path.name)

        if match:
            run_numbers.append(int(match.group(1)))

    return run_numbers


def create_run_id() -> str:
    """
    Create the next incremental run ID based on existing folders.

    Example:
        if data/retrieval_runs/run_004 exists,
        the next run will be run_005.
    """
    existing_numbers = get_existing_run_numbers(RETRIEVAL_RUNS_DIR)
    next_number = max(existing_numbers, default=0) + 1

    return f"run_{next_number:0{RUN_ID_WIDTH}d}"


# =============================================================================
# JSON PAYLOADS
# =============================================================================

def build_retrieval_json_payload(
    run_id: str,
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
) -> dict[str, Any]:
    """
    Build the full JSON payload for raw retrieval results.
    """
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    rules = [
        result_to_dict(result, rank=rank)
        for rank, result in enumerate(retrieved["rules"], start=1)
    ]

    articles = [
        result_to_dict(result, rank=rank)
        for rank, result in enumerate(retrieved["articles"], start=1)
    ]

    return {
        "run_id": run_id,
        "created_at": created_at,
        "query": query,
        "stage": "retrieval",
        "config": {
            "embedding_model": EMBED_MODEL_NAME,
            "top_k_rules": TOP_K_RULES,
            "top_k_articles": TOP_K_ARTICLES,
            "article_index_dir": str(ARTICLE_INDEX_DIR),
            "rule_index_dir": str(RULE_INDEX_DIR),
        },
        "summary": {
            "num_rule_results": len(rules),
            "num_article_results": len(articles),
        },
        "results": {
            "rules": rules,
            "articles": articles,
        },
    }


def build_postprocessed_json_payload(
    run_id: str,
    query: str,
    postprocessed: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the full JSON payload for postprocessed results.
    """
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    selected_rules = postprocessed["rules"]["selected"]
    discarded_rules = postprocessed["rules"]["discarded"]
    selected_articles = postprocessed["articles"]["selected"]
    discarded_articles = postprocessed["articles"]["discarded"]

    return {
        "run_id": run_id,
        "created_at": created_at,
        "query": query,
        "stage": "postprocessing",
        "config": {
            "rule_postprocessing": {
                "method": "textual_deduplication",
                "initial_top_k": TOP_K_RULES,
                "final_top_k": FINAL_RULES_K,
                "dedup_similarity_threshold": RULE_DEDUP_SIMILARITY_THRESHOLD,
            },
            "article_postprocessing": {
                "method": "min_length_filter_plus_cross_encoder_reranking",
                "initial_top_k": TOP_K_ARTICLES,
                "final_top_k": FINAL_ARTICLES_K,
                "min_article_chars": MIN_ARTICLE_CHARS,
                "rerank_model": RERANK_MODEL_NAME,
            },
        },
        "summary": {
            "selected_rules": len(selected_rules),
            "discarded_rules": len(discarded_rules),
            "selected_articles": len(selected_articles),
            "discarded_articles": len(discarded_articles),
        },
        "results": {
            "rules": {
                "selected": selected_rules,
                "discarded": discarded_rules,
            },
            "articles": {
                "selected": selected_articles,
                "discarded": discarded_articles,
            },
        },
    }


def build_generation_json_payload(
    run_id: str,
    query: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the full JSON payload for generation results.
    """
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    return {
        "run_id": run_id,
        "created_at": created_at,
        "query": query,
        "stage": "generation",
        "config": {
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL_NAME,
            "ollama_host": OLLAMA_CLOUD_HOST,
            "temperature": LLM_TEMPERATURE,
        },
        "results": generation,
    }

def build_eco_metric_json_payload(
    run_id: str,
    query: str,
    eco_metric: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the full JSON payload for eco-metric results.
    """
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    return {
        "run_id": run_id,
        "created_at": created_at,
        "query": query,
        "stage": "eco_metric",
        "results": eco_metric,
    }

# =============================================================================
# SAVING
# =============================================================================

def save_run_results(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
    postprocessed: dict[str, Any],
    generation: dict[str, Any],
    eco_metric: dict[str, Any],
) -> dict[str, Path]:
    """
    Save retrieval, postprocessing, generation and eco-metric results
    inside a dedicated folder.

    Output structure:
        data/retrieval_runs/
            run_001/
                retrieval_results.json
                postprocessed_results.json
                generation_results.json
                eco_metric_results.json
    """
    run_id = create_run_id()
    run_dir = RETRIEVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    retrieval_output_path = run_dir / RETRIEVAL_OUTPUT_FILENAME
    postprocessed_output_path = run_dir / POSTPROCESSED_OUTPUT_FILENAME
    generation_output_path = run_dir / GENERATION_OUTPUT_FILENAME
    eco_metric_output_path = run_dir / ECO_METRIC_OUTPUT_FILENAME

    retrieval_payload = build_retrieval_json_payload(
        run_id=run_id,
        query=query,
        retrieved=retrieved,
    )

    postprocessed_payload = build_postprocessed_json_payload(
        run_id=run_id,
        query=query,
        postprocessed=postprocessed,
    )

    generation_payload = build_generation_json_payload(
        run_id=run_id,
        query=query,
        generation=generation,
    )

    eco_metric_payload = build_eco_metric_json_payload(
        run_id=run_id,
        query=query,
        eco_metric=eco_metric,
    )

    with retrieval_output_path.open("w", encoding="utf-8") as file:
        json.dump(retrieval_payload, file, indent=2, ensure_ascii=False)

    with postprocessed_output_path.open("w", encoding="utf-8") as file:
        json.dump(postprocessed_payload, file, indent=2, ensure_ascii=False)

    with generation_output_path.open("w", encoding="utf-8") as file:
        json.dump(generation_payload, file, indent=2, ensure_ascii=False)

    with eco_metric_output_path.open("w", encoding="utf-8") as file:
        json.dump(eco_metric_payload, file, indent=2, ensure_ascii=False)

    return {
        "run_dir": run_dir,
        "retrieval_output_path": retrieval_output_path,
        "postprocessed_output_path": postprocessed_output_path,
        "generation_output_path": generation_output_path,
        "eco_metric_output_path": eco_metric_output_path,
    }

from generation.rag_pipeline.display import (
    print_eco_metric_result,
    print_generation_result,
    print_postprocessing_summary,
    print_retrieval_results,
    print_saved_paths,
)