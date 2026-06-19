from __future__ import annotations

from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
from datetime import datetime
import argparse
import json
import os
import re

from dotenv import load_dotenv
from ollama import Client

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder


# =============================================================================
# PATHS
# =============================================================================

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"

RETRIEVAL_RUNS_DIR = Path("data/retrieval_runs")


# =============================================================================
# MODELS
# =============================================================================

EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_PROVIDER = "ollama_cloud"
LLM_MODEL_NAME = "gpt-oss:20b"
OLLAMA_CLOUD_HOST = "https://ollama.com"
LLM_TEMPERATURE = 0.2


# =============================================================================
# RETRIEVAL / POSTPROCESSING CONFIG
# =============================================================================

TOP_K_RULES = 10
TOP_K_ARTICLES = 15

FINAL_RULES_K = 3
FINAL_ARTICLES_K = 5

MIN_ARTICLE_CHARS = 150
TEXT_PREVIEW_CHARS = 900

RULE_DEDUP_SIMILARITY_THRESHOLD = 0.88


# =============================================================================
# RUN CONFIG
# =============================================================================

RUN_ID_WIDTH = 3
RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")

RETRIEVAL_OUTPUT_FILENAME = "retrieval_results.json"
POSTPROCESSED_OUTPUT_FILENAME = "postprocessed_results.json"
GENERATION_OUTPUT_FILENAME = "generation_results.json"

DEFAULT_QUERY = "IF You exit from home THEN Turn off lights."


# =============================================================================
# LLAMAINDEX / LLM CONFIGURATION
# =============================================================================

def configure_llamaindex() -> None:
    """
    Configure the same embedding model used to build the persisted indexes.

    During retrieval, the input query must be embedded in the same vector space
    as the nodes stored in the vector indexes.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )

    Settings.embed_model = embed_model


def configure_llm_client() -> Client:
    """
    Configure the Ollama Cloud client.

    The API key is loaded from the .env file through python-dotenv.

    Required .env variable:
        OLLAMA_API_KEY=your_api_key
    """
    load_dotenv()

    api_key = os.environ.get("OLLAMA_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing OLLAMA_API_KEY. "
            "Add it to your .env file, for example:\n"
            "OLLAMA_API_KEY=your_api_key"
        )

    return Client(
        host=OLLAMA_CLOUD_HOST,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )


# =============================================================================
# INDEX LOADING
# =============================================================================

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


# =============================================================================
# NODE SERIALIZATION
# =============================================================================

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


# =============================================================================
# RETRIEVAL
# =============================================================================

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


# =============================================================================
# RULE POSTPROCESSING
# =============================================================================

def normalize_text_for_dedup(text: str) -> str:
    """
    Normalize text to make rule deduplication more robust.
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


def postprocess_context(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
) -> dict[str, Any]:
    """
    Apply the full postprocessing pipeline.
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


# =============================================================================
# GENERATION PROMPT
# =============================================================================

def build_selected_context_text(
    selected_items: list[dict[str, Any]],
    label: str,
) -> str:
    """
    Convert selected postprocessed items into a compact context string
    for the generation prompt.
    """
    parts: list[str] = []

    for item in selected_items:
        context_id = f"{label}_{item['final_rank']}"

        source = (
            item.get("rule_id")
            or item.get("source_title")
            or item.get("node_id")
        )

        section = item.get("section_title", "")
        text = item.get("text", "")

        parts.append(
            f"[{context_id}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Original retrieval score: {item.get('original_score')}\n"
            f"Rerank score: {item.get('rerank_score')}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(parts)


def build_generation_prompt(
    original_rule: str,
    postprocessed: dict[str, Any],
) -> str:
    """
    Build the final prompt for the LLM using the selected postprocessed context.
    """
    selected_rules = postprocessed["rules"]["selected"]
    selected_articles = postprocessed["articles"]["selected"]

    rules_context = build_selected_context_text(
        selected_items=selected_rules,
        label="RULE",
    )

    articles_context = build_selected_context_text(
        selected_items=selected_articles,
        label="ARTICLE",
    )

    return f"""
You are an assistant specialized in improving smart-home trigger-action rules.

Your task is to rewrite the original rule into a more sustainable version.

Original rule:
{original_rule}

Similar retrieved rules:
{rules_context}

Retrieved sustainability and smart-home knowledge:
{articles_context}

Requirements:
- Preserve the original functional intention.
- Make the rule more sustainable when possible.
- Prefer realistic smart-home mechanisms.
- You may use conditions such as occupancy, presence, time limits, thresholds, scheduling, standby reduction, or sensor-based control when appropriate.
- Do not invent unrelated devices.
- Do not make the rule overly complex.
- The improved rule must still be written as a trigger-action rule.
- Return only valid JSON.
- Do not include markdown fences.
- Do not include explanations outside the JSON object.

JSON schema:
{{
  "original_rule": "...",
  "improved_rule": "...",
  "green_strategy": "...",
  "preserved_intent": true,
  "explanation": "...",
  "used_context_ids": ["RULE_1", "ARTICLE_1"]
}}
""".strip()


# =============================================================================
# GENERATION
# =============================================================================

def parse_llm_json_response(response_text: str) -> dict[str, Any]:
    """
    Parse the LLM response as JSON.

    If the model returns extra text, try to extract the first JSON object.
    """
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return {
            "parse_error": True,
            "raw_response": response_text,
        }

    json_candidate = response_text[start : end + 1]

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError:
        return {
            "parse_error": True,
            "raw_response": response_text,
        }


def extract_ollama_response_content(response: Any) -> str:
    """
    Extract text content from an Ollama response.

    This supports both dictionary-like responses and object-like responses.
    """
    if isinstance(response, dict):
        message = response.get("message", {})

        if isinstance(message, dict):
            return str(message.get("content", ""))

        return str(getattr(message, "content", ""))

    message = getattr(response, "message", None)

    if isinstance(message, dict):
        return str(message.get("content", ""))

    if message is not None:
        return str(getattr(message, "content", ""))

    return str(response)


def generate_green_rule(
    query: str,
    postprocessed: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an improved green rule using Ollama Cloud.
    """
    client = configure_llm_client()

    prompt = build_generation_prompt(
        original_rule=query,
        postprocessed=postprocessed,
    )

    response = client.chat(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        options={
            "temperature": LLM_TEMPERATURE,
        },
    )

    raw_response = extract_ollama_response_content(response)
    parsed_response = parse_llm_json_response(raw_response)

    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
    }


# =============================================================================
# PRINTING
# =============================================================================

def print_rule_results(results: list[NodeWithScore]) -> None:
    print("\n" + "=" * 90)
    print("RETRIEVED SIMILAR RULES")
    print("=" * 90)

    if not results:
        print("No rule results found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result_to_dict(result, rank=rank)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {item['score']}")
        print(f"Rule ID: {item['rule_id']}")
        print(f"Topic: {item['llm_topic_name']}")
        print(f"Green category: {item['green_category']}")
        print(f"Node ID: {item['node_id']}")
        print("\nText preview:")
        print(item["text_preview"])


def print_article_results(results: list[NodeWithScore]) -> None:
    print("\n" + "=" * 90)
    print("RETRIEVED ARTICLE CHUNKS")
    print("=" * 90)

    if not results:
        print("No article results found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result_to_dict(result, rank=rank)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {item['score']}")
        print(f"Article: {item['source_title']}")
        print(f"Article ID: {item['article_id']}")
        print(f"Section: {item['section_title']}")
        print(f"Section path: {item['section_path']}")
        print(f"Green categories: {item['green_categories']}")
        print(f"Node ID: {item['node_id']}")
        print("\nText preview:")
        print(item["text_preview"])


def print_retrieval_results(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
) -> None:
    print("\n" + "=" * 90)
    print("QUERY")
    print("=" * 90)
    print(query)

    print_rule_results(retrieved["rules"])
    print_article_results(retrieved["articles"])


def print_postprocessing_summary(postprocessed: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("POSTPROCESSING SUMMARY")
    print("=" * 90)

    selected_rules = postprocessed["rules"]["selected"]
    discarded_rules = postprocessed["rules"]["discarded"]
    selected_articles = postprocessed["articles"]["selected"]
    discarded_articles = postprocessed["articles"]["discarded"]

    print(f"Selected rules: {len(selected_rules)}")
    print(f"Discarded rules: {len(discarded_rules)}")
    print(f"Selected articles: {len(selected_articles)}")
    print(f"Discarded articles: {len(discarded_articles)}")

    print("\nFINAL RULES")
    for item in selected_rules:
        print(
            f"[{item['final_rank']}] "
            f"original_rank={item['original_rank']} "
            f"original_score={item['original_score']}"
        )
        print(item["text_preview"])

    print("\nFINAL ARTICLES")
    for item in selected_articles:
        print(
            f"[{item['final_rank']}] "
            f"original_rank={item['original_rank']} "
            f"original_score={item['original_score']} "
            f"rerank_score={item['rerank_score']}"
        )
        print(f"{item['source_title']} / {item['section_title']}")
        print(item["text_preview"][:300])


def print_generation_result(generation: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("GENERATED GREEN RULE")
    print("=" * 90)

    parsed_response = generation.get("parsed_response", {})

    print(json.dumps(
        parsed_response,
        indent=2,
        ensure_ascii=False,
    ))


# =============================================================================
# RUN MANAGEMENT
# =============================================================================

def get_existing_run_numbers(runs_dir: Path) -> list[int]:
    """
    Return all numeric run IDs already present in the runs directory.
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


# =============================================================================
# SAVING
# =============================================================================

def save_run_results(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
    postprocessed: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Path]:
    """
    Save retrieval, postprocessing and generation results inside a dedicated folder.

    Output structure:
        data/retrieval_runs/
            run_001/
                retrieval_results.json
                postprocessed_results.json
                generation_results.json
    """
    run_id = create_run_id()
    run_dir = RETRIEVAL_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    retrieval_output_path = run_dir / RETRIEVAL_OUTPUT_FILENAME
    postprocessed_output_path = run_dir / POSTPROCESSED_OUTPUT_FILENAME
    generation_output_path = run_dir / GENERATION_OUTPUT_FILENAME

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

    with retrieval_output_path.open("w", encoding="utf-8") as file:
        json.dump(retrieval_payload, file, indent=2, ensure_ascii=False)

    with postprocessed_output_path.open("w", encoding="utf-8") as file:
        json.dump(postprocessed_payload, file, indent=2, ensure_ascii=False)

    with generation_output_path.open("w", encoding="utf-8") as file:
        json.dump(generation_payload, file, indent=2, ensure_ascii=False)

    return {
        "run_dir": run_dir,
        "retrieval_output_path": retrieval_output_path,
        "postprocessed_output_path": postprocessed_output_path,
        "generation_output_path": generation_output_path,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load persisted RAG indexes, retrieve relevant rules/article chunks, "
            "postprocess them, and generate a greener smart-home rule."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Input trigger-action rule or natural-language query.",
    )

    args = parser.parse_args()

    retrieved = retrieve_context(
        query=args.query,
    )

    print_retrieval_results(
        query=args.query,
        retrieved=retrieved,
    )

    postprocessed = postprocess_context(
        query=args.query,
        retrieved=retrieved,
    )

    print_postprocessing_summary(
        postprocessed=postprocessed,
    )

    generation = generate_green_rule(
        query=args.query,
        postprocessed=postprocessed,
    )

    print_generation_result(
        generation=generation,
    )

    output_paths = save_run_results(
        query=args.query,
        retrieved=retrieved,
        postprocessed=postprocessed,
        generation=generation,
    )

    print("\n" + "=" * 90)
    print("RUN SAVED")
    print("=" * 90)
    print(f"Run directory: {output_paths['run_dir']}")
    print(f"Retrieval JSON: {output_paths['retrieval_output_path']}")
    print(f"Postprocessed JSON: {output_paths['postprocessed_output_path']}")
    print(f"Generation JSON: {output_paths['generation_output_path']}")


if __name__ == "__main__":
    main()