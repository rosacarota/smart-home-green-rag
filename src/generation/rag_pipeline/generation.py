from __future__ import annotations

from typing import Any
import json

from generation.rag_pipeline.clients import configure_llm_client
from generation.rag_pipeline.config import (
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_selected_context_text(
    selected_items: list[dict[str, Any]],
    label: str,
) -> str:
    """
    Convert selected postprocessed items into a compact context string
    for the generation prompt.

    Example context IDs:
        RULE_1
        RULE_2
        ARTICLE_1
        ARTICLE_2
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
    Build the final RAG prompt for the LLM.

    The prompt receives:
        - the original trigger-action rule
        - selected similar rules
        - selected article chunks

    The model must return only a JSON object.
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
- Do not introduce specific brands, platforms, or device names unless they are explicitly present in the original rule.
- If the retrieved context contains brand-specific devices, treat them only as examples and generalize them to neutral device categories.

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
# RESPONSE PARSING
# =============================================================================

def parse_llm_json_response(response_text: str) -> dict[str, Any]:
    """
    Parse the LLM response as JSON.

    First, try to parse the full response.
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

    This supports both dictionary-like responses and object-like responses,
    because different versions of the ollama Python package may expose the
    response slightly differently.
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


# =============================================================================
# GENERATION
# =============================================================================

def generate_green_rule(
    query: str,
    postprocessed: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate an improved green rule using Ollama Cloud.

    Input:
        query:
            Original trigger-action rule.

        postprocessed:
            Output of postprocess_context(), containing selected rules
            and selected article chunks.

    Output:
        A dictionary containing:
            - provider
            - model
            - temperature
            - prompt
            - raw_response
            - parsed_response
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