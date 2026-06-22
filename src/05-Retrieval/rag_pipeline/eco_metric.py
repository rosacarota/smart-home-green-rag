from __future__ import annotations

from typing import Any
import json
import time

import jsonschema

from rag_pipeline.clients import configure_llm_client
from rag_pipeline.config import (
    ECO_JUDGE_MAX_RETRIES,
    ECO_JUDGE_MODEL_NAME,
    ECO_JUDGE_TEMPERATURE,
    JUDGE_PROMPT_PATH,
    JUDGE_SCHEMA_PATH,
)
from rag_pipeline.generation import (
    extract_ollama_response_content,
    parse_llm_json_response,
)


def load_judge_prompt() -> str:
    """
    Load the eco-metric judge system prompt.
    """
    if not JUDGE_PROMPT_PATH.exists():
        raise FileNotFoundError(
            f"Judge prompt not found: {JUDGE_PROMPT_PATH}"
        )

    return JUDGE_PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()


def load_judge_schema() -> dict[str, Any]:
    """
    Load the JSON schema required for the eco-metric judge response.
    """
    if not JUDGE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Judge schema not found: {JUDGE_SCHEMA_PATH}"
        )

    return json.loads(
        JUDGE_SCHEMA_PATH.read_text(encoding="utf-8")
    )


def build_eco_metric_user_prompt(
    original_rule: str,
    generated_rule: str,
    judge_schema: dict[str, Any],
    rule_id: str = "PIPELINE_RULE",
) -> str:
    """
    Build the user prompt for the eco-metric judge.

    The retrieved context is intentionally not included: the judge evaluates
    the rule pair itself, not the evidence used by the RAG.

    The JSON schema is included explicitly because, unlike the old OpenRouter
    version, this Ollama call does not enforce the schema through the API.
    """
    schema_text = json.dumps(
        judge_schema,
        indent=2,
        ensure_ascii=False,
    )

    lines = [
        f"rule_id: {rule_id}",
        "",
        "Original rule (r):",
        original_rule,
        "",
        "Generated green variant (r'):",
        generated_rule,
        "",
        "Required JSON schema:",
        schema_text,
        "",
        "Return only valid JSON.",
        "Do not include markdown fences.",
        "Do not include explanations outside the JSON object.",
    ]

    return "\n".join(lines)


def compute_scores(
    orig: dict[str, int],
    gen: dict[str, int],
    validity: dict[str, int],
) -> dict[str, float | int | str]:
    """
    Compute all eco-metric scores deterministically from feature scores.

    The LLM only assigns feature-level scores.
    The final mathematical computation is done here.
    """

    def s_awareness(fs: dict[str, int]) -> float:
        occ_norm = fs["OccupancyScore"] / 2
        env_norm = fs["EnvConditionScore"] / 1

        return 0.6 * occ_norm + 0.4 * env_norm

    def s_energy(fs: dict[str, int]) -> float:
        dur_norm = fs["DurationScore"] / 1
        std_norm = fs["StandbyScore"] / 2
        app_norm = fs["ApplianceScore"] / 2
        sch_norm = fs["SchedulingScore"] / 2

        return (
            0.4 * dur_norm
            + 0.4 * std_norm
            + 0.1 * app_norm
            + 0.1 * sch_norm
        )

    saw_orig = s_awareness(orig)
    saw_gen = s_awareness(gen)
    sen_orig = s_energy(orig)
    sen_gen = s_energy(gen)

    delta_awareness = saw_gen - saw_orig
    delta_energy = sen_gen - sen_orig
    eco_static = 0.5 * delta_awareness + 0.5 * delta_energy

    intent_preserved: int = validity["IntentPreserved"]
    comfort_safety_penalty: int = validity["ComfortSafetyPenalty"]

    if intent_preserved == 0:
        label = "reject"
        label_reason = "IntentPreserved=0"
    elif comfort_safety_penalty == 1:
        label = "reject"
        label_reason = "ComfortSafetyPenalty=1"
    elif eco_static >= 0.2:
        label = "strong_green"
        label_reason = f"EcoStatic={eco_static:.4f} >= 0.2"
    elif eco_static > 0:
        label = "accept"
        label_reason = f"EcoStatic={eco_static:.4f} > 0"
    else:
        label = "reject"
        label_reason = f"EcoStatic={eco_static:.4f} <= 0"

    return {
        "S_awareness_orig": round(saw_orig, 6),
        "S_awareness_gen": round(saw_gen, 6),
        "S_energy_orig": round(sen_orig, 6),
        "S_energy_gen": round(sen_gen, 6),
        "Delta_awareness": round(delta_awareness, 6),
        "Delta_energy": round(delta_energy, 6),
        "EcoStatic": round(eco_static, 6),
        "IntentPreserved": intent_preserved,
        "ComfortSafetyPenalty": comfort_safety_penalty,
        "label": label,
        "label_reason": label_reason,
    }


def validate_against_schema(
    response: dict[str, Any],
    schema: dict[str, Any],
    eval_id: str,
) -> None:
    """
    Validate the judge response against judge_schema.json.
    """
    try:
        jsonschema.validate(
            instance=response,
            schema=schema,
        )
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"Eco-metric judge response for {eval_id} failed schema validation: "
            f"{exc.message}"
        ) from exc


def call_eco_metric_judge(
    original_rule: str,
    generated_rule: str,
    rule_id: str = "PIPELINE_RULE",
) -> dict[str, Any]:
    """
    Call Ollama Cloud as eco-metric judge and return the parsed JSON response.

    The function retries if the response is not valid JSON or does not match
    the required schema.
    """
    client = configure_llm_client()

    system_prompt = load_judge_prompt()
    judge_schema = load_judge_schema()

    user_prompt = build_eco_metric_user_prompt(
        original_rule=original_rule,
        generated_rule=generated_rule,
        judge_schema=judge_schema,
        rule_id=rule_id,
    )

    last_error: Exception | None = None

    for attempt in range(1, ECO_JUDGE_MAX_RETRIES + 1):
        try:
            response = client.chat(
                model=ECO_JUDGE_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                stream=False,
                options={
                    "temperature": ECO_JUDGE_TEMPERATURE,
                },
            )

            raw_response = extract_ollama_response_content(response)
            parsed_response = parse_llm_json_response(raw_response)

            if parsed_response.get("parse_error"):
                raise ValueError(
                    f"Judge returned invalid JSON: {raw_response}"
                )

            validate_against_schema(
                response=parsed_response,
                schema=judge_schema,
                eval_id=rule_id,
            )

            return {
                "raw_response_text": raw_response,
                "parsed_response": parsed_response,
            }

        except Exception as exc:
            last_error = exc
            wait_seconds = 2 ** attempt

            print(
                f"Eco-metric attempt {attempt}/{ECO_JUDGE_MAX_RETRIES} "
                f"failed for {rule_id}: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Eco-metric judge failed after {ECO_JUDGE_MAX_RETRIES} attempts "
        f"for {rule_id}: {last_error}"
    )


def evaluate_rule_pair(
    original_rule: str,
    generated_rule: str,
    rule_id: str = "PIPELINE_RULE",
) -> dict[str, Any]:
    """
    Evaluate original rule and generated rule with the eco-metric.

    Returns:
        - original/generated feature analyses from the judge
        - deterministic eco-metric scores
        - final label
    """
    judge_output = call_eco_metric_judge(
        original_rule=original_rule,
        generated_rule=generated_rule,
        rule_id=rule_id,
    )

    raw_llm_response = judge_output["parsed_response"]

    orig_fs = raw_llm_response["original_rule_analysis"]["feature_scores"]
    gen_fs = raw_llm_response["generated_rule_analysis"]["feature_scores"]
    validity = raw_llm_response["validity_checks"]

    scores = compute_scores(
        orig=orig_fs,
        gen=gen_fs,
        validity=validity,
    )

    return {
        "eval_id": f"EVAL_{rule_id}",
        "rule_id": rule_id,
        "original_rule": original_rule,
        "generated_rule": generated_rule,
        "judge_model": ECO_JUDGE_MODEL_NAME,
        "judge_temperature": ECO_JUDGE_TEMPERATURE,
        "raw_llm_response": raw_llm_response,
        "raw_response_text": judge_output["raw_response_text"],
        "scores": scores,
        "label": scores["label"],
        "label_reason": scores["label_reason"],
        "label_source": "script",
    }