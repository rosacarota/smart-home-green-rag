from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from generation.rag_pipeline.clients import configure_llm_client
from generation.rag_pipeline.eco_metric import (
    build_eco_metric_user_prompt,
    compute_scores,
    load_judge_prompt,
    load_judge_schema,
    validate_against_schema,
)
from generation.rag_pipeline.generation import (
    extract_ollama_response_content,
    parse_llm_json_response,
)


DEFAULT_EXPERIMENTS_DIR = (
    ROOT / "data" / "synthetic_evaluation"
)

DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def sanitize_model_name(model_name: str) -> str:
    """
    Convert a model name into a safe folder name.

    Examples:
        gemma4:31b-cloud -> gemma4_31b_cloud
        openai/gpt-4o-mini -> openai_gpt_4o_mini
        google/gemini-2.5-flash -> google_gemini_2_5_flash
    """
    return (
        model_name
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def build_judge_folder_name(
    judge_provider: str,
    judge_model: str,
) -> str:
    """
    Build a safe folder name containing provider and model name.

    Examples:
        ollama + gemma4:31b-cloud
        -> ollama_gemma4_31b_cloud

        openrouter + openai/gpt-4o-mini
        -> openrouter_openai_gpt_4o_mini
    """
    safe_model_name = sanitize_model_name(judge_model)

    return f"{judge_provider}_{safe_model_name}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

    return records


def append_jsonl(record: dict[str, Any], path: Path) -> None:
    """
    Append one record to a JSONL file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def append_text_line(value: str, path: Path) -> None:
    """
    Append one plain-text line to a file.

    This is used to save rule IDs that need regeneration.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("a", encoding="utf-8") as file:
        file.write(value + "\n")


def build_eval_key(record: dict[str, Any]) -> str:
    """
    Build a stable key for resume logic.

    This uses both rule_id and generator_model, so the script still works
    if in the future multiple generator models are used.
    """
    rule_id = str(record.get("rule_id", ""))
    generator_model = str(record.get("generator_model", ""))

    return f"{rule_id}::{generator_model}"


def get_processed_eval_keys(output_path: Path) -> set[str]:
    """
    Read an existing judgment file and return already processed eval keys.
    """
    if not output_path.exists():
        return set()

    processed: set[str] = set()

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            processed.add(build_eval_key(record))

    return processed


def validate_generation_record(record: dict[str, Any], index: int) -> None:
    """
    Validate that a generation record contains all fields needed by the judge.
    """
    required_fields = [
        "rule_id",
        "target_category",
        "difficulty",
        "original_rule",
        "generated_rule",
        "generator_model",
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(
                f"Record {index} is missing required field '{field}'."
            )

        if not str(record[field]).strip():
            raise ValueError(
                f"Record {index} has empty field '{field}'."
            )


def get_invalidity_reason(scores: dict[str, Any]) -> str | None:
    """
    Decide whether a generated rule is invalid and must be regenerated.

    A generated rule is invalid if:
    - it does not preserve the original functional intent
    - it introduces an evident comfort or safety risk
    """
    if int(scores["IntentPreserved"]) == 0:
        return "IntentPreserved=0"

    if int(scores["ComfortSafetyPenalty"]) == 1:
        return "ComfortSafetyPenalty=1"

    return None


def is_non_retryable_error(error_message: str) -> bool:
    """
    Detect errors that should not be retried.

    These are usually authentication, permission, payment, invalid schema,
    unsupported parameter, or missing-model errors.
    """
    non_retryable_patterns = [
        "status code: 400",
        "status code: 401",
        "status code: 402",
        "status code: 403",
        "status code: 404",
        "badrequesterror",
        "authenticationerror",
        "permissiondeniederror",
        "notfounderror",
        "invalid api key",
        "incorrect api key",
        "unauthorized",
        "forbidden",
        "payment required",
        "not found",
        "model_not_found",
        "invalid_request_error",
        "unsupported_parameter",
        "unsupported response_format",
        "requires a subscription",
    ]

    normalized_error = error_message.casefold()

    return any(
        pattern in normalized_error
        for pattern in non_retryable_patterns
    )


def make_openrouter_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Make the existing judge schema more compatible with strict structured outputs.

    The original judge_schema.json is still the source of truth.
    This function only adds additionalProperties=False to object nodes,
    so the model cannot add hallucinated fields.
    """
    strict_schema = copy.deepcopy(schema)

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)

                properties = node.get("properties", {})
                if isinstance(properties, dict):
                    for child in properties.values():
                        visit(child)

            if node.get("type") == "array":
                visit(node.get("items"))

            for key in [
                "anyOf",
                "oneOf",
                "allOf",
            ]:
                value = node.get(key)

                if isinstance(value, list):
                    for child in value:
                        visit(child)

        elif isinstance(node, list):
            for item in node:
                visit(item)

        return node

    return visit(strict_schema)


def configure_openrouter_client() -> OpenAI:
    """
    Configure the OpenAI Python client to use OpenRouter as backend.

    Required .env variable:
        OPEN_ROUTER_KEY=your_api_key
    """
    load_dotenv()

    api_key = os.environ.get("OPEN_ROUTER_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing OPEN_ROUTER_KEY. "
            "Add it to your .env file, for example:\n"
            "OPEN_ROUTER_KEY=your_api_key"
        )

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,

    )


def extract_openrouter_response_content(response: Any) -> str:
    """
    Extract assistant content from an OpenRouter response returned through
    the OpenAI Python SDK.
    """
    choice = response.choices[0]
    message = choice.message
    content = message.content

    if content is None:
        raise ValueError(
            f"OpenRouter response did not contain message.content: {response}"
        )

    return str(content)


def call_ollama_judge_model(
    original_rule: str,
    generated_rule: str,
    judge_model: str,
    rule_id: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """
    Call one Ollama judge and return the parsed JSON response.
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

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat(
                model=judge_model,
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
                    "temperature": temperature,
                },
            )

            raw_response = extract_ollama_response_content(response)
            parsed_response = parse_llm_json_response(raw_response)

            if parsed_response.get("parse_error"):
                raise ValueError(
                    f"Ollama judge returned invalid JSON: {raw_response}"
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
            error_message = str(exc)

            if is_non_retryable_error(error_message):
                raise RuntimeError(
                    f"Ollama model access/request failed for {judge_model}: "
                    f"{error_message}"
                ) from exc

            wait_seconds = 2 ** attempt

            print(
                f"Ollama judge attempt {attempt}/{max_retries} failed "
                f"for {rule_id} with model {judge_model}: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Ollama judge failed after {max_retries} attempts for {rule_id} "
        f"with model {judge_model}: {last_error}"
    )


def call_openrouter_judge_model(
    original_rule: str,
    generated_rule: str,
    judge_model: str,
    rule_id: str,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """
    Call one OpenRouter judge through the OpenAI Python SDK.

    Inference runs on OpenRouter.
    The code uses the OpenAI client with base_url pointing to OpenRouter.
    """
    client = configure_openrouter_client()

    system_prompt = load_judge_prompt()
    judge_schema = load_judge_schema()
    openrouter_schema = make_openrouter_strict_schema(judge_schema)

    user_prompt = build_eco_metric_user_prompt(
        original_rule=original_rule,
        generated_rule=generated_rule,
        judge_schema=judge_schema,
        rule_id=rule_id,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "eco_metric_judgment",
                        "strict": True,
                        "schema": openrouter_schema,
                    },
                },
                extra_body={
                    "provider": {
                        "require_parameters": True,
                    },
                    "reasoning": {
                        "effort": "none",
                    },
                },
            )

            raw_response = extract_openrouter_response_content(response)
            parsed_response = parse_llm_json_response(raw_response)

            if parsed_response.get("parse_error"):
                raise ValueError(
                    f"OpenRouter judge returned invalid JSON: {raw_response}"
                )

            validate_against_schema(
                response=parsed_response,
                schema=judge_schema,
                eval_id=rule_id,
            )

            return {
                "raw_response_text": raw_response,
                "parsed_response": parsed_response,
                "provider_response": response.model_dump(),
            }

        except Exception as exc:
            last_error = exc
            error_message = str(exc)

            if is_non_retryable_error(error_message):
                raise RuntimeError(
                    f"OpenRouter model access/request failed for {judge_model}: "
                    f"{error_message}"
                ) from exc

            wait_seconds = 2 ** attempt

            print(
                f"OpenRouter judge attempt {attempt}/{max_retries} failed "
                f"for {rule_id} with model {judge_model}: {exc}. "
                f"Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"OpenRouter judge failed after {max_retries} attempts for {rule_id} "
        f"with model {judge_model}: {last_error}"
    )


def call_judge_model(
    original_rule: str,
    generated_rule: str,
    judge_provider: str,
    judge_model: str,
    rule_id: str,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """
    Dispatch the judge call to Ollama or OpenRouter.
    """
    if judge_provider == "ollama":
        return call_ollama_judge_model(
            original_rule=original_rule,
            generated_rule=generated_rule,
            judge_model=judge_model,
            rule_id=rule_id,
            temperature=temperature,
            max_retries=max_retries,
        )

    if judge_provider == "openrouter":
        return call_openrouter_judge_model(
            original_rule=original_rule,
            generated_rule=generated_rule,
            judge_model=judge_model,
            rule_id=rule_id,
            temperature=temperature,
            max_retries=max_retries,
        )

    raise ValueError(
        f"Unsupported judge provider: {judge_provider}"
    )


def build_judgment_record(
    generation_record: dict[str, Any],
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    judge_output: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the final judgment record saved to JSONL.
    """
    raw_llm_response = judge_output["parsed_response"]

    original_feature_scores = (
        raw_llm_response["original_rule_analysis"]["feature_scores"]
    )

    generated_feature_scores = (
        raw_llm_response["generated_rule_analysis"]["feature_scores"]
    )

    validity = raw_llm_response["validity_checks"]

    scores = compute_scores(
        orig=original_feature_scores,
        gen=generated_feature_scores,
        validity=validity,
    )

    invalid_reason = get_invalidity_reason(scores)
    needs_regeneration = invalid_reason is not None
    generation_status = "invalid" if needs_regeneration else "valid"

    if needs_regeneration:
        final_label = "invalid"
        label_reason = invalid_reason
    else:
        final_label = scores["label"]
        label_reason = scores["label_reason"]

    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),

        "rule_id": generation_record["rule_id"],
        "target_category": generation_record["target_category"],
        "difficulty": generation_record["difficulty"],
        "source_distribution_count": generation_record.get(
            "source_distribution_count",
            None,
        ),

        "original_rule": generation_record["original_rule"],
        "generated_rule": generation_record["generated_rule"],

        "generator_model": generation_record["generator_model"],
        "generator_temperature": generation_record.get(
            "generator_temperature",
            None,
        ),

        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_temperature": judge_temperature,

        "S_awareness_orig": scores["S_awareness_orig"],
        "S_awareness_gen": scores["S_awareness_gen"],
        "S_energy_orig": scores["S_energy_orig"],
        "S_energy_gen": scores["S_energy_gen"],
        "Delta_awareness": scores["Delta_awareness"],
        "Delta_energy": scores["Delta_energy"],
        "EcoStatic": scores["EcoStatic"],

        "IntentPreserved": scores["IntentPreserved"],
        "ComfortSafetyPenalty": scores["ComfortSafetyPenalty"],

        "generation_status": generation_status,
        "needs_regeneration": needs_regeneration,
        "invalid_reason": invalid_reason,

        "label": final_label,
        "eco_label": scores["label"],
        "label_reason": label_reason,
        "label_source": "script",

        "scores": scores,
        "raw_llm_response": raw_llm_response,
        "raw_response_text": judge_output["raw_response_text"],
    }


def build_invalid_generation_record(
    judgment_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact record for invalid generations.

    This file is useful for later regeneration.
    """
    return {
        "created_at": judgment_record["created_at"],
        "rule_id": judgment_record["rule_id"],
        "target_category": judgment_record["target_category"],
        "difficulty": judgment_record["difficulty"],

        "original_rule": judgment_record["original_rule"],
        "generated_rule": judgment_record["generated_rule"],

        "generator_model": judgment_record["generator_model"],
        "judge_provider": judgment_record["judge_provider"],
        "judge_model": judgment_record["judge_model"],

        "invalid_reason": judgment_record["invalid_reason"],
        "IntentPreserved": judgment_record["IntentPreserved"],
        "ComfortSafetyPenalty": judgment_record["ComfortSafetyPenalty"],

        "EcoStatic": judgment_record["EcoStatic"],
        "eco_label": judgment_record["eco_label"],
    }


def judge_generation_record(
    generation_record: dict[str, Any],
    judge_provider: str,
    judge_model: str,
    judge_temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    """
    Evaluate one generated rule with one judge model.
    """
    original_rule = str(generation_record["original_rule"]).strip()
    generated_rule = str(generation_record["generated_rule"]).strip()
    rule_id = str(generation_record["rule_id"]).strip()

    judge_output = call_judge_model(
        original_rule=original_rule,
        generated_rule=generated_rule,
        judge_provider=judge_provider,
        judge_model=judge_model,
        rule_id=rule_id,
        temperature=judge_temperature,
        max_retries=max_retries,
    )

    return build_judgment_record(
        generation_record=generation_record,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        judge_output=judge_output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one LLM judge over generated synthetic rules."
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default="exp_001",
        help="Experiment folder name.",
    )

    parser.add_argument(
        "--judge-provider",
        type=str,
        choices=[
            "ollama",
            "openrouter",
        ],
        default="ollama",
        help="Judge provider to use.",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        required=True,
        help=(
            'Judge model name. Examples: '
            '"gemma4:31b-cloud" for Ollama, '
            '"openai/gpt-4o-mini" for OpenRouter, '
            '"google/gemini-2.5-flash" for OpenRouter.'
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_JUDGE_TEMPERATURE,
        help="Judge temperature.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts for each judgment.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of generations to judge.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between judgments.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing judgments and failures for this judge.",
    )

    args = parser.parse_args()

    experiment_dir = DEFAULT_EXPERIMENTS_DIR / args.experiment_id
    generations_path = experiment_dir / "generations.jsonl"

    judges_dir = experiment_dir / "judges"

    judge_folder_name = build_judge_folder_name(
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )

    judge_model_dir = judges_dir / judge_folder_name

    judge_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = judge_model_dir / "judgments.jsonl"
    failures_path = judge_model_dir / "failures.jsonl"
    invalid_path = judge_model_dir / "invalid_generations.jsonl"
    invalid_ids_path = judge_model_dir / "invalid_rule_ids.txt"

    if args.force:
        if output_path.exists():
            output_path.unlink()

        if failures_path.exists():
            failures_path.unlink()

        if invalid_path.exists():
            invalid_path.unlink()

        if invalid_ids_path.exists():
            invalid_ids_path.unlink()

    generation_records = load_jsonl(generations_path)

    if args.limit is not None:
        generation_records = generation_records[: args.limit]

    processed_eval_keys = get_processed_eval_keys(output_path)

    print(f"Loaded generation records: {len(generation_records)}")
    print(f"Already judged: {len(processed_eval_keys)}")
    print(f"Judge provider: {args.judge_provider}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge folder: {judge_model_dir}")
    print(f"Output path: {output_path}")
    print(f"Invalid generations path: {invalid_path}")
    print(f"Invalid rule IDs path: {invalid_ids_path}")

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    invalid_count = 0

    for index, record in enumerate(generation_records, start=1):
        try:
            validate_generation_record(
                record=record,
                index=index,
            )

            eval_key = build_eval_key(record)

            if eval_key in processed_eval_keys:
                skipped_count += 1
                continue

            print(
                f"[{index}/{len(generation_records)}] Judging "
                f"{record['rule_id']} ... ",
                end="",
                flush=True,
            )

            judgment_record = judge_generation_record(
                generation_record=record,
                judge_provider=args.judge_provider,
                judge_model=args.judge_model,
                judge_temperature=args.temperature,
                max_retries=args.max_retries,
            )

            append_jsonl(
                record=judgment_record,
                path=output_path,
            )

            if judgment_record["needs_regeneration"]:
                invalid_count += 1

                invalid_record = build_invalid_generation_record(
                    judgment_record=judgment_record,
                )

                append_jsonl(
                    record=invalid_record,
                    path=invalid_path,
                )

                append_text_line(
                    value=judgment_record["rule_id"],
                    path=invalid_ids_path,
                )

            processed_eval_keys.add(eval_key)
            processed_count += 1

            print(
                f"OK -> {judgment_record['label']} "
                f"(status={judgment_record['generation_status']}, "
                f"EcoStatic={judgment_record['EcoStatic']})"
            )

            time.sleep(args.sleep)

        except Exception as exc:
            failed_count += 1

            failure_record = {
                "created_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "rule_id": record.get("rule_id", ""),
                "target_category": record.get("target_category", ""),
                "difficulty": record.get("difficulty", ""),
                "original_rule": record.get("original_rule", ""),
                "generated_rule": record.get("generated_rule", ""),
                "generator_model": record.get("generator_model", ""),
                "judge_provider": args.judge_provider,
                "judge_model": args.judge_model,
                "error": str(exc),
            }

            append_jsonl(
                record=failure_record,
                path=failures_path,
            )

            print(f"FAILED: {exc}")

    print("\nDone.")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Invalid / needs regeneration: {invalid_count}")
    print(f"Judgments saved to: {output_path}")
    print(f"Failures saved to: {failures_path}")
    print(f"Invalid generations saved to: {invalid_path}")
    print(f"Invalid rule IDs saved to: {invalid_ids_path}")


if __name__ == "__main__":
    main()