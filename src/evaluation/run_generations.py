from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from generation.rag_pipeline.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
)
from generation.rag_pipeline.generation import generate_green_rule
from generation.rag_pipeline.postprocessing import postprocess_context
from generation.rag_pipeline.retrieval import retrieve_context


DEFAULT_INPUT_PATH = (
    ROOT / "data" / "synthetic_evaluation" / "synthetic_green_rules_360.jsonl"
)

DEFAULT_EXPERIMENTS_DIR = (
    ROOT / "data" / "synthetic_evaluation"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def get_processed_rule_ids(output_path: Path) -> set[str]:
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

            rule_id = record.get("rule_id")

            if rule_id:
                processed.add(str(rule_id))

    return processed


def extract_generated_rule(generation: dict[str, Any]) -> str:
    parsed_response = generation.get("parsed_response", {})

    generated_rule = (
        parsed_response.get("improved_rule")
        or parsed_response.get("generated_rule")
    )

    if not generated_rule:
        raise ValueError(
            "Could not extract generated rule. "
            "Expected field 'improved_rule' in parsed_response."
        )

    return str(generated_rule).strip()


def build_generation_record(
    source_record: dict[str, Any],
    generation: dict[str, Any],
    generated_rule: str,
) -> dict[str, Any]:
    parsed_response = generation.get("parsed_response", {})

    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rule_id": source_record["rule_id"],
        "target_category": source_record["target_category"],
        "difficulty": source_record["difficulty"],
        "source_distribution_count": source_record.get(
            "source_distribution_count",
            None,
        ),
        "original_rule": source_record["original_rule"],
        "generated_rule": generated_rule,
        "generator_model": LLM_MODEL_NAME,
        "generator_temperature": LLM_TEMPERATURE,
        "green_strategy": parsed_response.get("green_strategy", ""),
        "preserved_intent_generation": parsed_response.get("preserved_intent", None),
        "generation_explanation": parsed_response.get("explanation", ""),
        "used_context_ids": parsed_response.get("used_context_ids", []),
        "raw_generation_response": generation.get("raw_response", ""),
        "parsed_generation_response": parsed_response,
    }


def run_generation_for_record(record: dict[str, Any]) -> dict[str, Any]:
    original_rule = str(record["original_rule"]).strip()

    retrieved = retrieve_context(
        query=original_rule,
    )

    postprocessed = postprocess_context(
        query=original_rule,
        retrieved=retrieved,
    )

    generation = generate_green_rule(
        query=original_rule,
        postprocessed=postprocessed,
    )

    generated_rule = extract_generated_rule(
        generation=generation,
    )

    return build_generation_record(
        source_record=record,
        generation=generation,
        generated_rule=generated_rule,
    )


def validate_input_record(record: dict[str, Any], index: int) -> None:
    required_fields = [
        "rule_id",
        "target_category",
        "difficulty",
        "original_rule",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run green-rule generation over the synthetic dataset."
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default="exp_001",
        help="Experiment folder name.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to process.",
    )

    args = parser.parse_args()

    experiment_dir = DEFAULT_EXPERIMENTS_DIR / args.experiment_id
    output_path = experiment_dir / "generations.jsonl"
    failures_path = experiment_dir / "generation_failures.jsonl"

    records = load_jsonl(DEFAULT_INPUT_PATH)

    if args.limit is not None:
        records = records[: args.limit]

    processed_rule_ids = get_processed_rule_ids(output_path)

    print(f"Input path: {DEFAULT_INPUT_PATH}")
    print(f"Loaded records: {len(records)}")
    print(f"Already processed: {len(processed_rule_ids)}")
    print(f"Output path: {output_path}")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for index, record in enumerate(records, start=1):
        try:
            validate_input_record(
                record=record,
                index=index,
            )

            rule_id = str(record["rule_id"])

            if rule_id in processed_rule_ids:
                skipped_count += 1
                continue

            print(
                f"[{index}/{len(records)}] Generating {rule_id} ... ",
                end="",
                flush=True,
            )

            generation_record = run_generation_for_record(record)

            append_jsonl(
                record=generation_record,
                path=output_path,
            )

            processed_rule_ids.add(rule_id)
            processed_count += 1

            print("OK")

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
    print(f"Generations saved to: {output_path}")
    print(f"Failures saved to: {failures_path}")


if __name__ == "__main__":
    main()