"""
02-run_llm_judge.py
====================
Runs the LLM-as-judge pipeline over the evaluation inputs produced by
01-build_eval_inputs.py.

For each rule pair the script:
    1. Formats the judge prompt.
    2. Calls the LLM via OpenRouter and validates the raw response against
       judge_schema.json.
    3. Computes all numeric eco-metric scores deterministically from the
       feature scores returned by the LLM.
    4. Assigns the final label deterministcally (reject / accept / strong_green).
    5. Appends the enriched record to judgments.jsonl.

The numeric computation is intentionally kept in this script (not delegated to
the LLM) so that the mathematical part is fully reproducible.

Input
-----
data/dataset-rules/evaluation/eval_inputs.jsonl

Output
------
data/dataset-rules/evaluation/judgments.jsonl
    One JSON object per line, containing:
        eval_id, rule_id, original_rule, generated_rule,
        raw_llm_response (dict),
        scores (all computed metrics),
        label, label_source ("script")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jsonschema
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

JUDGE_PROMPT_PATH = SCRIPT_DIR / "judge_prompt.txt"
JUDGE_SCHEMA_PATH = SCRIPT_DIR / "judge_schema.json"

INPUT_PATH = ROOT / "data" / "dataset-rules" / "evaluation" / "eval_inputs.jsonl"
OUTPUT_PATH = ROOT / "data" / "dataset-rules" / "evaluation" / "judgments.jsonl"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"          # TODO: change to a stronger model if needed
MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS = 1.0

# If True, re-judge records that already have an entry in judgments.jsonl.
FORCE_REPROCESS = False

# ---------------------------------------------------------------------------
# Eco-metric computation (deterministic — no LLM involvement)
# ---------------------------------------------------------------------------


def compute_scores(
    orig: dict[str, int],
    gen: dict[str, int],
    validity: dict[str, int],
) -> dict[str, float | int | str]:
    """Compute all eco-metric scores from the LLM feature scores.

    Parameters
    ----------
    orig:
        feature_scores block from original_rule_analysis.
    gen:
        feature_scores block from generated_rule_analysis.
    validity:
        validity_checks block.

    Returns
    -------
    dict with all intermediate and final numeric scores, plus the label.
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
        return 0.4 * dur_norm + 0.4 * std_norm + 0.1 * app_norm + 0.1 * sch_norm

    saw_orig = s_awareness(orig)
    saw_gen = s_awareness(gen)
    sen_orig = s_energy(orig)
    sen_gen = s_energy(gen)

    delta_awareness = saw_gen - saw_orig
    delta_energy = sen_gen - sen_orig
    eco_static = 0.5 * delta_awareness + 0.5 * delta_energy

    intent_preserved: int = validity["IntentPreserved"]
    comfort_safety_penalty: int = validity["ComfortSafetyPenalty"]

    # Deterministic label assignment
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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    load_dotenv()
    api_key = (
        os.getenv("OPEN-ROUTER-KEY")
        or os.getenv("OPEN_ROUTER_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "Missing OpenRouter API key. Add it to .env as OPEN-ROUTER-KEY=..."
        )
    return api_key


def build_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=load_api_key(),
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Smart Home Green RAG — Judge",
        },
    )


def build_user_prompt(record: dict) -> str:
    """Format the user-facing part of the judge prompt.

    Only the original rule and the generated variant are passed to the judge.
    The retrieved context is intentionally excluded: the judge must evaluate
    the rule pair on its own merits, not on the sources the RAG used to
    generate it (which would bias the evaluation).
    """
    lines = [
        f"rule_id: {record['rule_id']}",
        "",
        "Original rule (r):",
        record["original_rule"],
        "",
        "Generated green variant (r'):",
        record["generated_rule"],
        "",
        "Return only valid JSON matching the schema.",
    ]
    return "\n".join(lines)



def call_llm_judge(
    client: OpenAI,
    system_prompt: str,
    judge_schema: dict,
    record: dict,
) -> dict:
    """Call the LLM and return the parsed JSON response.

    Retries up to MAX_RETRIES times with exponential back-off.
    """
    user_prompt = build_user_prompt(record)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "eco_metric_judgment",
                        "strict": True,
                        "schema": judge_schema,
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )

            content = completion.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model.")

            return json.loads(content)

        except Exception as exc:
            last_error = exc
            wait = 2 ** attempt
            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed for {record['eval_id']}: "
                f"{exc}. Retrying in {wait}s…"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"LLM judge failed after {MAX_RETRIES} attempts for {record['eval_id']}: "
        f"{last_error}"
    )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_against_schema(response: dict, schema: dict, eval_id: str) -> None:
    """Raise jsonschema.ValidationError if the response does not match the schema."""
    try:
        jsonschema.validate(instance=response, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(
            f"LLM response for {eval_id} failed schema validation: {exc.message}"
        ) from exc


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Eval inputs not found: {INPUT_PATH}\n"
            "Run 01-build_eval_inputs.py first."
        )

    system_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()
    judge_schema = json.loads(JUDGE_SCHEMA_PATH.read_text(encoding="utf-8"))

    eval_inputs = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(eval_inputs)} evaluation pairs from {INPUT_PATH}")

    # Build set of already-processed eval_ids to support resumable runs
    existing_judgments = load_jsonl(OUTPUT_PATH)
    processed_ids: set[str] = {r["eval_id"] for r in existing_judgments}
    if processed_ids and not FORCE_REPROCESS:
        print(f"Skipping {len(processed_ids)} already-processed records.")

    client = build_client()
    n_processed = 0
    n_skipped = 0
    n_errors = 0

    for record in eval_inputs:
        eval_id: str = record["eval_id"]

        if eval_id in processed_ids and not FORCE_REPROCESS:
            n_skipped += 1
            continue

        print(f"  Judging {eval_id} …", end=" ", flush=True)

        try:
            raw_response = call_llm_judge(
                client=client,
                system_prompt=system_prompt,
                judge_schema=judge_schema,
                record=record,
            )

            validate_against_schema(raw_response, judge_schema, eval_id)

            # Extract feature scores and compute all metrics deterministically
            orig_fs = raw_response["original_rule_analysis"]["feature_scores"]
            gen_fs = raw_response["generated_rule_analysis"]["feature_scores"]
            validity = raw_response["validity_checks"]

            scores = compute_scores(orig_fs, gen_fs, validity)

            judgment = {
                "eval_id": eval_id,
                "rule_id": record["rule_id"],
                "original_rule": record["original_rule"],
                "generated_rule": record["generated_rule"],
                "raw_llm_response": raw_response,
                "scores": scores,
                # label and label_reason are also top-level for easy filtering
                "label": scores["label"],
                "label_reason": scores["label_reason"],
                "label_source": "script",  # numeric decision made here, not by LLM
            }

            append_jsonl(judgment, OUTPUT_PATH)
            n_processed += 1
            print(f"→ {scores['label']} (EcoStatic={scores['EcoStatic']:.4f})")

        except Exception as exc:
            n_errors += 1
            print(f"ERROR: {exc}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(
        f"\nDone. Processed={n_processed}, Skipped={n_skipped}, Errors={n_errors}."
    )
    print(f"Judgments saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
