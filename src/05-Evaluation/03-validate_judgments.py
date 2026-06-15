"""
03-validate_judgments.py
=========================
Offline analysis of the judgments produced by 02-run_llm_judge.py.

This script does two things:
    1. Integrity check — ensures every judgment is schema-valid and that the
                         EcoStatic and label stored by the script are consistent
                         with the feature scores returned by the LLM.
    2. Statistics report — label distribution, EcoStatic distribution,
                           per-feature score averages for both r and r'.

Run this after 02-run_llm_judge.py to verify the pipeline output before
drawing conclusions.

Input
-----
data/dataset-rules/evaluation/judgments.jsonl

Output
------
data/dataset-rules/evaluation/report.txt  (also printed to stdout)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import jsonschema


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]

JUDGE_SCHEMA_PATH = SCRIPT_DIR / "judge_schema.json"

INPUT_PATH = ROOT / "data" / "dataset-rules" / "evaluation" / "judgments.jsonl"
REPORT_PATH = ROOT / "data" / "dataset-rules" / "evaluation" / "report.txt"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Judgments file not found: {path}\n"
            "Run 02-run_llm_judge.py first."
        )
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------


def _s_awareness(fs: dict) -> float:
    return 0.6 * fs["OccupancyScore"] / 2 + 0.4 * fs["EnvConditionScore"]


def _s_energy(fs: dict) -> float:
    return (
        0.4 * fs["DurationScore"]
        + 0.4 * fs["StandbyScore"] / 2
        + 0.1 * fs["ApplianceScore"] / 2
        + 0.1 * fs["SchedulingScore"] / 2
    )


def _expected_label(eco: float, ip: int, csp: int) -> str:
    if ip == 0 or csp == 1:
        return "reject"
    if eco >= 0.2:
        return "strong_green"
    if eco > 0:
        return "accept"
    return "reject"


def check_integrity(judgments: list[dict], schema: dict) -> list[str]:
    """Validate every judgment against the schema and verify numeric consistency.

    Returns a list of warning strings (empty list → all clean).
    """
    warnings: list[str] = []

    for j in judgments:
        eval_id = j.get("eval_id", "UNKNOWN")

        # --- Schema validation on the raw LLM response ---
        raw = j.get("raw_llm_response", {})
        try:
            jsonschema.validate(instance=raw, schema=schema)
        except jsonschema.ValidationError as exc:
            warnings.append(f"[SCHEMA] {eval_id}: {exc.message}")
            continue

        # --- Recompute EcoStatic and label, compare with stored values ---
        try:
            orig_fs = raw["original_rule_analysis"]["feature_scores"]
            gen_fs  = raw["generated_rule_analysis"]["feature_scores"]
            validity = raw["validity_checks"]

            recomputed_eco = round(
                0.5 * (_s_awareness(gen_fs) - _s_awareness(orig_fs))
                + 0.5 * (_s_energy(gen_fs) - _s_energy(orig_fs)),
                6,
            )

            stored_eco = j.get("scores", {}).get("EcoStatic")
            if stored_eco is not None and abs(stored_eco - recomputed_eco) > 1e-5:
                warnings.append(
                    f"[ECOSTATIC] {eval_id}: stored={stored_eco}, "
                    f"recomputed={recomputed_eco}"
                )

            ip  = validity["IntentPreserved"]
            csp = validity["ComfortSafetyPenalty"]
            stored_label   = j.get("label")
            expected_label = _expected_label(recomputed_eco, ip, csp)

            if stored_label != expected_label:
                warnings.append(
                    f"[LABEL] {eval_id}: stored={stored_label}, "
                    f"expected={expected_label} (EcoStatic={recomputed_eco})"
                )

        except (KeyError, TypeError) as exc:
            warnings.append(f"[SCORES] {eval_id}: {exc}")

    return warnings


# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------


def _avg(values: list[float]) -> str:
    return f"{sum(values)/len(values):.4f}" if values else "n/a"


def compute_stats(judgments: list[dict]) -> str:
    """Build a human-readable statistics report string."""
    lines: list[str] = ["=" * 60, "EVALUATION REPORT", "=" * 60, ""]

    total = len(judgments)
    lines.append(f"Total judgments: {total}")
    lines.append("")

    # Label distribution
    label_counts = Counter(j.get("label", "unknown") for j in judgments)
    lines.append("Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / total if total else 0.0
        lines.append(f"  {label:<15} {count:>5}  ({pct:.1f}%)")
    lines.append("")

    # EcoStatic distribution
    eco_values = [
        j["scores"]["EcoStatic"]
        for j in judgments
        if "scores" in j and "EcoStatic" in j["scores"]
    ]
    if eco_values:
        lines.append("EcoStatic distribution:")
        lines.append(f"  min  = {min(eco_values):.4f}")
        lines.append(f"  max  = {max(eco_values):.4f}")
        lines.append(f"  mean = {_avg(eco_values)}")
        lines.append(f"  n    = {len(eco_values)}")
        lines.append("")

    # Validity checks summary
    n_intent_failed  = sum(1 for j in judgments if j.get("scores", {}).get("IntentPreserved") == 0)
    n_safety_failed  = sum(1 for j in judgments if j.get("scores", {}).get("ComfortSafetyPenalty") == 1)
    lines.append("Validity checks:")
    lines.append(f"  IntentPreserved=0     : {n_intent_failed} ({100*n_intent_failed/total:.1f}%)")
    lines.append(f"  ComfortSafetyPenalty=1: {n_safety_failed} ({100*n_safety_failed/total:.1f}%)")
    lines.append("")

    # Per-feature average scores — original rule (r) and generated variant (r')
    feature_fields = [
        "OccupancyScore",
        "EnvConditionScore",
        "DurationScore",
        "SchedulingScore",
        "StandbyScore",
        "ApplianceScore",
    ]
    lines.append(f"{'Feature':<25}  {'avg(r)':>8}  {'avg(r\\')':<8}")
    lines.append("-" * 45)
    for field in feature_fields:
        orig_vals = [
            j["raw_llm_response"]["original_rule_analysis"]["feature_scores"].get(field, 0)
            for j in judgments
            if "raw_llm_response" in j
        ]
        gen_vals = [
            j["raw_llm_response"]["generated_rule_analysis"]["feature_scores"].get(field, 0)
            for j in judgments
            if "raw_llm_response" in j
        ]
        lines.append(f"  {field:<23}  {_avg(orig_vals):>8}  {_avg(gen_vals):<8}")
    lines.append("")

    # Delta summary
    delta_awareness_vals = [j["scores"]["Delta_awareness"] for j in judgments if "scores" in j]
    delta_energy_vals    = [j["scores"]["Delta_energy"]    for j in judgments if "scores" in j]
    if delta_awareness_vals:
        lines.append("Mean deltas (r' minus r):")
        lines.append(f"  Delta_awareness = {_avg(delta_awareness_vals)}")
        lines.append(f"  Delta_energy    = {_avg(delta_energy_vals)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    schema    = json.loads(JUDGE_SCHEMA_PATH.read_text(encoding="utf-8"))
    judgments = load_jsonl(INPUT_PATH)
    print(f"Loaded {len(judgments)} judgments from {INPUT_PATH}\n")

    # 1. Integrity check
    print("--- Integrity check ---")
    warnings = check_integrity(judgments, schema)
    if warnings:
        print(f"Found {len(warnings)} issue(s):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("All judgments passed integrity check. ✓")

    # 2. Statistics report
    print("\n--- Statistics ---")
    report = compute_stats(judgments)
    print(report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
