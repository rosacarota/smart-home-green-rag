from pathlib import Path
import json

import pandas as pd


FINAL_BATCH_DIR = Path("data/dataset-rules/final_batch")
OUTPUT_DIR = Path("data/dataset-rules/final_dataset")

OUTPUT_JSON_PATH = OUTPUT_DIR / "green_rules_final_dataset.json"

TARGET_GREEN_RELEVANCE = {
    "medium",
    "high",
}

OUTPUT_COLUMNS = [
    "sample_id",
    "llm_topic_name",
    "green_category",
    "if_then_rule",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_batch_files() -> list[Path]:
    batch_files = sorted(
        FINAL_BATCH_DIR.glob("topic_*/topic_*_batch_*.json")
    )

    if not batch_files:
        raise FileNotFoundError(
            f"No batch JSON files found in {FINAL_BATCH_DIR}. "
            "Expected files like data/dataset-rules/final_batch/topic_1/topic_1_batch_001.json"
        )

    return batch_files


def extract_rows_from_batch(batch_path: Path) -> list[dict]:
    parsed = read_json(batch_path)

    topic_context = parsed.get("topic_context", {})
    rows = parsed.get("rows", [])

    if not isinstance(topic_context, dict):
        raise ValueError(f"Invalid or missing topic_context in {batch_path}")

    if not isinstance(rows, list):
        raise ValueError(f"Invalid or missing rows list in {batch_path}")

    llm_topic_name = topic_context.get("llm_topic_name")
    green_category = topic_context.get("green_category")

    extracted_rows = []

    for row in rows:
        green_relevance = str(row.get("green_relevance", "")).strip().lower()

        if green_relevance not in TARGET_GREEN_RELEVANCE:
            continue

        sample_id = row.get("sample_id")
        if_then_rule = row.get("if_then_rule")

        if not sample_id or not if_then_rule:
            continue

        row_llm_topic_name = row.get("llm_topic_name") or llm_topic_name
        row_green_category = row.get("green_category") or green_category

        if not row_llm_topic_name or not row_green_category:
            raise ValueError(
                f"Missing llm_topic_name or green_category for row {sample_id} "
                f"in file {batch_path}"
            )

        extracted_rows.append(
            {
                "sample_id": str(sample_id).strip(),
                "llm_topic_name": str(row_llm_topic_name).strip(),
                "green_category": str(row_green_category).strip(),
                "if_then_rule": str(if_then_rule).strip(),
            }
        )

    return extracted_rows


def build_final_dataset() -> list[dict]:
    batch_files = collect_batch_files()

    all_rows = []

    for batch_path in batch_files:
        batch_rows = extract_rows_from_batch(batch_path)
        all_rows.extend(batch_rows)

    if not all_rows:
        raise ValueError(
            "No rows found with green_relevance equal to 'medium' or 'high'."
        )

    df = pd.DataFrame(all_rows)

    # Perfect match deduplication on if_then_rule.
    # We only strip leading/trailing spaces; no lowercase or semantic normalization.
    df["_dedupe_key"] = df["if_then_rule"].astype(str).str.strip()

    df = df.drop_duplicates(
        subset=["_dedupe_key"],
        keep="first",
    ).copy()

    df = df[OUTPUT_COLUMNS]

    return df.to_dict(orient="records")


def save_json(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "rows": rows
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    batch_files = collect_batch_files()
    print(f"Found batch JSON files: {len(batch_files)}")

    rows = build_final_dataset()
    save_json(rows)

    print("Done.")
    print(f"Final JSON saved at: {OUTPUT_JSON_PATH}")
    print(f"Final unique rows: {len(rows)}")


if __name__ == "__main__":
    main()