from pathlib import Path
import json
import os
import re
import sys
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_PROMPT_PATH = SCRIPT_DIR / "prompt_batch.txt"

INPUT_BATCH_DIR = Path("data/batch")
OUTPUT_BATCH_DIR = Path("data/final_batch")

FULL_DATASET_WITH_IDS_PATH = INPUT_BATCH_DIR / (
    "dataset_candidates_rag_bertopic_clusters_no_smart_home_related_with_ids.csv"
)

ALL_LABELED_OUTPUT_PATH = OUTPUT_BATCH_DIR / "llm_labeled_rules.json"

FINAL_MERGED_OUTPUT_PATH = OUTPUT_BATCH_DIR / (
    "dataset_candidates_rag_bertopic_clusters_no_smart_home_related_llm_labeled.csv"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "openai/gpt-4o-mini"

MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS_SECONDS = 1.0

FORCE_REPROCESS_BATCHES = False


INPUT_COLUMNS = [
    "sample_id",
    "topic_name",
    "topic_text",
]

# Columns returned by the LLM and saved inside each batch JSON under "rows".
# These are intentionally minimal because the topic context is saved once
# at the beginning of each batch JSON file.
BATCH_ROW_COLUMNS = [
    "sample_id",
    "if_then_rule",
    "green_relevance",
    "keep_for_rag",
]

# Columns used for cumulative outputs:
# - data/final_batch/llm_labeled_rules.json
# - data/final_batch/dataset_candidates_rag_bertopic_clusters_no_smart_home_related_llm_labeled.csv
#
# These keep the topic context expanded into each row for easier merging/filtering.
FINAL_OUTPUT_COLUMNS = [
    "sample_id",
    "llm_topic_name",
    "green_category",
    "global_green_relevance",
    "if_then_rule",
    "green_relevance",
    "keep_for_rag",
]

ALLOWED_GREEN_RELEVANCE = [
    "high",
    "medium",
    "low",
    "none",
]


BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                    },
                    "if_then_rule": {
                        "type": "string",
                    },
                    "green_relevance": {
                        "type": "string",
                        "enum": ALLOWED_GREEN_RELEVANCE,
                    },
                    "keep_for_rag": {
                        "type": "boolean",
                    },
                },
                "required": [
                    "sample_id",
                    "if_then_rule",
                    "green_relevance",
                    "keep_for_rag",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


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


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path.read_text(encoding="utf-8").strip()


def build_client() -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=load_api_key(),
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Smart Home Green RAG",
        },
    )


def parse_topic_selection_from_cli() -> set[int] | None:
    args = sys.argv[1:]

    if not args:
        raise ValueError(
            "No topic selection provided.\n"
            "Examples:\n"
            "  python infer_batches_with_context.py 2\n"
            "  python infer_batches_with_context.py 2-5\n"
            "  python infer_batches_with_context.py 1 3 8-12\n"
            "  python infer_batches_with_context.py outlier\n"
            "  python infer_batches_with_context.py all"
        )

    if len(args) == 1 and args[0].lower() == "all":
        return None

    selected_topics: set[int] = set()

    for raw_arg in args:
        arg = raw_arg.strip().lower()

        if not arg:
            continue

        if arg == "all":
            return None

        if arg in {"outlier", "outliers", "-1"}:
            selected_topics.add(-1)
            continue

        if "-" in arg:
            start_text, end_text = arg.split("-", maxsplit=1)

            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid topic interval: {arg}. Expected format like 2-5."
                ) from exc

            if start > end:
                raise ValueError(
                    f"Invalid topic interval: {arg}. Start must be <= end."
                )

            selected_topics.update(range(start, end + 1))
            continue

        try:
            selected_topics.add(int(arg))
        except ValueError as exc:
            raise ValueError(
                f"Invalid topic id: {arg}. "
                "Use an integer, an interval like 2-5, 'outlier', or 'all'."
            ) from exc

    if not selected_topics:
        raise ValueError("No valid topic IDs selected.")

    return selected_topics


def extract_topic_id_from_path(batch_path: Path) -> int:
    match = re.search(r"topic_(-?\d+)", batch_path.parent.name)

    if not match:
        raise ValueError(f"Cannot extract topic_id from path: {batch_path}")

    return int(match.group(1))


def get_all_batch_files() -> list[Path]:
    batch_files = sorted(INPUT_BATCH_DIR.glob("topic_*/topic_*_batch_*.csv"))

    if not batch_files:
        raise FileNotFoundError(
            f"No topic batch files found in {INPUT_BATCH_DIR}."
        )

    return batch_files


def get_selected_batch_files() -> list[Path]:
    all_files = get_all_batch_files()
    requested_topics = parse_topic_selection_from_cli()

    if requested_topics is None:
        return all_files

    selected_files = [
        path for path in all_files
        if extract_topic_id_from_path(path) in requested_topics
    ]

    found_topics = {
        extract_topic_id_from_path(path)
        for path in selected_files
    }

    missing_topics = requested_topics - found_topics

    if missing_topics:
        raise FileNotFoundError(
            f"Some requested topics were not found: {sorted(missing_topics)}"
        )

    return selected_files


def group_batch_files_by_topic(batch_files: list[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}

    for batch_path in batch_files:
        topic_id = extract_topic_id_from_path(batch_path)
        grouped.setdefault(topic_id, []).append(batch_path)

    for topic_id in grouped:
        grouped[topic_id] = sorted(grouped[topic_id])

    return dict(sorted(grouped.items()))


def topic_output_dir(topic_id: int) -> Path:
    return OUTPUT_BATCH_DIR / f"topic_{topic_id}"


def context_output_path(topic_id: int) -> Path:
    return topic_output_dir(topic_id) / f"topic_{topic_id}_context.json"


def batch_output_path(batch_path: Path) -> Path:
    relative_path = batch_path.relative_to(INPUT_BATCH_DIR)
    return (OUTPUT_BATCH_DIR / relative_path).with_suffix(".json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_topic_context(topic_id: int) -> dict[str, Any]:
    path = context_output_path(topic_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing context file for topic {topic_id}: {path}\n"
            f"Run generate_topic_contexts.py {topic_id} first."
        )

    context = read_json(path)

    required_context_fields = [
        "topic_id",
        "llm_topic_name",
        "topic_summary",
        "green_category",
        "global_green_relevance",
    ]

    missing_fields = [
        field for field in required_context_fields
        if field not in context
    ]

    if missing_fields:
        raise ValueError(
            f"Context file {path} is missing fields: {missing_fields}"
        )

    return context


def validate_input_batch(batch_df: pd.DataFrame, batch_path: Path) -> None:
    missing_cols = [col for col in INPUT_COLUMNS if col not in batch_df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing columns in batch file {batch_path}: {missing_cols}"
        )

    if batch_df["sample_id"].duplicated().any():
        duplicated = batch_df.loc[
            batch_df["sample_id"].duplicated(),
            "sample_id",
        ].tolist()

        raise ValueError(
            f"Duplicated sample_id values in {batch_path}: {duplicated[:10]}"
        )


def validate_batch_rows_output(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    batch_path: Path,
) -> None:
    missing_cols = [
        col for col in BATCH_ROW_COLUMNS
        if col not in output_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"LLM output for {batch_path} is missing columns: {missing_cols}"
        )

    input_ids = set(input_df["sample_id"].astype(str))
    output_ids = set(output_df["sample_id"].astype(str))

    missing_ids = input_ids - output_ids
    extra_ids = output_ids - input_ids

    if missing_ids:
        raise ValueError(
            f"LLM output for {batch_path} is missing sample_id values: "
            f"{list(missing_ids)[:10]}"
        )

    if extra_ids:
        raise ValueError(
            f"LLM output for {batch_path} returned unknown sample_id values: "
            f"{list(extra_ids)[:10]}"
        )

    invalid_relevance = sorted(
        set(output_df["green_relevance"].astype(str)) - set(ALLOWED_GREEN_RELEVANCE)
    )

    if invalid_relevance:
        raise ValueError(
            f"LLM output for {batch_path} contains invalid green_relevance values: "
            f"{invalid_relevance}"
        )

    if not output_df["keep_for_rag"].map(lambda x: isinstance(x, bool)).all():
        raise ValueError(
            f"LLM output for {batch_path} contains non-boolean keep_for_rag values."
        )


def validate_final_output(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    batch_path: Path,
) -> None:
    missing_cols = [
        col for col in FINAL_OUTPUT_COLUMNS
        if col not in output_df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Final output for {batch_path} is missing columns: {missing_cols}"
        )

    input_ids = set(input_df["sample_id"].astype(str))
    output_ids = set(output_df["sample_id"].astype(str))

    missing_ids = input_ids - output_ids
    extra_ids = output_ids - input_ids

    if missing_ids:
        raise ValueError(
            f"Final output for {batch_path} is missing sample_id values: "
            f"{list(missing_ids)[:10]}"
        )

    if extra_ids:
        raise ValueError(
            f"Final output for {batch_path} returned unknown sample_id values: "
            f"{list(extra_ids)[:10]}"
        )


def add_context_to_rows(
    rows_df: pd.DataFrame,
    context: dict[str, Any],
) -> pd.DataFrame:
    final_df = rows_df[BATCH_ROW_COLUMNS].copy()

    final_df["llm_topic_name"] = context["llm_topic_name"]
    final_df["green_category"] = context["green_category"]
    final_df["global_green_relevance"] = context["global_green_relevance"]

    final_df = final_df[FINAL_OUTPUT_COLUMNS].copy()

    return final_df


def build_batch_user_prompt(
    batch_df: pd.DataFrame,
    batch_path: Path,
    context: dict[str, Any],
) -> str:
    topic_id = extract_topic_id_from_path(batch_path)

    lines = [
        f"Input file: {batch_path}",
        f"Topic id: {topic_id}",
        "",
        "Fixed topic-level context:",
        f"llm_topic_name: {context['llm_topic_name']}",
        f"topic_summary: {context['topic_summary']}",
        f"green_category: {context['green_category']}",
        f"global_green_relevance: {context['global_green_relevance']}",
        "",
        f"Number of samples in this batch: {len(batch_df)}",
        "",
        "Samples:",
        "",
    ]

    for i, row in enumerate(batch_df.itertuples(index=False), start=1):
        lines.append(f"[{i}]")
        lines.append(f"sample_id: {row.sample_id}")
        lines.append(f"topic_name: {row.topic_name}")
        lines.append(f"topic_text: {row.topic_text}")
        lines.append("")

    lines.append("Return one output row for each sample_id.")
    lines.append("Preserve every sample_id exactly.")

    return "\n".join(lines)


def call_openrouter_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
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
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "green_rule_batch_labeling",
                        "strict": True,
                        "schema": BATCH_SCHEMA,
                    },
                },
                extra_body={
                    "provider": {
                        "require_parameters": True,
                    }
                },
            )

            content = completion.choices[0].message.content

            if not content:
                raise ValueError("Empty response content from model.")

            return json.loads(content)

        except Exception as exc:
            last_error = exc
            wait_seconds = 2 ** attempt

            print(
                f"OpenRouter call failed. Attempt {attempt}/{MAX_RETRIES}. "
                f"Retrying in {wait_seconds}s. Error: {exc}"
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"OpenRouter call failed after {MAX_RETRIES} attempts: {last_error}"
    )


def infer_single_batch(
    client: OpenAI,
    batch_prompt: str,
    batch_path: Path,
    context: dict[str, Any],
) -> pd.DataFrame:
    batch_df = pd.read_csv(batch_path)
    validate_input_batch(batch_df, batch_path)

    output_path = batch_output_path(batch_path)

    if output_path.exists() and not FORCE_REPROCESS_BATCHES:
        existing_json = read_json(output_path)

        existing_context = existing_json.get("topic_context")
        existing_rows = existing_json.get("rows", [])

        try:
            if not isinstance(existing_context, dict):
                raise ValueError("Missing or invalid topic_context.")

            if not isinstance(existing_rows, list):
                raise ValueError("Missing or invalid rows.")

            existing_rows_df = pd.DataFrame(existing_rows)

            validate_batch_rows_output(
                input_df=batch_df,
                output_df=existing_rows_df,
                batch_path=batch_path,
            )

            existing_final_df = add_context_to_rows(
                rows_df=existing_rows_df,
                context=existing_context,
            )

            validate_final_output(
                input_df=batch_df,
                output_df=existing_final_df,
                batch_path=batch_path,
            )

            print(f"Skipping already processed batch: {batch_path}")

            return existing_final_df

        except Exception:
            print(f"Existing output is invalid, reprocessing: {output_path}")

    print(f"Processing: {batch_path} ({len(batch_df)} rows)")

    user_prompt = build_batch_user_prompt(
        batch_df=batch_df,
        batch_path=batch_path,
        context=context,
    )

    parsed_json = call_openrouter_json(
        client=client,
        system_prompt=batch_prompt,
        user_prompt=user_prompt,
    )

    rows_df = pd.DataFrame(parsed_json["rows"])

    validate_batch_rows_output(
        input_df=batch_df,
        output_df=rows_df,
        batch_path=batch_path,
    )

    rows_df = rows_df[BATCH_ROW_COLUMNS].copy()

    final_output_df = add_context_to_rows(
        rows_df=rows_df,
        context=context,
    )

    validate_final_output(
        input_df=batch_df,
        output_df=final_output_df,
        batch_path=batch_path,
    )

    write_json(
        output_path,
        {
            "topic_context": context,
            "rows": rows_df.to_dict(orient="records"),
        },
    )

    print(f"Saved: {output_path}")

    time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    return final_output_df


def collect_all_existing_outputs() -> pd.DataFrame:
    json_files = sorted(OUTPUT_BATCH_DIR.glob("topic_*/topic_*_batch_*.json"))

    if not json_files:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    rows = []

    for json_file in json_files:
        parsed = read_json(json_file)

        context = parsed.get("topic_context")
        file_rows = parsed.get("rows", [])

        if not isinstance(context, dict):
            raise ValueError(f"Missing or invalid topic_context in {json_file}")

        if not isinstance(file_rows, list):
            raise ValueError(f"Invalid rows field in {json_file}")

        for row in file_rows:
            row_with_context = dict(row)
            row_with_context["llm_topic_name"] = context["llm_topic_name"]
            row_with_context["green_category"] = context["green_category"]
            row_with_context["global_green_relevance"] = context[
                "global_green_relevance"
            ]

            rows.append(row_with_context)

    all_df = pd.DataFrame(rows)

    if all_df.empty:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    all_df = all_df[FINAL_OUTPUT_COLUMNS].copy()
    all_df = all_df.drop_duplicates(subset=["sample_id"], keep="first")

    return all_df


def merge_with_full_dataset(all_labeled_df: pd.DataFrame) -> pd.DataFrame:
    if not FULL_DATASET_WITH_IDS_PATH.exists():
        raise FileNotFoundError(
            f"Full dataset with IDs not found: {FULL_DATASET_WITH_IDS_PATH}"
        )

    full_df = pd.read_csv(FULL_DATASET_WITH_IDS_PATH)

    if "sample_id" not in full_df.columns:
        raise ValueError(
            f"Missing sample_id column in {FULL_DATASET_WITH_IDS_PATH}"
        )

    merged_df = full_df.merge(
        all_labeled_df,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    missing_labels = merged_df["if_then_rule"].isna().sum()

    if missing_labels > 0:
        print(f"Warning: {missing_labels} rows have no LLM label yet.")

    return merged_df


def save_global_outputs() -> None:
    all_labeled_df = collect_all_existing_outputs()

    all_labeled_json = {
        "rows": all_labeled_df[FINAL_OUTPUT_COLUMNS].to_dict(orient="records")
    }

    write_json(ALL_LABELED_OUTPUT_PATH, all_labeled_json)

    print(f"Saved cumulative LLM labels at: {ALL_LABELED_OUTPUT_PATH}")
    print(f"Total labeled samples so far: {len(all_labeled_df)}")

    merged_df = merge_with_full_dataset(all_labeled_df)
    merged_df.to_csv(FINAL_MERGED_OUTPUT_PATH, index=False)

    print(f"Saved cumulative merged dataset at: {FINAL_MERGED_OUTPUT_PATH}")
    print(f"Final shape: {merged_df.shape}")

    if "green_relevance" in merged_df.columns:
        print("\nGreen relevance distribution:")
        print(merged_df["green_relevance"].value_counts(dropna=False))

    if "keep_for_rag" in merged_df.columns:
        print("\nKeep for RAG distribution:")
        print(merged_df["keep_for_rag"].value_counts(dropna=False))


def main() -> None:
    OUTPUT_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    batch_prompt = load_prompt(BATCH_PROMPT_PATH)
    client = build_client()

    selected_batch_files = get_selected_batch_files()
    batches_by_topic = group_batch_files_by_topic(selected_batch_files)

    print(f"Model: {MODEL}")
    print(f"Selected topics: {list(batches_by_topic.keys())}")
    print(f"Selected batch files: {len(selected_batch_files)}")

    for topic_id, topic_batch_files in batches_by_topic.items():
        print("\n" + "=" * 80)
        print(f"Topic {topic_id}")
        print(f"Batch files: {len(topic_batch_files)}")

        context = load_topic_context(topic_id)

        for batch_path in topic_batch_files:
            infer_single_batch(
                client=client,
                batch_prompt=batch_prompt,
                batch_path=batch_path,
                context=context,
            )

    save_global_outputs()


if __name__ == "__main__":
    main()