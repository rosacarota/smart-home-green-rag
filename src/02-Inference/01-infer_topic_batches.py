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


# =========================
# Paths
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent

CONTEXT_PROMPT_PATH = SCRIPT_DIR / "prompt_context.txt"
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


# =========================
# OpenRouter config
# =========================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "openai/gpt-4o-mini"

MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS_SECONDS = 1.0

N_CONTEXT_SAMPLES = 10

FORCE_RECREATE_CONTEXT = False
FORCE_REPROCESS_BATCHES = False


# =========================
# Columns
# =========================

INPUT_COLUMNS = [
    "sample_id",
    "topic_name",
    "topic_text",
]

BATCH_MODEL_OUTPUT_COLUMNS = [
    "sample_id",
    "if_then_rule",
    "green_relevance",
    "keep_for_rag",
]

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

ALLOWED_GREEN_CATEGORIES = [
    "lighting_saving",
    "hvac_optimization",
    "standby_reduction",
    "water_saving",
    "appliance_scheduling",
    "energy_monitoring",
    "comfort_security",
    "weak_or_irrelevant",
]


# =========================
# JSON schemas
# =========================

CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic_id": {
            "type": "integer",
        },
        "llm_topic_name": {
            "type": "string",
            "description": "Concise functional name for the whole topic.",
        },
        "topic_summary": {
            "type": "string",
            "description": "Short description of what this topic generally contains.",
        },
        "green_category": {
            "type": "string",
            "enum": ALLOWED_GREEN_CATEGORIES,
        },
        "global_green_relevance": {
            "type": "string",
            "enum": ALLOWED_GREEN_RELEVANCE,
        },
    },
    "required": [
        "topic_id",
        "llm_topic_name",
        "topic_summary",
        "green_category",
        "global_green_relevance",
    ],
    "additionalProperties": False,
}


BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "description": "One output row for each input sample.",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "The original sample_id copied exactly from the input.",
                    },
                    "if_then_rule": {
                        "type": "string",
                        "description": "Rule rewritten as IF <trigger/context> THEN <action>.",
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


# =========================
# Env / prompts / client
# =========================

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
    api_key = load_api_key()

    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "Smart Home Green RAG",
        },
    )


# =========================
# CLI topic selection
# =========================

def parse_topic_selection_from_cli() -> set[int] | None:
    """
    Examples:
    python infer_topic_batches.py 2
        -> {2}

    python infer_topic_batches.py 2-5
        -> {2, 3, 4, 5}

    python infer_topic_batches.py 1 3 8-10
        -> {1, 3, 8, 9, 10}

    python infer_topic_batches.py outlier
        -> {-1}

    python infer_topic_batches.py all
        -> None
    """
    args = sys.argv[1:]

    if not args:
        raise ValueError(
            "No topic selection provided.\n"
            "Usage examples:\n"
            "  python infer_topic_batches.py 2\n"
            "  python infer_topic_batches.py 2-5\n"
            "  python infer_topic_batches.py 1 3 8-12\n"
            "  python infer_topic_batches.py outlier\n"
            "  python infer_topic_batches.py all"
        )

    if len(args) == 1 and args[0].lower() == "all":
        return None

    selected_topics: set[int] = set()

    for arg in args:
        arg = arg.strip().lower()

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
            f"No topic batch files found in {INPUT_BATCH_DIR}. "
            "Expected files like data/batch/topic_0/topic_0_batch_001.csv"
        )

    return batch_files


def get_selected_batch_files() -> list[Path]:
    batch_files = get_all_batch_files()
    requested_topic_ids = parse_topic_selection_from_cli()

    if requested_topic_ids is None:
        return batch_files

    selected_files = []

    for batch_path in batch_files:
        topic_id = extract_topic_id_from_path(batch_path)

        if topic_id in requested_topic_ids:
            selected_files.append(batch_path)

    found_topic_ids = {
        extract_topic_id_from_path(path)
        for path in selected_files
    }

    missing_topic_ids = requested_topic_ids - found_topic_ids

    if missing_topic_ids:
        raise FileNotFoundError(
            f"Some requested topics were not found in {INPUT_BATCH_DIR}: "
            f"{sorted(missing_topic_ids)}"
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


# =========================
# Validation
# =========================

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


def validate_context_output(context: dict[str, Any], topic_id: int) -> None:
    required = [
        "topic_id",
        "llm_topic_name",
        "topic_summary",
        "green_category",
        "global_green_relevance",
    ]

    missing = [field for field in required if field not in context]

    if missing:
        raise ValueError(f"Context for topic {topic_id} is missing fields: {missing}")

    if int(context["topic_id"]) != int(topic_id):
        raise ValueError(
            f"Context topic_id mismatch. Expected {topic_id}, got {context['topic_id']}"
        )

    if context["green_category"] not in ALLOWED_GREEN_CATEGORIES:
        raise ValueError(
            f"Invalid green_category for topic {topic_id}: {context['green_category']}"
        )

    if context["global_green_relevance"] not in ALLOWED_GREEN_RELEVANCE:
        raise ValueError(
            f"Invalid global_green_relevance for topic {topic_id}: "
            f"{context['global_green_relevance']}"
        )


def validate_batch_model_output(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    batch_path: Path,
) -> None:
    missing_cols = [
        col for col in BATCH_MODEL_OUTPUT_COLUMNS
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
        set(output_df["green_relevance"].astype(str))
        - set(ALLOWED_GREEN_RELEVANCE)
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


# =========================
# Prompt construction
# =========================

def sample_context_examples_for_topic(
    topic_batch_files: list[Path],
    n_samples: int = N_CONTEXT_SAMPLES,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select random context examples distributed across all batch files of a topic.

    If a topic has multiple batch files, the 10 examples are spread as evenly
    as possible across them.
    """
    batch_dfs = []

    for batch_path in topic_batch_files:
        batch_df = pd.read_csv(batch_path)
        validate_input_batch(batch_df, batch_path)
        batch_df = batch_df[INPUT_COLUMNS].copy()
        batch_df["source_batch_file"] = str(batch_path)
        batch_dfs.append(batch_df)

    non_empty_batches = [df for df in batch_dfs if len(df) > 0]

    if not non_empty_batches:
        raise ValueError("Cannot sample context examples from empty topic batches.")

    full_topic_df = pd.concat(non_empty_batches, ignore_index=True)

    if len(full_topic_df) <= n_samples:
        return full_topic_df

    n_files = len(non_empty_batches)
    base_quota = n_samples // n_files
    remainder = n_samples % n_files

    selected_parts = []

    for idx, batch_df in enumerate(non_empty_batches):
        quota = base_quota + (1 if idx < remainder else 0)
        quota = min(quota, len(batch_df))

        if quota <= 0:
            continue

        selected_parts.append(
            batch_df.sample(
                n=quota,
                random_state=random_state + idx,
            )
        )

    sampled_df = pd.concat(selected_parts, ignore_index=True)

    if len(sampled_df) < n_samples:
        already_selected = set(sampled_df["sample_id"].astype(str))

        remaining_df = full_topic_df[
            ~full_topic_df["sample_id"].astype(str).isin(already_selected)
        ]

        n_missing = n_samples - len(sampled_df)

        if len(remaining_df) > 0:
            fill_df = remaining_df.sample(
                n=min(n_missing, len(remaining_df)),
                random_state=random_state + 999,
            )

            sampled_df = pd.concat([sampled_df, fill_df], ignore_index=True)

    return sampled_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)


def build_context_user_prompt(
    topic_id: int,
    context_samples_df: pd.DataFrame,
) -> str:
    topic_name = str(context_samples_df["topic_name"].iloc[0])

    lines = [
        f"Topic id: {topic_id}",
        f"Automatic BERTopic topic name: {topic_name}",
        f"Number of representative samples: {len(context_samples_df)}",
        "",
        "Representative samples:",
        "",
    ]

    for i, row in enumerate(context_samples_df.itertuples(index=False), start=1):
        lines.append(f"[{i}]")
        lines.append(f"sample_id: {row.sample_id}")
        lines.append(f"topic_name: {row.topic_name}")
        lines.append(f"topic_text: {row.topic_text}")
        lines.append("")

    lines.append("Infer the topic-level context for this BERTopic cluster.")

    return "\n".join(lines)


def build_batch_user_prompt(
    batch_df: pd.DataFrame,
    batch_path: Path,
    context: dict[str, Any],
) -> str:
    topic_id = extract_topic_id_from_path(batch_path)
    n_samples = len(batch_df)

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
        f"Number of samples in this batch: {n_samples}",
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


# =========================
# OpenRouter call
# =========================

def call_openrouter_json(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    response_schema: dict[str, Any],
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
                        "name": schema_name,
                        "strict": True,
                        "schema": response_schema,
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


# =========================
# Output paths / IO
# =========================

def topic_output_dir(topic_id: int) -> Path:
    return OUTPUT_BATCH_DIR / f"topic_{topic_id}"


def context_output_path(topic_id: int) -> Path:
    return topic_output_dir(topic_id) / f"topic_{topic_id}_context.json"


def batch_output_path(batch_path: Path) -> Path:
    """
    Input:
    data/batch/topic_2/topic_2_batch_001.csv

    Output:
    data/final_batch/topic_2/topic_2_batch_001.json
    """
    relative_path = batch_path.relative_to(INPUT_BATCH_DIR)
    output_path = OUTPUT_BATCH_DIR / relative_path
    return output_path.with_suffix(".json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# Context creation
# =========================

def get_or_create_topic_context(
    client: OpenAI,
    context_prompt: str,
    topic_id: int,
    topic_batch_files: list[Path],
) -> dict[str, Any]:
    output_path = context_output_path(topic_id)

    if output_path.exists() and not FORCE_RECREATE_CONTEXT:
        context = read_json(output_path)
        validate_context_output(context, topic_id)
        print(f"Using existing context: {output_path}")
        return context

    print(f"Creating topic context for topic {topic_id}")

    context_samples_df = sample_context_examples_for_topic(
        topic_batch_files=topic_batch_files,
        n_samples=N_CONTEXT_SAMPLES,
    )

    user_prompt = build_context_user_prompt(
        topic_id=topic_id,
        context_samples_df=context_samples_df,
    )

    context = call_openrouter_json(
        client=client,
        system_prompt=context_prompt,
        user_prompt=user_prompt,
        schema_name="green_topic_context",
        response_schema=CONTEXT_SCHEMA,
    )

    validate_context_output(context, topic_id)

    write_json(output_path, context)

    print(f"Saved topic context: {output_path}")

    time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    return context


# =========================
# Batch inference
# =========================

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
        existing_df = pd.DataFrame(existing_json["rows"])

        try:
            validate_final_output(batch_df, existing_df, batch_path)
            print(f"Skipping already processed batch: {batch_path}")
            return existing_df[FINAL_OUTPUT_COLUMNS].copy()
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
        schema_name="green_rule_batch_labeling",
        response_schema=BATCH_SCHEMA,
    )

    model_output_df = pd.DataFrame(parsed_json["rows"])

    validate_batch_model_output(batch_df, model_output_df, batch_path)

    output_df = model_output_df[BATCH_MODEL_OUTPUT_COLUMNS].copy()

    # Add fixed topic-level context to each row.
    output_df["llm_topic_name"] = context["llm_topic_name"]
    output_df["green_category"] = context["green_category"]
    output_df["global_green_relevance"] = context["global_green_relevance"]

    output_df = output_df[FINAL_OUTPUT_COLUMNS].copy()

    write_json(
        output_path,
        {
            "topic_context": context,
            "rows": output_df.to_dict(orient="records"),
        },
    )

    print(f"Saved: {output_path}")

    time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    return output_df


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
            f"Output for {batch_path} is missing columns: {missing_cols}"
        )

    input_ids = set(input_df["sample_id"].astype(str))
    output_ids = set(output_df["sample_id"].astype(str))

    missing_ids = input_ids - output_ids
    extra_ids = output_ids - input_ids

    if missing_ids:
        raise ValueError(
            f"Output for {batch_path} is missing sample_id values: "
            f"{list(missing_ids)[:10]}"
        )

    if extra_ids:
        raise ValueError(
            f"Output for {batch_path} returned unknown sample_id values: "
            f"{list(extra_ids)[:10]}"
        )


# =========================
# Aggregation / merge
# =========================

def collect_all_existing_outputs() -> pd.DataFrame:
    json_files = sorted(OUTPUT_BATCH_DIR.glob("topic_*/topic_*_batch_*.json"))

    if not json_files:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    rows = []

    for json_file in json_files:
        parsed = read_json(json_file)

        file_rows = parsed.get("rows", [])

        if not isinstance(file_rows, list):
            raise ValueError(f"Invalid rows field in {json_file}")

        rows.extend(file_rows)

    all_df = pd.DataFrame(rows)

    if all_df.empty:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)

    all_df = all_df[FINAL_OUTPUT_COLUMNS].copy()

    all_df = all_df.drop_duplicates(
        subset=["sample_id"],
        keep="first",
    )

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


# =========================
# Main
# =========================

def main() -> None:
    OUTPUT_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    context_prompt = load_prompt(CONTEXT_PROMPT_PATH)
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

        context = get_or_create_topic_context(
            client=client,
            context_prompt=context_prompt,
            topic_id=topic_id,
            topic_batch_files=topic_batch_files,
        )

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