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

INPUT_BATCH_DIR = Path("data/batch")
OUTPUT_BATCH_DIR = Path("data/final_batch")


# =========================
# OpenRouter config
# =========================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "openai/gpt-4o-mini"

MAX_RETRIES = 3
SLEEP_BETWEEN_REQUESTS_SECONDS = 1.0


# =========================
# Context config
# =========================

N_CONTEXT_SAMPLES = 10

# If True, existing topic_X_context.json files are recreated.
# If False, existing valid context files are reused.
FORCE_RECREATE_CONTEXT = False


# =========================
# Columns
# =========================

INPUT_COLUMNS = [
    "sample_id",
    "topic_name",
    "topic_text",
]

ALLOWED_GREEN_RELEVANCE = [
    "high",
    "medium",
    "low",
    "none",
]


# =========================
# JSON schema
# =========================

CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "topic_id": {
            "type": "integer",
            "description": "The BERTopic topic id.",
        },
        "llm_topic_name": {
            "type": "string",
            "description": "A concise functional name for the whole topic.",
        },
        "topic_summary": {
            "type": "string",
            "description": "A short summary of what the topic generally contains.",
        },
        "green_category": {
            "type": "string",
            "description": (
                "A concise snake_case category created by the model to describe "
                "the main green or smart-home function of the topic. "
                "Examples: hvac_optimization, voice_thermostat_control, "
                "lighting_efficiency, irrigation_saving, occupancy_based_control."
            ),
        },
        "global_green_relevance": {
            "type": "string",
            "enum": ALLOWED_GREEN_RELEVANCE,
            "description": "The global green relevance of the whole topic.",
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


# =========================
# Env / prompt / client
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
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=load_api_key(),
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
    Parse topic selection from CLI.

    Examples:
      python generate_topic_contexts.py 2
      python generate_topic_contexts.py 2-5
      python generate_topic_contexts.py 1 3 8-12
      python generate_topic_contexts.py outlier
      python generate_topic_contexts.py all
    """
    args = sys.argv[1:]

    if not args:
        raise ValueError(
            "No topic selection provided.\n"
            "Examples:\n"
            "  python generate_topic_contexts.py 2\n"
            "  python generate_topic_contexts.py 2-5\n"
            "  python generate_topic_contexts.py 1 3 8-12\n"
            "  python generate_topic_contexts.py outlier\n"
            "  python generate_topic_contexts.py all"
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


# =========================
# Batch discovery
# =========================

def extract_topic_id_from_path(batch_path: Path) -> int:
    """
    Extract topic_id from paths like:
      data/batch/topic_2/topic_2_batch_001.csv
      data/batch/topic_-1/topic_-1_batch_001.csv
    """
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
            f"Some requested topics were not found in {INPUT_BATCH_DIR}: "
            f"{sorted(missing_topics)}"
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
    required_fields = [
        "topic_id",
        "llm_topic_name",
        "topic_summary",
        "green_category",
        "global_green_relevance",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in context
    ]

    if missing_fields:
        raise ValueError(
            f"Context for topic {topic_id} is missing fields: {missing_fields}"
        )

    if int(context["topic_id"]) != int(topic_id):
        raise ValueError(
            f"Context topic_id mismatch. Expected {topic_id}, "
            f"got {context['topic_id']}"
        )

    green_category = str(context["green_category"]).strip()

    if not green_category:
        raise ValueError("green_category cannot be empty.")

    if not re.fullmatch(r"[a-z][a-z0-9_]*", green_category):
        raise ValueError(
            f"green_category must be snake_case, got: {green_category}"
        )

    if context["global_green_relevance"] not in ALLOWED_GREEN_RELEVANCE:
        raise ValueError(
            f"Invalid global_green_relevance: "
            f"{context['global_green_relevance']}"
        )


# =========================
# Sampling
# =========================

def sample_context_examples_for_topic(
    topic_batch_files: list[Path],
    n_samples: int = N_CONTEXT_SAMPLES,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Select random context examples distributed across all batch files of a topic.

    If a topic has multiple batch files, the examples are spread as evenly
    as possible across them.
    """
    batch_dfs = []

    for batch_path in topic_batch_files:
        batch_df = pd.read_csv(batch_path)
        validate_input_batch(batch_df, batch_path)

        batch_df = batch_df[INPUT_COLUMNS].copy()
        batch_df["source_batch_file"] = str(batch_path)

        batch_dfs.append(batch_df)

    non_empty_batches = [
        df for df in batch_dfs
        if len(df) > 0
    ]

    if not non_empty_batches:
        raise ValueError("Cannot sample context examples from empty topic batches.")

    full_topic_df = pd.concat(non_empty_batches, ignore_index=True)

    if len(full_topic_df) <= n_samples:
        return full_topic_df.reset_index(drop=True)

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

        missing_n = n_samples - len(sampled_df)

        if len(remaining_df) > 0:
            fill_df = remaining_df.sample(
                n=min(missing_n, len(remaining_df)),
                random_state=random_state + 999,
            )

            sampled_df = pd.concat([sampled_df, fill_df], ignore_index=True)

    return sampled_df.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)


# =========================
# Prompt construction
# =========================

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


# =========================
# OpenRouter call
# =========================

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
                        "name": "green_topic_context",
                        "strict": True,
                        "schema": CONTEXT_SCHEMA,
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    )

    validate_context_output(context, topic_id)

    write_json(output_path, context)

    print(f"Saved topic context: {output_path}")

    time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

    return context


# =========================
# Main
# =========================

def main() -> None:
    OUTPUT_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    context_prompt = load_prompt(CONTEXT_PROMPT_PATH)
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

        get_or_create_topic_context(
            client=client,
            context_prompt=context_prompt,
            topic_id=topic_id,
            topic_batch_files=topic_batch_files,
        )


if __name__ == "__main__":
    main()