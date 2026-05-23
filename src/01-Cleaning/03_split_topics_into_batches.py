from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/dataset_candidates_rag_bertopic_clusters_no_smart_home_related.csv"
)

OUTPUT_DIR = Path("data/batch")

FULL_OUTPUT_PATH = OUTPUT_DIR / "dataset_candidates_rag_bertopic_clusters_no_smart_home_related_with_ids.csv"

MAX_ROWS_PER_BATCH = 100


REQUIRED_COLUMNS = [
    "topic_id",
    "topic_name",
    "topic_text",
]


# Columns exported in each topic-specific CSV for the LLM.
# topic_id and numtopic_id are intentionally NOT exported here.
# The raw trigger/action/title/desc columns are not needed because they are already
# included inside topic_text.
COLUMNS_FOR_LLM = [
    "sample_id",
    "topic_name",
    "topic_text",
]


def make_numtopic_id(topic_id: int) -> str:
    """
    Create a readable topic identifier.

    Examples:
    - topic_id = -1 -> T_OUTLIER
    - topic_id = 0  -> T000
    - topic_id = 24 -> T024
    """
    if topic_id == -1:
        return "T_OUTLIER"

    return f"T{topic_id:03d}"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df["topic_id"] = df["topic_id"].astype(int)

    # Topic-level readable ID.
    # This is used only to build stable sample IDs and for the full output file.
    df["numtopic_id"] = df["topic_id"].apply(make_numtopic_id)

    # Sort rows before creating sample IDs.
    # This makes sample numbering deterministic if the input order is stable.
    df = df.sort_values(["topic_id"]).reset_index(drop=True)

    # Sample-level readable ID.
    #
    # Examples:
    # - first sample in topic 24 becomes T024_S0001
    # - second sample in topic 24 becomes T024_S0002
    df["sample_id"] = (
        df["numtopic_id"]
        + "_S"
        + df.groupby("topic_id").cumcount().add(1).astype(str).str.zfill(4)
    )

    # Save a full copy of the original dataset with the new IDs added.
    # This file keeps all original columns plus numtopic_id and sample_id.
    df.to_csv(FULL_OUTPUT_PATH, index=False)

    manifest_rows = []

    for topic_id, group in df.groupby("topic_id", sort=True):
        topic_folder = OUTPUT_DIR / f"topic_{topic_id}"
        topic_folder.mkdir(parents=True, exist_ok=True)

        group = group.reset_index(drop=True)

        n_batches = (len(group) + MAX_ROWS_PER_BATCH - 1) // MAX_ROWS_PER_BATCH

        for batch_idx in range(n_batches):
            start = batch_idx * MAX_ROWS_PER_BATCH
            end = start + MAX_ROWS_PER_BATCH

            batch_df = group.iloc[start:end].copy()

            batch_number = batch_idx + 1

            output_file = topic_folder / f"topic_{topic_id}_batch_{batch_number:03d}.csv"

            # Export only LLM-facing columns.
            batch_df[COLUMNS_FOR_LLM].to_csv(output_file, index=False)

            manifest_rows.append(
                {
                    "topic_id": topic_id,
                    "numtopic_id": make_numtopic_id(topic_id),
                    "topic_name": group["topic_name"].iloc[0],
                    "batch_id": batch_number,
                    "n_batches_for_topic": n_batches,
                    "n_samples": len(batch_df),
                    "folder_path": str(topic_folder),
                    "file_path": str(output_file),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["topic_id", "batch_id"]
    )

    manifest_path = OUTPUT_DIR / "_manifest_topic_batches.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Done.")
    print(f"Input file: {INPUT_PATH}")
    print(f"Full output file with IDs: {FULL_OUTPUT_PATH}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Max rows per batch: {MAX_ROWS_PER_BATCH}")
    print(f"Number of topic folders: {manifest['topic_id'].nunique()}")
    print(f"Number of batch files: {len(manifest)}")
    print(f"Total samples exported: {manifest['n_samples'].sum()}")
    print(f"Manifest saved at: {manifest_path}")


if __name__ == "__main__":
    main()