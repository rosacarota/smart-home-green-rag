from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_PATH = (
    ROOT / "data" / "synthetic_evaluation" / "synthetic_green_rules_360.xlsx"
)

DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "synthetic_evaluation" / "synthetic_green_rules_360.jsonl"
)

REQUIRED_COLUMNS = [
    "rule_id",
    "target_category",
    "difficulty",
    "original_rule",
]


def normalize_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_excel_dataset(input_path: Path, sheet_name: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Excel file not found: {input_path}")

    dataframe = pd.read_excel(
        input_path,
        sheet_name=sheet_name,
    )

    dataframe.columns = [
        normalize_column_name(column)
        for column in dataframe.columns
    ]

    return dataframe


def keep_required_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    optional_columns = [
        "source_distribution_count",
    ]

    selected_columns = REQUIRED_COLUMNS + [
        column
        for column in optional_columns
        if column in dataframe.columns
    ]

    return dataframe[selected_columns].copy()


def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.dropna(
        subset=REQUIRED_COLUMNS,
    ).copy()

    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            dataframe[column] = dataframe[column].astype(str).str.strip()

    dataframe = dataframe[
        dataframe["original_rule"].str.startswith("IF ")
        & dataframe["original_rule"].str.contains(" THEN ")
    ].copy()

    dataframe = dataframe.drop_duplicates(
        subset=["rule_id"],
        keep="first",
    ).copy()

    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        raise ValueError("Dataset is empty after cleaning.")

    duplicated_rule_ids = dataframe[
        dataframe["rule_id"].duplicated(keep=False)
    ]

    if not duplicated_rule_ids.empty:
        raise ValueError(
            "Duplicated rule_id values found: "
            + ", ".join(duplicated_rule_ids["rule_id"].unique())
        )

    allowed_difficulties = {
        "easy",
        "medium",
        "already_good",
        "adversarial",
    }

    invalid_difficulties = sorted(
        set(dataframe["difficulty"]) - allowed_difficulties
    )

    if invalid_difficulties:
        raise ValueError(
            "Invalid difficulty values found: "
            + ", ".join(invalid_difficulties)
        )


def write_jsonl(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = dataframe.to_dict(
        orient="records",
    )

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(records)} records to {output_path}")


def print_summary(dataframe: pd.DataFrame) -> None:
    print("\nDataset summary")
    print("=" * 80)
    print(f"Total records: {len(dataframe)}")

    print("\nBy category:")
    print(
        dataframe["target_category"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBy difficulty:")
    print(
        dataframe["difficulty"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nBy category and difficulty:")
    print(
        dataframe
        .groupby(["target_category", "difficulty"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert synthetic rules Excel dataset to JSONL."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the synthetic rules Excel file.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the JSONL dataset will be written.",
    )

    parser.add_argument(
        "--sheet-name",
        type=str,
        default="Synthetic_Rules",
        help="Excel sheet containing the rules.",
    )

    args = parser.parse_args()

    dataframe = load_excel_dataset(
        input_path=args.input_path,
        sheet_name=args.sheet_name,
    )

    dataframe = keep_required_columns(dataframe)
    dataframe = clean_dataset(dataframe)

    validate_dataset(dataframe)

    write_jsonl(
        dataframe=dataframe,
        output_path=args.output_path,
    )

    print_summary(dataframe)


if __name__ == "__main__":
    main()