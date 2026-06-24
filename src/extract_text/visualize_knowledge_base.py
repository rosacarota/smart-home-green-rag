from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import random

import numpy as np
import pandas as pd
import plotly.express as px


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"
OUTPUT_DIR = KNOWLEDGE_BASE_DIR / "visualizations"

GREEN_CATEGORIES = [
    "lighting_efficiency",
    "occupancy_based_control",
    "hvac_optimization",
    "appliance_energy_management",
    "environmental_awareness",
    "comfort_security",
    "water_saving",
    "scheduling_optimization",
    "standby_reduction",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_numeric_vector(value: Any, min_length: int = 10) -> bool:
    if not isinstance(value, list) or len(value) < min_length:
        return False

    sample = value[: min(20, len(value))]
    return all(isinstance(item, (int, float)) for item in sample)


def looks_like_embedding_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False

    checked = 0
    valid = 0

    for embedding in value.values():
        checked += 1
        if is_numeric_vector(embedding):
            valid += 1
        if checked >= 5:
            break

    return checked > 0 and valid == checked


def find_first_key_recursively(obj: Any, key: str) -> Any | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]

        for value in obj.values():
            result = find_first_key_recursively(value, key)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_first_key_recursively(item, key)
            if result is not None:
                return result

    return None


def find_embedding_dict(obj: Any) -> dict[str, list[float]]:
    direct = find_first_key_recursively(obj, "embedding_dict")

    if looks_like_embedding_dict(direct):
        return {str(key): value for key, value in direct.items()}

    # Fallback: scan recursively for any dictionary shaped like
    # {node_id: [float, float, ...]}.
    stack = [obj]

    while stack:
        current = stack.pop()

        if looks_like_embedding_dict(current):
            return {str(key): value for key, value in current.items()}

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)

    raise ValueError("Could not find an embedding dictionary in the vector store JSON.")


def find_metadata_dict(obj: Any) -> dict[str, dict[str, Any]]:
    direct = find_first_key_recursively(obj, "metadata_dict")

    if isinstance(direct, dict):
        return {
            str(key): value
            for key, value in direct.items()
            if isinstance(value, dict)
        }

    return {}


def find_vector_store_file(index_dir: Path) -> Path:
    candidates = sorted(index_dir.glob("*vector_store*.json"))

    if candidates:
        return candidates[0]

    for path in sorted(index_dir.glob("*.json")):
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue

        if find_first_key_recursively(payload, "embedding_dict") is not None:
            return path

    raise FileNotFoundError(
        f"No vector store JSON file found in: {index_dir}. "
        "Run the KB indexing script first."
    )


def extract_text_from_node_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    # Common LlamaIndex serialization shapes.
    if isinstance(payload.get("text"), str):
        return payload["text"]

    text_resource = payload.get("text_resource")
    if isinstance(text_resource, dict) and isinstance(text_resource.get("text"), str):
        return text_resource["text"]

    data = payload.get("__data__")
    if isinstance(data, dict):
        text = extract_text_from_node_payload(data)
        if text:
            return text

    # Last resort: recursively find the first text-looking field.
    for value in payload.values():
        if isinstance(value, dict):
            text = extract_text_from_node_payload(value)
            if text:
                return text

    return ""


def extract_metadata_from_node_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata

    data = payload.get("__data__")
    if isinstance(data, dict):
        metadata = extract_metadata_from_node_payload(data)
        if metadata:
            return metadata

    for value in payload.values():
        if isinstance(value, dict):
            metadata = extract_metadata_from_node_payload(value)
            if metadata:
                return metadata

    return {}


def load_docstore_map(index_dir: Path) -> dict[str, dict[str, Any]]:
    docstore_path = index_dir / "docstore.json"

    if not docstore_path.exists():
        return {}

    payload = read_json(docstore_path)
    data = payload.get("docstore/data", payload)

    if not isinstance(data, dict):
        return {}

    node_map: dict[str, dict[str, Any]] = {}

    for node_id, node_payload in data.items():
        text = extract_text_from_node_payload(node_payload)
        metadata = extract_metadata_from_node_payload(node_payload)

        node_map[str(node_id)] = {
            "text": text,
            "metadata": metadata,
        }

    return node_map


def parse_metadata_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()

    if not stripped:
        return value

    if stripped[0] in "[{" and stripped[-1] in "]}":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value

    return value


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: parse_metadata_value(value) for key, value in metadata.items()}


def active_categories(metadata: dict[str, Any]) -> list[str]:
    categories: list[str] = []

    for category in GREEN_CATEGORIES:
        value = metadata.get(category, 0)
        try:
            is_active = int(value) == 1
        except (TypeError, ValueError):
            is_active = str(value).strip().casefold() in {"true", "yes", "present"}

        if is_active:
            categories.append(category)

    return categories


def infer_green_category(metadata: dict[str, Any]) -> str:
    for key in ("green_category", "dominant_green_category"):
        value = str(metadata.get(key, "")).strip()
        if value:
            return value

    categories = active_categories(metadata)
    return categories[0] if categories else "uncategorized"


def text_preview(text: str, max_chars: int = 240) -> str:
    text = " ".join(str(text or "").split())

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."


def load_index_records(index_dir: Path, source_label: str) -> list[dict[str, Any]]:
    vector_store_path = find_vector_store_file(index_dir)
    vector_store_payload = read_json(vector_store_path)

    embedding_dict = find_embedding_dict(vector_store_payload)
    metadata_dict = find_metadata_dict(vector_store_payload)
    docstore_map = load_docstore_map(index_dir)

    records: list[dict[str, Any]] = []

    for node_id, embedding in embedding_dict.items():
        docstore_record = docstore_map.get(node_id, {})
        metadata = {}

        if isinstance(docstore_record.get("metadata"), dict):
            metadata.update(docstore_record["metadata"])

        if isinstance(metadata_dict.get(node_id), dict):
            metadata.update(metadata_dict[node_id])

        metadata = normalize_metadata(metadata)

        content_type = str(metadata.get("content_type", source_label)).strip() or source_label
        green_category = infer_green_category(metadata)
        categories = active_categories(metadata)

        text = str(docstore_record.get("text") or "").strip()

        if not text:
            text = str(metadata.get("if_then_rule") or "").strip()

        if not text:
            text = str(metadata.get("article_summary") or "").strip()

        label = (
            metadata.get("sample_id")
            or metadata.get("article_id")
            or metadata.get("source_title")
            or node_id
        )

        records.append(
            {
                "node_id": node_id,
                "source": source_label,
                "content_type": content_type,
                "green_category": green_category,
                "active_green_categories": ", ".join(categories),
                "label": str(label),
                "article_id": str(metadata.get("article_id", "")),
                "source_title": str(metadata.get("source_title", "")),
                "section_title": str(metadata.get("section_title", "")),
                "sample_id": str(metadata.get("sample_id", "")),
                "llm_topic_name": str(metadata.get("llm_topic_name", "")),
                "word_count": metadata.get("word_count", ""),
                "text_preview": text_preview(text),
                "embedding": embedding,
            }
        )

    return records


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    if embeddings.shape[0] < 3:
        raise ValueError("At least 3 vectors are needed for visualization.")

    if method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=3, random_state=random_state)
        return reducer.fit_transform(embeddings)

    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP is not installed. Run: pip install umap-learn"
        ) from error

    safe_neighbors = min(max(2, n_neighbors), embeddings.shape[0] - 1)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )

    return reducer.fit_transform(embeddings)


def build_dataframe(records: list[dict[str, Any]], coordinates: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record, coordinate in zip(records, coordinates):
        row = {key: value for key, value in record.items() if key != "embedding"}
        row["x"] = float(coordinate[0])
        row["y"] = float(coordinate[1])
        row["z"] = float(coordinate[2])
        rows.append(row)

    return pd.DataFrame(rows)


def create_plot(df: pd.DataFrame, color_by: str):
    if color_by not in df.columns:
        color_by = "green_category"

    hover_columns = [
        "source",
        "content_type",
        "green_category",
        "active_green_categories",
        "article_id",
        "source_title",
        "section_title",
        "sample_id",
        "llm_topic_name",
        "node_id",
        "text_preview",
    ]

    available_hover_columns = [column for column in hover_columns if column in df.columns]

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=color_by,
        symbol="source",
        hover_name="label",
        hover_data=available_hover_columns,
        title="Knowledge Base Embedding Visualization",
    )

    fig.update_layout(
        legend_title_text=color_by,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig


def write_summary(df: pd.DataFrame, output_path: Path, args: argparse.Namespace) -> None:
    summary = {
        "article_index_dir": str(args.article_index_dir),
        "rule_index_dir": str(args.rule_index_dir),
        "method": args.method,
        "color_by": args.color_by,
        "total_points": int(len(df)),
        "source_counts": df["source"].value_counts().to_dict(),
        "green_category_counts": df["green_category"].value_counts().to_dict(),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the persisted knowledge-base embeddings from "
            "article_index and rule_index with UMAP or PCA."
        )
    )

    parser.add_argument(
        "--article-index-dir",
        type=Path,
        default=ARTICLE_INDEX_DIR,
        help="Path to the persisted article vector index.",
    )
    parser.add_argument(
        "--rule-index-dir",
        type=Path,
        default=RULE_INDEX_DIR,
        help="Path to the persisted rule vector index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where HTML/CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "pca"],
        default="umap",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="green_category",
        help="Column used to color points, e.g. green_category, source, content_type.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional maximum number of points to visualize. 0 means all points.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by UMAP/PCA and optional sampling.",
    )
    parser.add_argument(
        "--offline-html",
        action="store_true",
        help="Embed plotly.js in the HTML file so it works offline. File will be larger.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading article index vectors...")
    article_records = load_index_records(
        index_dir=args.article_index_dir,
        source_label="article_index",
    )
    print(f"Article vectors loaded: {len(article_records)}")

    print("Loading rule index vectors...")
    rule_records = load_index_records(
        index_dir=args.rule_index_dir,
        source_label="rule_index",
    )
    print(f"Rule vectors loaded: {len(rule_records)}")

    records = [*article_records, *rule_records]

    if args.sample_size and args.sample_size > 0 and len(records) > args.sample_size:
        random.seed(args.random_state)
        records = random.sample(records, args.sample_size)
        print(f"Sampled points: {len(records)}")

    if not records:
        raise ValueError("No vectors found to visualize.")

    embeddings = np.array([record["embedding"] for record in records], dtype=np.float32)
    print(f"Embedding matrix shape: {embeddings.shape}")

    print(f"Reducing embeddings with {args.method.upper()}...")
    coordinates = reduce_embeddings(
        embeddings=embeddings,
        method=args.method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
    )

    df = build_dataframe(records=records, coordinates=coordinates)

    csv_path = args.output_dir / "kb_embedding_points.csv"
    html_path = args.output_dir / "kb_embedding_visualization_3d.html"
    summary_path = args.output_dir / "kb_embedding_visualization_report.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    fig = create_plot(df=df, color_by=args.color_by)
    fig.write_html(
        html_path,
        include_plotlyjs=True if args.offline_html else "cdn",
        full_html=True,
    )

    write_summary(df=df, output_path=summary_path, args=args)

    print(f"CSV points saved to: {csv_path}")
    print(f"Interactive visualization saved to: {html_path}")
    print(f"Visualization report saved to: {summary_path}")


if __name__ == "__main__":
    main()
