from pathlib import Path
from typing import Any
import json
import re

from llama_index.core.schema import TextNode


RULES_INPUT_PATH = Path(
    "data/dataset-rules/final_dataset/green_rules_final_dataset_normalized.json"
)
RULE_NODES_OUTPUT_PATH = Path(
    "data/dataset-rules/final_dataset/rule_nodes.jsonl"
)
RULE_NODES_PREVIEW_PATH = Path(
    "data/dataset-rules/final_dataset/rule_nodes_preview.json"
)
RULE_NODES_REPORT_PATH = Path(
    "data/dataset-rules/final_dataset/rule_nodes_report.json"
)

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

EXCLUDED_EMBED_METADATA_KEYS = [
    "sample_id",
    "rule_id",
    "if_then_rule",
    "original_green_category",
]

EXCLUDED_LLM_METADATA_KEYS = [
    "sample_id",
    "rule_id",
]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    normalized = str(text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def make_metadata_vector_store_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (dict, list, tuple, set)):
            safe_metadata[key] = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            safe_metadata[key] = value

    return safe_metadata


def load_rule_rows(input_path: Path = RULES_INPUT_PATH) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Rules dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
            "Rules dataset must be either a list of objects or an object with a 'rows' list."
        )

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_rows.append(row)

    if not normalized_rows:
        raise ValueError(f"No valid rule rows found in: {input_path}")

    return normalized_rows


def build_rule_text(row: dict[str, Any]) -> str:
    green_category = normalize_text(row.get("green_category"))
    topic = normalize_text(row.get("llm_topic_name"))
    rule = normalize_text(row.get("if_then_rule"))

    parts = []

    if green_category:
        parts.append(f"Green category: {green_category}")

    if topic:
        parts.append(f"Topic: {topic}")

    parts.append(f"Rule: {rule}")

    return "\n".join(parts).strip()


def build_rule_metadata(row: dict[str, Any], rule_index: int) -> dict[str, Any]:
    sample_id = normalize_text(row.get("sample_id"))
    green_category = normalize_text(row.get("green_category"))
    original_green_category = normalize_text(row.get("original_green_category"))
    topic = normalize_text(row.get("llm_topic_name"))
    rule = normalize_text(row.get("if_then_rule"))

    metadata: dict[str, Any] = {
        "content_type": "green_rule",
        "sample_id": sample_id,
        "rule_id": sample_id or f"rule_{rule_index:06d}",
        "llm_topic_name": topic,
        "green_category": green_category,
        "dominant_green_category": green_category,
        "original_green_category": original_green_category,
        "if_then_rule": rule,
    }

    for category in GREEN_CATEGORIES:
        metadata[category] = 1 if green_category == category else 0

    return make_metadata_vector_store_safe(metadata)


def build_rule_nodes(rows: list[dict[str, Any]] | None = None) -> list[TextNode]:
    if rows is None:
        rows = load_rule_rows()

    nodes: list[TextNode] = []
    used_node_ids: set[str] = set()
    skipped_rows = 0

    for index, row in enumerate(rows, start=1):
        rule_text = normalize_text(row.get("if_then_rule"))

        if not rule_text:
            skipped_rows += 1
            continue

        metadata = build_rule_metadata(row=row, rule_index=index)
        sample_id = metadata.get("sample_id") or f"rule_{index:06d}"
        base_node_id = f"rule__{sample_id}"
        node_id = base_node_id
        duplicate_counter = 2

        while node_id in used_node_ids:
            node_id = f"{base_node_id}__dup_{duplicate_counter}"
            duplicate_counter += 1

        used_node_ids.add(node_id)

        excluded_embed_keys = [
            key for key in EXCLUDED_EMBED_METADATA_KEYS if key in metadata
        ]
        excluded_llm_keys = [
            key for key in EXCLUDED_LLM_METADATA_KEYS if key in metadata
        ]

        node = TextNode(
            id_=node_id,
            text=build_rule_text(row),
            metadata=metadata,
            excluded_embed_metadata_keys=excluded_embed_keys,
            excluded_llm_metadata_keys=excluded_llm_keys,
        )

        nodes.append(node)

    print(f"Rule rows loaded: {len(rows)}")
    print(f"Rule nodes created: {len(nodes)}")
    print(f"Skipped rows without rule text: {skipped_rows}")

    return nodes


def node_to_record(node: TextNode, node_index: int) -> dict[str, Any]:
    return {
        "node_index": node_index,
        "node_id": node.node_id,
        "text": node.text,
        "metadata": node.metadata,
        "excluded_embed_metadata_keys": node.excluded_embed_metadata_keys,
        "excluded_llm_metadata_keys": node.excluded_llm_metadata_keys,
        "text_characters": len(node.text),
        "word_count": len(node.text.split()),
    }


def save_rule_nodes(nodes: list[TextNode]) -> None:
    RULE_NODES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RULE_NODES_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for index, node in enumerate(nodes, start=1):
            record = node_to_record(node, node_index=index)
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")

    preview = [
        node_to_record(node, node_index=index)
        for index, node in enumerate(nodes[:50], start=1)
    ]

    with RULE_NODES_PREVIEW_PATH.open("w", encoding="utf-8") as file:
        json.dump(preview, file, indent=2, ensure_ascii=False)

    category_counts = {
        category: 0
        for category in GREEN_CATEGORIES
    }

    for node in nodes:
        green_category = node.metadata.get("green_category")
        if green_category in category_counts:
            category_counts[green_category] += 1

    report = {
        "rules_input_path": str(RULES_INPUT_PATH),
        "rule_nodes_output_path": str(RULE_NODES_OUTPUT_PATH),
        "rule_nodes_preview_path": str(RULE_NODES_PREVIEW_PATH),
        "nodes_created": len(nodes),
        "category_counts": category_counts,
    }

    with RULE_NODES_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Rule nodes saved to: {RULE_NODES_OUTPUT_PATH}")
    print(f"Rule nodes preview saved to: {RULE_NODES_PREVIEW_PATH}")
    print(f"Rule nodes report saved to: {RULE_NODES_REPORT_PATH}")


def main() -> None:
    nodes = build_rule_nodes()
    save_rule_nodes(nodes)

    if nodes:
        first_node = nodes[0]
        print("\nFirst rule node ID:")
        print(first_node.node_id)

        print("\nFirst rule node metadata:")
        print(json.dumps(first_node.metadata, indent=2, ensure_ascii=False))

        print("\nFirst rule node text:")
        print(first_node.text)


if __name__ == "__main__":
    main()
