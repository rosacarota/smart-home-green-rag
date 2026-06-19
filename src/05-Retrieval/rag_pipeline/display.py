from __future__ import annotations

from typing import Any
import json

from llama_index.core.schema import NodeWithScore

from rag_pipeline.node_utils import result_to_dict


def print_rule_results(results: list[NodeWithScore]) -> None:
    print("\n" + "=" * 90)
    print("RETRIEVED SIMILAR RULES")
    print("=" * 90)

    if not results:
        print("No rule results found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result_to_dict(result, rank=rank)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {item['score']}")
        print(f"Rule ID: {item['rule_id']}")
        print(f"Topic: {item['llm_topic_name']}")
        print(f"Green category: {item['green_category']}")
        print(f"Node ID: {item['node_id']}")
        print("\nText preview:")
        print(item["text_preview"])


def print_article_results(results: list[NodeWithScore]) -> None:
    print("\n" + "=" * 90)
    print("RETRIEVED ARTICLE CHUNKS")
    print("=" * 90)

    if not results:
        print("No article results found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result_to_dict(result, rank=rank)

        print("\n" + "-" * 90)
        print(f"Rank: {rank}")
        print(f"Score: {item['score']}")
        print(f"Article: {item['source_title']}")
        print(f"Article ID: {item['article_id']}")
        print(f"Section: {item['section_title']}")
        print(f"Section path: {item['section_path']}")
        print(f"Green categories: {item['green_categories']}")
        print(f"Node ID: {item['node_id']}")
        print("\nText preview:")
        print(item["text_preview"])


def print_retrieval_results(
    query: str,
    retrieved: dict[str, list[NodeWithScore]],
) -> None:
    print("\n" + "=" * 90)
    print("QUERY")
    print("=" * 90)
    print(query)

    print_rule_results(retrieved["rules"])
    print_article_results(retrieved["articles"])


def print_postprocessing_summary(postprocessed: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("POSTPROCESSING SUMMARY")
    print("=" * 90)

    selected_rules = postprocessed["rules"]["selected"]
    discarded_rules = postprocessed["rules"]["discarded"]
    selected_articles = postprocessed["articles"]["selected"]
    discarded_articles = postprocessed["articles"]["discarded"]

    print(f"Selected rules: {len(selected_rules)}")
    print(f"Discarded rules: {len(discarded_rules)}")
    print(f"Selected articles: {len(selected_articles)}")
    print(f"Discarded articles: {len(discarded_articles)}")

    print("\nFINAL RULES")
    for item in selected_rules:
        print(
            f"[{item['final_rank']}] "
            f"original_rank={item['original_rank']} "
            f"original_score={item['original_score']}"
        )
        print(item["text_preview"])

    print("\nFINAL ARTICLES")
    for item in selected_articles:
        print(
            f"[{item['final_rank']}] "
            f"original_rank={item['original_rank']} "
            f"original_score={item['original_score']} "
            f"rerank_score={item['rerank_score']}"
        )
        print(f"{item['source_title']} / {item['section_title']}")
        print(item["text_preview"][:300])


def print_generation_result(generation: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("GENERATED GREEN RULE")
    print("=" * 90)

    parsed_response = generation.get("parsed_response", {})

    print(json.dumps(
        parsed_response,
        indent=2,
        ensure_ascii=False,
    ))


def print_saved_paths(output_paths: dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("RUN SAVED")
    print("=" * 90)
    print(f"Run directory: {output_paths['run_dir']}")
    print(f"Retrieval JSON: {output_paths['retrieval_output_path']}")
    print(f"Postprocessed JSON: {output_paths['postprocessed_output_path']}")
    print(f"Generation JSON: {output_paths['generation_output_path']}")