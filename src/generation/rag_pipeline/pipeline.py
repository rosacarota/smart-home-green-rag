from __future__ import annotations

from typing import Any

from generation.rag_pipeline.eco_metric import evaluate_rule_pair
from generation.rag_pipeline.generation import generate_green_rule
from generation.rag_pipeline.postprocessing import postprocess_context
from generation.rag_pipeline.retrieval import retrieve_context
from generation.rag_pipeline.run_io import save_run_results


def extract_generated_rule(generation: dict[str, Any]) -> str:
    """
    Extract the generated rule from the LLM parsed response.

    Expected field:
        improved_rule
    """
    parsed_response = generation.get("parsed_response", {})

    generated_rule = (
        parsed_response.get("improved_rule")
        or parsed_response.get("generated_rule")
    )

    if not generated_rule:
        raise ValueError(
            "Could not extract generated rule from LLM response. "
            "Expected field 'improved_rule'."
        )

    return str(generated_rule).strip()


def run_rag_pipeline(query: str) -> dict[str, Any]:
    """
    Run the full RAG pipeline.

    Pipeline:
        1. Retrieve similar rules and article chunks.
        2. Postprocess retrieved context.
        3. Generate an improved green rule with the LLM.
        4. Evaluate original and generated rule with the eco-metric.
        5. Save retrieval, postprocessing, generation and eco-metric results.
    """
    retrieved = retrieve_context(
        query=query,
    )

    postprocessed = postprocess_context(
        query=query,
        retrieved=retrieved,
    )

    generation = generate_green_rule(
        query=query,
        postprocessed=postprocessed,
    )

    generated_rule = extract_generated_rule(
        generation=generation,
    )

    eco_metric = evaluate_rule_pair(
        original_rule=query,
        generated_rule=generated_rule,
        rule_id="PIPELINE_RULE",
    )

    output_paths = save_run_results(
        query=query,
        retrieved=retrieved,
        postprocessed=postprocessed,
        generation=generation,
        eco_metric=eco_metric,
    )

    return {
        "query": query,
        "retrieved": retrieved,
        "postprocessed": postprocessed,
        "generation": generation,
        "eco_metric": eco_metric,
        "output_paths": output_paths,
    }