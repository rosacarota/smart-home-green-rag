from __future__ import annotations

from typing import Any

from rag_pipeline.generation import generate_green_rule
from rag_pipeline.postprocessing import postprocess_context
from rag_pipeline.retrieval import retrieve_context
from rag_pipeline.run_io import save_run_results


def run_rag_pipeline(query: str) -> dict[str, Any]:
    """
    Run the full RAG pipeline.

    Pipeline:
        1. Retrieve similar rules and article chunks.
        2. Postprocess retrieved context.
        3. Generate an improved green rule with the LLM.
        4. Save retrieval, postprocessing and generation results.

    Returns:
        A dictionary containing all intermediate and final outputs.
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

    output_paths = save_run_results(
        query=query,
        retrieved=retrieved,
        postprocessed=postprocessed,
        generation=generation,
    )

    return {
        "query": query,
        "retrieved": retrieved,
        "postprocessed": postprocessed,
        "generation": generation,
        "output_paths": output_paths,
    }