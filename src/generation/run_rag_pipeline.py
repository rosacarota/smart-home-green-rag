from __future__ import annotations

from pathlib import Path
import argparse
import sys


CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from generation.rag_pipeline.config import DEFAULT_QUERY
from generation.rag_pipeline.display import (
    print_eco_metric_result,
    print_generation_result,
    print_postprocessing_summary,
    print_retrieval_results,
    print_saved_paths,
)
from generation.rag_pipeline.pipeline import run_rag_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full RAG pipeline: retrieval, postprocessing, "
            "LLM generation and JSON persistence."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Input trigger-action rule or natural-language query.",
    )

    args = parser.parse_args()

    result = run_rag_pipeline(
        query=args.query,
    )

    print_retrieval_results(
        query=result["query"],
        retrieved=result["retrieved"],
    )

    print_postprocessing_summary(
        postprocessed=result["postprocessed"],
    )

    print_generation_result(
        generation=result["generation"],
    )

    print_eco_metric_result(
        eco_metric=result["eco_metric"],
    )

    print_saved_paths(
        output_paths=result["output_paths"],
    )


if __name__ == "__main__":
    main()