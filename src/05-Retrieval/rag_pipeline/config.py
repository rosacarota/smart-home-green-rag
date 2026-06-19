from __future__ import annotations

from pathlib import Path
import re


# =============================================================================
# PATHS
# =============================================================================

KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
ARTICLE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "article_index"
RULE_INDEX_DIR = KNOWLEDGE_BASE_DIR / "rule_index"

RETRIEVAL_RUNS_DIR = Path("data/retrieval_runs")


# =============================================================================
# MODELS
# =============================================================================

EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_PROVIDER = "ollama_cloud"
LLM_MODEL_NAME = "gpt-oss:20b"
OLLAMA_CLOUD_HOST = "https://ollama.com"
LLM_TEMPERATURE = 0.2


# =============================================================================
# RETRIEVAL / POSTPROCESSING CONFIG
# =============================================================================

TOP_K_RULES = 10
TOP_K_ARTICLES = 15

FINAL_RULES_K = 3
FINAL_ARTICLES_K = 5

MIN_ARTICLE_CHARS = 150
TEXT_PREVIEW_CHARS = 900

RULE_DEDUP_SIMILARITY_THRESHOLD = 0.88


# =============================================================================
# RUN CONFIG
# =============================================================================

RUN_ID_WIDTH = 3
RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")

RETRIEVAL_OUTPUT_FILENAME = "retrieval_results.json"
POSTPROCESSED_OUTPUT_FILENAME = "postprocessed_results.json"
GENERATION_OUTPUT_FILENAME = "generation_results.json"

DEFAULT_QUERY = "IF You exit from home THEN Turn off lights."