from __future__ import annotations

import os

from dotenv import load_dotenv
from ollama import Client

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from generation.rag_pipeline.config import (
    EMBED_MODEL_NAME,
    OLLAMA_CLOUD_HOST,
)


def configure_llamaindex() -> None:
    """
    Configure the same embedding model used to build the persisted indexes.

    During retrieval, the input query must be embedded in the same vector space
    as the nodes stored in the vector indexes.
    """
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
    )

    Settings.embed_model = embed_model


def configure_llm_client() -> Client:
    """
    Configure the Ollama Cloud client.

    The API key is loaded from the .env file through python-dotenv.

    Required .env variable:
        OLLAMA_API_KEY=your_api_key
    """
    load_dotenv()

    api_key = os.environ.get("OLLAMA_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "Missing OLLAMA_API_KEY. "
            "Add it to your .env file, for example:\n"
            "OLLAMA_API_KEY=your_api_key"
        )

    return Client(
        host=OLLAMA_CLOUD_HOST,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )