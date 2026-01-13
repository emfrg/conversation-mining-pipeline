"""Embedding utilities for deduplication.

Provides:
- Config helpers for deduplication/decomposition settings
- Vertex AI embeddings model initialization
- Batched embedding computation
"""

import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()

BATCH_SIZE = 100


def get_dedup_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get deduplication config, with fallback to main embedding config.

    Args:
        config: Optional config dict. If None, imports global config.

    Returns:
        Dict with embedding_model, dimensions, similarity_threshold, second_pass_threshold.
    """
    if config is None:
        from config import config as global_config

        config = global_config

    dedup_config = config.get("deduplication", {})
    embedding_config = config.get("embedding", {})

    return {
        "embedding_model": dedup_config.get(
            "embedding_model", embedding_config.get("model", "text-embedding-005")
        ),
        "dimensions": dedup_config.get(
            "dimensions", embedding_config.get("dimensions", 768)
        ),
        "similarity_threshold": dedup_config.get("similarity_threshold", 0.95),
        "second_pass_threshold": dedup_config.get("second_pass_threshold", 0.92),
        "second_pass": dedup_config.get("second_pass", True),
    }


def get_decomposition_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get decomposition config.

    Args:
        config: Optional config dict. If None, imports global config.

    Returns:
        Dict with enabled, with_presuppositions.
    """
    if config is None:
        from config import config as global_config

        config = global_config

    decomp_config = config.get("decomposition", {})
    return {
        "enabled": decomp_config.get("enabled", True),
        "with_presuppositions": decomp_config.get("with_presuppositions", True),
    }


def get_embedding_model(config: dict[str, Any] | None = None) -> VertexAIEmbeddings:
    """Initialize Vertex AI embeddings model from deduplication config.

    Args:
        config: Optional config dict. If None, imports global config.

    Returns:
        Configured VertexAIEmbeddings instance.
    """
    dedup_config = get_dedup_config(config)
    embedding_model = dedup_config["embedding_model"]
    embedding_dim = int(dedup_config["dimensions"])

    return VertexAIEmbeddings(
        model_name=embedding_model,
        project=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("VERTEX_AI_LOCATION", "europe-west1"),
        dimensions=embedding_dim,
    )


def compute_embeddings_batched(
    questions: list[str],
    model: VertexAIEmbeddings,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Compute embeddings for a list of questions in batches.

    Args:
        questions: List of question texts.
        model: Vertex AI embeddings model.
        batch_size: Number of questions per batch.

    Returns:
        Array of shape (n_questions, embedding_dim).
    """
    all_embeddings = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        embeddings = model.embed_documents(batch)
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)
