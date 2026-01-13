"""Core deduplication functions - reusable by pipeline and standalone scripts.

This module provides the core logic for:
- Question decomposition (breaking compound questions into atomic units)
- Vector-based deduplication (greedy clustering by similarity)
- FAQ synthesis (using LLM to create clean FAQ questions)
- Embedding utilities (Vertex AI embeddings)

Usage:
    from src.analysis.dedup import (
        decompose_all_questions,
        deduplicate_greedy_decomposed,
        synthesize_all_questions,
        get_embedding_model,
    )
"""

from .core import (
    # Embeddings
    BATCH_SIZE,
    compute_embeddings_batched,
    # Decomposition
    count_atomic_questions,
    decompose_all_faqs,
    decompose_all_questions,
    decompose_single_faq,
    decompose_single_question,
    # Deduplication
    deduplicate_greedy,
    deduplicate_greedy_decomposed,
    deduplicate_greedy_pass2,
    get_decomposition_config,
    get_dedup_config,
    get_embedding_model,
    process_cluster,
    process_cluster_decomposed,
    process_cluster_pass2,
    # Synthesis
    synthesize_all_questions,
    synthesize_single_question,
)

__all__ = [
    # Embeddings
    "BATCH_SIZE",
    "get_dedup_config",
    "get_decomposition_config",
    "get_embedding_model",
    "compute_embeddings_batched",
    # Decomposition
    "decompose_single_question",
    "decompose_all_questions",
    "count_atomic_questions",
    "decompose_single_faq",
    "decompose_all_faqs",
    # Deduplication
    "deduplicate_greedy",
    "deduplicate_greedy_decomposed",
    "deduplicate_greedy_pass2",
    "process_cluster",
    "process_cluster_decomposed",
    "process_cluster_pass2",
    # Synthesis
    "synthesize_single_question",
    "synthesize_all_questions",
]
