"""Core deduplication functions - reusable by pipeline and standalone scripts."""

from .decompose import (
    count_atomic_questions,
    decompose_all_faqs,
    decompose_all_questions,
    decompose_single_faq,
    decompose_single_question,
)
from .deduplicate import (
    deduplicate_greedy,
    deduplicate_greedy_decomposed,
    deduplicate_greedy_pass2,
    process_cluster,
    process_cluster_decomposed,
    process_cluster_pass2,
)
from .embeddings import (
    BATCH_SIZE,
    compute_embeddings_batched,
    get_decomposition_config,
    get_dedup_config,
    get_embedding_model,
)
from .synthesize import (
    synthesize_all_questions,
    synthesize_single_question,
)

__all__ = [
    # Embeddings
    "BATCH_SIZE",
    "compute_embeddings_batched",
    "get_dedup_config",
    "get_decomposition_config",
    "get_embedding_model",
    # Decompose
    "count_atomic_questions",
    "decompose_all_faqs",
    "decompose_all_questions",
    "decompose_single_faq",
    "decompose_single_question",
    # Deduplicate
    "deduplicate_greedy",
    "deduplicate_greedy_decomposed",
    "deduplicate_greedy_pass2",
    "process_cluster",
    "process_cluster_decomposed",
    "process_cluster_pass2",
    # Synthesize
    "synthesize_all_questions",
    "synthesize_single_question",
]
