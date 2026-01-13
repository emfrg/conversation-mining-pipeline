"""Deduplication functions using vector similarity.

Uses greedy clustering algorithm: iterates through questions, groups similar ones
based on cosine similarity threshold, and selects a representative per group.

Provides:
- deduplicate_greedy: Basic greedy deduplication
- deduplicate_greedy_decomposed: Deduplication preserving question type metadata
- process_cluster_decomposed: Full cluster processing for Pass 1
- process_cluster_pass2: Full cluster processing for Pass 2 (synthesized FAQs)

Uses VertexAIEmbeddings for vector representation and sklearn cosine_similarity.
"""

from typing import Any

import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import compute_embeddings_batched


def deduplicate_greedy(
    questions: list[str],
    ids: list[str],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Greedy de-duplication within a single cluster (legacy mode).

    Algorithm:
    1. For each question (in order), if not yet assigned to a group:
       - Create new group with this question as representative
       - Add all questions with similarity >= threshold to this group
    2. Return groups with representative question and members.

    Args:
        questions: List of canonical FAQ questions.
        ids: List of conversation IDs corresponding to questions.
        similarity_matrix: Pairwise cosine similarity matrix.
        threshold: Minimum similarity to consider as duplicate.

    Returns:
        List of groups, each with representative question, count, and members.
    """
    n = len(questions)
    assigned = [False] * n
    groups: list[dict[str, Any]] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new group with item i as representative
        group_indices = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if not assigned[j] and similarity_matrix[i, j] >= threshold:
                group_indices.append(j)
                assigned[j] = True

        # Build group result
        representative = questions[i]
        variants = [questions[j] for j in group_indices if j != i]
        original_ids = [ids[j] for j in group_indices]

        groups.append(
            {
                "representative_question": representative,
                "count": len(group_indices),
                "original_ids": original_ids,
                "variants": variants if variants else [],
            }
        )

    # Sort by count descending
    groups.sort(key=lambda x: x["count"], reverse=True)
    return groups


def deduplicate_greedy_decomposed(
    questions: list[str],
    original_ids: list[str],
    question_types: list[str],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """Greedy de-duplication for decomposed atomic questions.

    Similar to deduplicate_greedy but handles the decomposed data structure:
    - Each question may come from a different original conversation
    - Groups track unique original_ids (a question might appear in multiple convos)
    - Tracks question type (explicit vs presupposition)

    Args:
        questions: List of atomic questions.
        original_ids: List of original conversation IDs.
        question_types: List of question types ("explicit" or "presupposition").
        similarity_matrix: Pairwise cosine similarity matrix.
        threshold: Minimum similarity to consider as duplicate.

    Returns:
        List of groups, each with representative question, count, and metadata.
    """
    n = len(questions)
    assigned = [False] * n
    groups: list[dict[str, Any]] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new group with item i as representative
        group_indices = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if not assigned[j] and similarity_matrix[i, j] >= threshold:
                group_indices.append(j)
                assigned[j] = True

        # Build group result - collect unique original IDs
        representative = questions[i]
        variants = [questions[j] for j in group_indices if j != i]

        # Get unique original IDs (a question might be shared by multiple convos)
        unique_orig_ids = list(set(original_ids[j] for j in group_indices))

        # Determine if this group is primarily presuppositions
        presup_count = sum(
            1 for j in group_indices if question_types[j] == "presupposition"
        )
        is_presupposition = presup_count > len(group_indices) // 2

        groups.append(
            {
                "representative_question": representative,
                "count": len(group_indices),
                "unique_conversation_count": len(unique_orig_ids),
                "original_ids": unique_orig_ids,
                "variants": variants if variants else [],
                "is_presupposition": is_presupposition,
            }
        )

    # Sort by unique_conversation_count descending (more meaningful than raw count)
    groups.sort(key=lambda x: x["unique_conversation_count"], reverse=True)
    return groups


def deduplicate_greedy_pass2(
    faqs: list[dict],
    similarity_matrix: np.ndarray,
    threshold: float,
    is_presupposition: bool = False,
) -> list[dict]:
    """Greedy deduplication for Pass 2 (union original_ids when merging).

    Args:
        faqs: List of FAQ dicts with synthesized_question and original_ids.
        similarity_matrix: Pairwise cosine similarity matrix.
        threshold: Minimum similarity to consider as duplicate.
        is_presupposition: Whether these FAQs are presuppositions.

    Returns:
        List of deduplicated FAQ groups.
    """
    n = len(faqs)
    assigned = [False] * n
    groups = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new group with item i as representative
        group_indices = [i]
        assigned[i] = True

        for j in range(i + 1, n):
            if not assigned[j] and similarity_matrix[i, j] >= threshold:
                group_indices.append(j)
                assigned[j] = True

        # Build merged group - union all original_ids
        representative_faq = faqs[i]
        all_original_ids: set[str] = set()
        variants = []

        for idx in group_indices:
            all_original_ids.update(faqs[idx].get("original_ids", []))
            if idx != i:
                variants.append(faqs[idx].get("synthesized_question", ""))

        groups.append(
            {
                "synthesized_question": representative_faq.get(
                    "synthesized_question", ""
                ),
                "original_ids": list(all_original_ids),
                "unique_conversation_count": len(all_original_ids),
                "variants_merged": variants if variants else [],
                "merge_count": len(group_indices),
                "is_presupposition": is_presupposition,
            }
        )

    # Sort by unique_conversation_count descending
    groups.sort(key=lambda x: x["unique_conversation_count"], reverse=True)
    return groups


# =============================================================================
# Cluster Processing Functions
# =============================================================================


def process_cluster(
    cluster_id: str,
    cluster_data: dict,
    embeddings_model: VertexAIEmbeddings,
    threshold: float,
) -> dict:
    """Process a single cluster for de-duplication (legacy mode).

    Args:
        cluster_id: Cluster identifier.
        cluster_data: Cluster data with items.
        embeddings_model: Vertex AI embeddings model.
        threshold: Similarity threshold for de-duplication.

    Returns:
        Dict with cluster results.
    """
    items = cluster_data.get("items", [])
    if not items:
        return {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": 0,
            "unique_count": 0,
            "reduction_pct": 0.0,
            "questions": [],
        }

    # Extract questions and IDs
    questions = [item["canonical_faq_question"] for item in items]
    ids = [item["id"] for item in items]

    # Compute embeddings for this cluster's questions
    print(f"      Computing embeddings for {len(questions)} questions...")
    embeddings = compute_embeddings_batched(questions, embeddings_model)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Run greedy de-duplication
    groups = deduplicate_greedy(questions, ids, similarity_matrix, threshold)

    original_count = len(questions)
    unique_count = len(groups)
    reduction_pct = (
        round((1 - unique_count / original_count) * 100, 1) if original_count > 0 else 0
    )

    return {
        "cluster_title": cluster_data.get("cluster_title", ""),
        "original_count": original_count,
        "unique_count": unique_count,
        "reduction_pct": reduction_pct,
        "questions": groups,
    }


def process_cluster_decomposed(
    cluster_id: str,
    cluster_data: dict,
    embeddings_model: VertexAIEmbeddings,
    threshold: float,
    question_type_filter: str | None = None,
) -> dict:
    """Process a cluster from decomposed data for de-duplication.

    Uses atomic questions (including presuppositions) as the deduplication unit.
    Each atomic question links back to its original conversation ID.

    Args:
        cluster_id: Cluster identifier.
        cluster_data: Cluster data with decomposed items.
        embeddings_model: Vertex AI embeddings model.
        threshold: Similarity threshold for de-duplication.
        question_type_filter: Filter to only process questions of this type
            ("explicit" or "presupposition"). If None, processes all.

    Returns:
        Dict with cluster results.
    """
    items = cluster_data.get("items", [])
    if not items:
        return {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": 0,
            "unique_count": 0,
            "reduction_pct": 0.0,
            "questions": [],
            "input_mode": "decomposed",
            "question_type_filter": question_type_filter,
        }

    # Flatten atomic questions: each becomes a separate item for deduplication
    questions: list[str] = []
    original_ids: list[str] = []
    question_types: list[str] = []

    for item in items:
        orig_id = item.get("original_id", "")
        for atomic in item.get("atomic_questions", []):
            text = atomic.get("text", "")
            q_type = atomic.get("type", "explicit")
            # Filter by question type if specified
            if question_type_filter and q_type != question_type_filter:
                continue
            if text:
                questions.append(text)
                original_ids.append(orig_id)
                question_types.append(q_type)

    if not questions:
        return {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": 0,
            "unique_count": 0,
            "reduction_pct": 0.0,
            "questions": [],
            "input_mode": "decomposed",
            "question_type_filter": question_type_filter,
        }

    # Compute embeddings for all atomic questions
    print(f"      Computing embeddings for {len(questions)} atomic questions...")
    embeddings = compute_embeddings_batched(questions, embeddings_model)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Run greedy de-duplication
    groups = deduplicate_greedy_decomposed(
        questions, original_ids, question_types, similarity_matrix, threshold
    )

    original_count = len(questions)
    unique_count = len(groups)
    reduction_pct = (
        round((1 - unique_count / original_count) * 100, 1) if original_count > 0 else 0
    )

    return {
        "cluster_title": cluster_data.get("cluster_title", ""),
        "original_count": original_count,
        "unique_count": unique_count,
        "reduction_pct": reduction_pct,
        "questions": groups,
        "input_mode": "decomposed",
        "question_type_filter": question_type_filter,
    }


def process_cluster_pass2(
    cluster_id: str,
    cluster_data: dict,
    embeddings_model: VertexAIEmbeddings,
    threshold: float,
    is_presupposition: bool = False,
) -> dict:
    """Process a single cluster for Pass 2 deduplication.

    Args:
        cluster_id: Cluster identifier.
        cluster_data: Cluster data with items from decomposed_synthesized.json.
        embeddings_model: Vertex AI embeddings model.
        threshold: Similarity threshold for deduplication.
        is_presupposition: Whether these FAQs are presuppositions.

    Returns:
        Dict with cluster results.
    """
    items = cluster_data.get("items", [])
    if not items:
        return {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": 0,
            "unique_count": 0,
            "reduction_pct": 0.0,
            "questions": [],
            "is_presupposition": is_presupposition,
        }

    # Extract FAQ texts
    faq_texts = [item.get("synthesized_question", "") for item in items]

    # Compute embeddings
    print(f"      Computing embeddings for {len(faq_texts)} atomic FAQs...")
    embeddings = compute_embeddings_batched(faq_texts, embeddings_model)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Run greedy deduplication
    groups = deduplicate_greedy_pass2(
        items, similarity_matrix, threshold, is_presupposition
    )

    original_count = len(items)
    unique_count = len(groups)
    reduction_pct = (
        round((1 - unique_count / original_count) * 100, 1) if original_count > 0 else 0
    )

    return {
        "cluster_title": cluster_data.get("cluster_title", ""),
        "original_count": original_count,
        "unique_count": unique_count,
        "reduction_pct": reduction_pct,
        "questions": groups,
        "is_presupposition": is_presupposition,
    }
