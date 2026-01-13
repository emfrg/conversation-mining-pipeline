"""Question decomposition functions.

Breaks compound questions into atomic sub-questions using LLM.

Provides:
- decompose_single_question: Async function to decompose one question
- decompose_all_questions: Async batch decomposition for original questions
- decompose_all_faqs: Async batch decomposition for Pass 2 synthesized FAQs

All functions are async and require `await` when called.
Uses the question_decomposer_agent for LLM calls with retry logic.
"""

import asyncio
from typing import Any

from src.agents.question_decomposer_agent import get_question_decomposer
from src.schemas import QuestionDecomposition


async def decompose_single_question(
    agent: Any,
    question: str,
    max_retries: int,
) -> dict | None:
    """Decompose a single question using LLM.

    Args:
        agent: The question decomposer agent.
        question: The FAQ question to decompose.
        max_retries: Max retries for LLM failures.

    Returns:
        Decomposition result dict with atomic_questions, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            response: QuestionDecomposition = await agent.ainvoke(
                {"question": question}
            )
            return response.model_dump()  # type: ignore[no-any-return]
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)

    return None


async def decompose_all_questions(
    clusters_data: dict,
    with_presuppositions: bool,
    max_concurrent: int,
    max_retries: int,
) -> dict:
    """Decompose all questions across all clusters.

    Args:
        clusters_data: Dict with clusters structure (either "clusters" or "by_cluster" key).
        with_presuppositions: Whether to extract presuppositions.
        max_concurrent: Maximum concurrent LLM calls.
        max_retries: Max retries per question.

    Returns:
        Dict with decomposed questions by cluster (using "by_cluster" structure).
    """
    # Get the appropriate agent
    agent = get_question_decomposer(with_presuppositions=with_presuppositions)

    # Support both input formats: "clusters" (from clusters_named.json) and "by_cluster"
    if "clusters" in clusters_data:
        source_clusters = clusters_data["clusters"]
        items_key = "items"
        question_key = "canonical_faq_question"
    else:
        source_clusters = clusters_data.get("by_cluster", {})
        items_key = "items"
        question_key = "canonical_faq_question"

    # Collect all questions with their cluster/item references
    all_items: list[tuple[str, str, str, dict]] = []
    for cluster_id, cluster_data in source_clusters.items():
        for item in cluster_data.get(items_key, []):
            question = item.get(question_key, "")
            if question:
                all_items.append((cluster_id, item["id"], question, item))

    total = len(all_items)
    print(f"    Decomposing {total} questions...")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0

    async def process_one(
        cluster_id: str, item_id: str, question: str, item: dict
    ) -> tuple[str, str, dict, dict | None]:
        nonlocal completed, failed
        async with semaphore:
            result = await decompose_single_question(agent, question, max_retries)
            return (cluster_id, item_id, item, result)

    # Initialize cluster structure
    results_by_cluster: dict[str, dict] = {}
    for cluster_id, cluster_data in source_clusters.items():
        results_by_cluster[cluster_id] = {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "items": [],
        }

    # Process in batches for progress reporting
    batch_size = max_concurrent * 2
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i : i + batch_size]

        tasks = [
            process_one(cluster_id, item_id, question, item)
            for cluster_id, item_id, question, item in batch
        ]
        results = await asyncio.gather(*tasks)

        for cluster_id, item_id, item, decomposition in results:
            if decomposition:
                decomposed_item = {
                    "original_id": item_id,
                    "original_question": item.get(question_key, ""),
                    "question_type": decomposition.get("question_type", "simple"),
                    "atomic_questions": decomposition.get("atomic_questions", []),
                }
                results_by_cluster[cluster_id]["items"].append(decomposed_item)
                completed += 1
            else:
                # Fallback: treat original question as single atomic question
                decomposed_item = {
                    "original_id": item_id,
                    "original_question": item.get(question_key, ""),
                    "question_type": "simple",
                    "atomic_questions": [
                        {
                            "text": item.get(question_key, ""),
                            "type": "explicit",
                        }
                    ],
                }
                results_by_cluster[cluster_id]["items"].append(decomposed_item)
                failed += 1

        done = i + len(batch)
        print(
            f"      Progress: {done}/{total} ({completed} decomposed, {failed} fallback)"
        )

        # Small delay between batches
        if i + batch_size < len(all_items):
            await asyncio.sleep(0.5)

    print(f"    Decomposition complete: {completed} decomposed, {failed} used fallback")
    return results_by_cluster


def count_atomic_questions(results_by_cluster: dict) -> tuple[int, int, int]:
    """Count total atomic questions and breakdown by type.

    Args:
        results_by_cluster: Decomposed results by cluster.

    Returns:
        Tuple of (total_atomic, explicit_count, presupposition_count).
    """
    total = 0
    explicit = 0
    presupposition = 0

    for cluster_data in results_by_cluster.values():
        for item in cluster_data.get("items", []):
            for atomic in item.get("atomic_questions", []):
                total += 1
                if atomic.get("type") == "presupposition":
                    presupposition += 1
                else:
                    explicit += 1

    return total, explicit, presupposition


# =============================================================================
# Pass 2 Decomposition (for synthesized FAQs)
# =============================================================================


async def decompose_single_faq(
    agent: Any,
    faq_text: str,
    max_retries: int,
) -> dict | None:
    """Decompose a single synthesized FAQ using LLM (no presuppositions for pass 2).

    Args:
        agent: The question decomposer agent.
        faq_text: The synthesized FAQ question to decompose.
        max_retries: Max retries for LLM failures.

    Returns:
        Decomposition result dict, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            response: QuestionDecomposition = await agent.ainvoke(
                {"question": faq_text}
            )
            return response.model_dump()  # type: ignore[no-any-return]
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)

    return None


async def decompose_all_faqs(
    input_data: dict,
    max_concurrent: int,
    max_retries: int,
) -> dict:
    """Decompose all synthesized FAQs across all clusters (Pass 2).

    Args:
        input_data: Dict with by_cluster from deduplicated_questions.json.
        max_concurrent: Maximum concurrent LLM calls.
        max_retries: Max retries per FAQ.

    Returns:
        Dict with decomposed FAQs by cluster.
    """
    # Get the decomposer agent (no presuppositions for pass 2)
    agent = get_question_decomposer(with_presuppositions=False)

    # Collect all FAQs with their cluster references
    all_faqs: list[tuple[str, int, dict]] = []
    for cluster_id, cluster_data in input_data.get("by_cluster", {}).items():
        for idx, faq_group in enumerate(cluster_data.get("questions", [])):
            all_faqs.append((cluster_id, idx, faq_group))

    total = len(all_faqs)
    print(f"    Decomposing {total} synthesized FAQs...")

    semaphore = asyncio.Semaphore(max_concurrent)
    compound_count = 0
    simple_count = 0

    async def process_one(
        cluster_id: str, idx: int, faq_group: dict
    ) -> tuple[str, int, dict, dict | None]:
        async with semaphore:
            synthesized = faq_group.get(
                "synthesized_question", faq_group.get("representative_question", "")
            )
            result = await decompose_single_faq(agent, synthesized, max_retries)
            return (cluster_id, idx, faq_group, result)

    # Initialize results structure
    results_by_cluster: dict[str, dict] = {}
    for cluster_id, cluster_data in input_data.get("by_cluster", {}).items():
        results_by_cluster[cluster_id] = {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "items": [],
        }

    # Process in batches for progress reporting
    batch_size = max_concurrent * 2
    for i in range(0, len(all_faqs), batch_size):
        batch = all_faqs[i : i + batch_size]

        tasks = [
            process_one(cluster_id, idx, faq_group)
            for cluster_id, idx, faq_group in batch
        ]
        results = await asyncio.gather(*tasks)

        for cluster_id, _idx, faq_group, decomposition in results:
            synthesized = faq_group.get(
                "synthesized_question", faq_group.get("representative_question", "")
            )
            original_ids = faq_group.get("original_ids", [])
            unique_conv_count = faq_group.get("unique_conversation_count", 0)

            # Preserve the is_presupposition flag from Pass 1
            is_presup = faq_group.get("is_presupposition", False)

            if decomposition and decomposition.get("question_type") == "compound":
                compound_count += 1
                # Split into multiple atomic FAQs, each inheriting original_ids
                for atomic in decomposition.get("atomic_questions", []):
                    decomposed_item = {
                        "synthesized_question": atomic.get("text", ""),
                        "question_type": "explicit",
                        "original_ids": original_ids,
                        "unique_conversation_count": unique_conv_count,
                        "parent_faq": synthesized,
                        "is_split": True,
                        "is_presupposition": is_presup,
                    }
                    results_by_cluster[cluster_id]["items"].append(decomposed_item)
            else:
                simple_count += 1
                # Keep as-is (simple question)
                decomposed_item = {
                    "synthesized_question": synthesized,
                    "question_type": "simple",
                    "original_ids": original_ids,
                    "unique_conversation_count": unique_conv_count,
                    "is_split": False,
                    "is_presupposition": is_presup,
                }
                results_by_cluster[cluster_id]["items"].append(decomposed_item)

        done = i + len(batch)
        print(
            f"      Progress: {done}/{total} ({compound_count} compound, {simple_count} simple)"
        )

        # Small delay between batches
        if i + batch_size < len(all_faqs):
            await asyncio.sleep(0.5)

    print(
        f"    Decomposition complete: {compound_count} compound, {simple_count} simple"
    )
    return results_by_cluster
