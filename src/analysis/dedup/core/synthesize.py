"""FAQ synthesis functions using LLM.

Generates clean, canonical FAQ questions from groups of similar questions.
Uses the faq_synthesizer_agent to combine representative question and variants
into a single well-formed question.

Provides:
- synthesize_single_question: Async function to synthesize one question group
- synthesize_all_questions: Async batch synthesis across all clusters

All functions are async and require `await` when called.
Uses JSON response format with retry logic for parsing failures.
"""

import asyncio
from typing import Any

from src.schemas import FAQSynthesis


async def synthesize_single_question(
    agent: Any,
    group: dict,
    max_retries: int,
) -> str | None:
    """Synthesize a clean FAQ question from a group using LLM.

    Args:
        agent: The FAQ synthesizer agent.
        group: Question group with representative and variants.
        max_retries: Max retries for LLM failures.

    Returns:
        Synthesized question string, or None on failure.
    """
    representative = group["representative_question"]
    variants = group.get("variants", [])

    # Format variants for prompt
    if variants:
        variants_text = "\n".join(f"- {v}" for v in variants)
    else:
        variants_text = "(no variants - single question)"

    for attempt in range(max_retries):
        try:
            response: FAQSynthesis = await agent.ainvoke(
                {"representative": representative, "variants": variants_text}
            )
            return response.faq_question
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)

    return None


async def synthesize_all_questions(
    results_by_cluster: dict,
    max_concurrent: int,
    max_retries: int,
) -> dict:
    """Synthesize FAQ questions for all groups across all clusters.

    Args:
        results_by_cluster: Dict of cluster results with question groups.
        max_concurrent: Maximum concurrent LLM calls.
        max_retries: Max retries per question.

    Returns:
        Updated results_by_cluster with synthesized_question added.
    """
    # Import agent here to avoid circular imports and allow --no-synthesize to skip loading
    from src.agents.faq_synthesizer_agent import faq_synthesizer

    # Collect all groups with their cluster/index references
    all_groups: list[tuple[str, int, dict]] = []
    for cluster_id, cluster_result in results_by_cluster.items():
        for idx, group in enumerate(cluster_result["questions"]):
            all_groups.append((cluster_id, idx, group))

    total = len(all_groups)
    print(f"    Synthesizing FAQ questions for {total} unique question groups...")

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0

    async def process_one(
        cluster_id: str, idx: int, group: dict
    ) -> tuple[str, int, str | None]:
        async with semaphore:
            result = await synthesize_single_question(
                faq_synthesizer, group, max_retries
            )
            return (cluster_id, idx, result)

    # Process in batches for progress reporting
    batch_size = max_concurrent * 2
    for i in range(0, len(all_groups), batch_size):
        batch = all_groups[i : i + batch_size]

        tasks = [process_one(cid, idx, grp) for cid, idx, grp in batch]
        results = await asyncio.gather(*tasks)

        for cluster_id, idx, synthesized in results:
            if synthesized:
                results_by_cluster[cluster_id]["questions"][idx][
                    "synthesized_question"
                ] = synthesized
                completed += 1
            else:
                # Fallback to representative question
                rep = results_by_cluster[cluster_id]["questions"][idx][
                    "representative_question"
                ]
                results_by_cluster[cluster_id]["questions"][idx][
                    "synthesized_question"
                ] = rep
                failed += 1

        done = i + len(batch)
        print(
            f"      Progress: {done}/{total} ({completed} synthesized, {failed} fallback)"
        )

        # Small delay between batches
        if i + batch_size < len(all_groups):
            await asyncio.sleep(0.5)

    print(f"    Synthesis complete: {completed} synthesized, {failed} used fallback")
    return results_by_cluster
