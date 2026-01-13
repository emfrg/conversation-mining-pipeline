"""Step 11: Deduplicate FAQs - Remove duplicate questions and synthesize clean FAQs.

Self-contained pipeline step that runs the full deduplication pipeline:
1. Decompose questions into atomic units (with presuppositions)
2. Deduplicate using vector similarity
3. Synthesize clean FAQ questions using LLM
4. Optional second pass on synthesized FAQs

Features:
- Resume support: skips already-completed substeps
- --recreate support: force regeneration of all outputs

Reads:
    config.experiment.dir/clusters_named.json - Named clusters from Step 6.

Writes:
    config.experiment.dir/decomposed_questions.json
    config.experiment.dir/deduplicated_questions.json
    config.experiment.dir/decomposed_synthesized.json (if second_pass)
    config.experiment.dir/deduplicated_synthesized.json (if second_pass)
    config.experiment.dir/faq_summary.json - Final clean FAQ export
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

from config import config
from src.analysis.dedup import (
    count_atomic_questions,
    decompose_all_faqs,
    decompose_all_questions,
    get_decomposition_config,
    get_dedup_config,
    get_embedding_model,
    process_cluster_decomposed,
    process_cluster_pass2,
    synthesize_all_questions,
)
from src.utils.helpers import read_json_file

# =============================================================================
# Pass 1: Decomposition
# =============================================================================


async def _run_decomposition(
    experiment_dir: str,
    clusters_data: dict,
    with_presuppositions: bool,
    recreate: bool,
) -> tuple[str, dict]:
    """Run Pass 1 decomposition step.

    Args:
        experiment_dir: Path to experiment directory.
        clusters_data: Loaded clusters_named.json data.
        with_presuppositions: Whether to extract presuppositions.
        recreate: Whether to force regeneration.

    Returns:
        Tuple of (output_path, decomposed_data).
    """
    output_path = os.path.join(experiment_dir, "decomposed_questions.json")

    # Check for existing output (resume support)
    if not recreate and os.path.exists(output_path):
        print("    Skipping: decomposed_questions.json already exists")
        with open(output_path) as f:
            return output_path, json.load(f)

    # Count original questions
    total_original = sum(
        len(cluster.get("items", []))
        for cluster in clusters_data.get("clusters", {}).values()
    )
    print(
        f"    Found {total_original} questions across {len(clusters_data.get('clusters', {}))} clusters"
    )

    # Get LLM config
    max_concurrent = config["llm"].get("max_concurrency", 15)
    max_retries = config["llm"].get("max_retries", 5)

    mode = "with presuppositions" if with_presuppositions else "decomposition only"
    print(f"    Mode: {mode}")

    # Run decomposition using core module
    results_by_cluster = await decompose_all_questions(
        clusters_data,
        with_presuppositions,
        max_concurrent,
        max_retries,
    )

    # Count results
    total_atomic, explicit_count, presupposition_count = count_atomic_questions(
        results_by_cluster
    )

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "with_presuppositions": with_presuppositions,
            "global_mode": False,
            "total_original_questions": total_original,
            "total_atomic_questions": total_atomic,
            "explicit_questions": explicit_count,
            "presupposition_questions": presupposition_count,
            "expansion_ratio": (
                round(total_atomic / total_original, 2) if total_original > 0 else 0
            ),
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Saved to {output_path}")
    print(
        f"    Original: {total_original} -> Atomic: {total_atomic} ({output['metadata']['expansion_ratio']}x)"
    )

    return output_path, output


# =============================================================================
# Pass 1: Deduplication
# =============================================================================


async def _run_deduplication(
    experiment_dir: str,
    decomposed_data: dict,
    threshold: float,
    recreate: bool,
    with_presuppositions: bool = False,
) -> tuple[str, dict]:
    """Run Pass 1 deduplication step.

    Processes explicit questions and presuppositions separately to avoid
    over-aggregation, then merges results.

    Args:
        experiment_dir: Path to experiment directory.
        decomposed_data: Loaded decomposed_questions.json data.
        threshold: Similarity threshold for deduplication.
        recreate: Whether to force regeneration.
        with_presuppositions: Whether presuppositions were extracted.

    Returns:
        Tuple of (output_path, deduplicated_data).
    """
    output_path = os.path.join(experiment_dir, "deduplicated_questions.json")

    # Check for existing output (resume support)
    if not recreate and os.path.exists(output_path):
        print("    Skipping: deduplicated_questions.json already exists")
        with open(output_path) as f:
            return output_path, json.load(f)

    dedup_config = get_dedup_config()

    print("    Initializing embedding model...")
    embeddings_model = get_embedding_model()
    print(f"      Model: {dedup_config['embedding_model']}")
    print(f"      Dimensions: {dedup_config['dimensions']}")
    print(f"      Threshold: {threshold}")

    # Process each cluster - separate dedup for explicit questions vs presuppositions
    results_by_cluster = {}
    total_original = 0
    total_unique = 0
    total_questions = 0
    total_presuppositions = 0

    clusters = decomposed_data.get("by_cluster", {})
    for cluster_id, cluster_data in clusters.items():
        print(
            f"    Processing {cluster_id}: {cluster_data.get('cluster_title', '')}..."
        )

        # Process explicit questions
        print("      Deduplicating explicit questions...")
        explicit_result = process_cluster_decomposed(
            cluster_id,
            cluster_data,
            embeddings_model,
            threshold,
            question_type_filter="explicit",
        )

        cluster_questions = []
        for q in explicit_result.get("questions", []):
            q["is_presupposition"] = False  # Override - these are explicit questions
            cluster_questions.append(q)

        explicit_original = explicit_result["original_count"]
        explicit_unique = explicit_result["unique_count"]
        total_questions += explicit_unique

        # Process presuppositions separately (if enabled)
        presup_original = 0
        presup_unique = 0
        if with_presuppositions:
            print("      Deduplicating presuppositions...")
            presup_result = process_cluster_decomposed(
                cluster_id,
                cluster_data,
                embeddings_model,
                threshold,
                question_type_filter="presupposition",
            )
            for q in presup_result.get("questions", []):
                q["is_presupposition"] = True  # Override - these are presuppositions
                cluster_questions.append(q)

            presup_original = presup_result["original_count"]
            presup_unique = presup_result["unique_count"]
            total_presuppositions += presup_unique

        # Merge results
        cluster_original = explicit_original + presup_original
        cluster_unique = explicit_unique + presup_unique
        total_original += cluster_original
        total_unique += cluster_unique

        # Sort by unique_conversation_count descending
        cluster_questions.sort(
            key=lambda x: x.get("unique_conversation_count", 0), reverse=True
        )

        cluster_reduction_pct = (
            round((1 - cluster_unique / cluster_original) * 100, 1)
            if cluster_original > 0
            else 0
        )

        results_by_cluster[cluster_id] = {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": cluster_original,
            "unique_count": cluster_unique,
            "reduction_pct": cluster_reduction_pct,
            "questions": cluster_questions,
            "input_mode": "decomposed",
            "explicit_count": explicit_unique,
            "presupposition_count": presup_unique,
        }

        print(
            f"      {cluster_original} -> {cluster_unique} "
            f"({cluster_reduction_pct}% reduction, "
            f"{explicit_unique} questions + {presup_unique} presuppositions)"
        )

    # Synthesize FAQ questions using LLM
    max_concurrent = config["llm"].get("max_concurrency", 15)
    max_retries = config["llm"].get("max_retries", 5)
    results_by_cluster = await synthesize_all_questions(
        results_by_cluster, max_concurrent, max_retries
    )

    # Build final output
    overall_reduction_pct = (
        round((1 - total_unique / total_original) * 100, 1) if total_original > 0 else 0
    )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "similarity_threshold": threshold,
            "embedding_model": dedup_config["embedding_model"],
            "embedding_dimensions": dedup_config["dimensions"],
            "total_original_questions": total_original,
            "total_unique_questions": total_unique,
            "overall_reduction_pct": overall_reduction_pct,
            "synthesized": True,
            "input_mode": "decomposed",
            "global_mode": False,
            "separate_type_dedup": True,
            "question_count": total_questions,
            "presupposition_count": total_presuppositions,
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Saved to {output_path}")
    print(
        f"    Total: {total_original} -> {total_unique} ({overall_reduction_pct}% reduction)"
    )
    print(f"    Questions: {total_questions}, Presuppositions: {total_presuppositions}")

    return output_path, output


# =============================================================================
# Pass 2: Decompose and Deduplicate Synthesized FAQs
# =============================================================================


async def _run_pass2_decomposition(
    experiment_dir: str,
    deduplicated_data: dict,
    recreate: bool,
) -> tuple[str, dict]:
    """Run Pass 2 decomposition on synthesized FAQs.

    Args:
        experiment_dir: Path to experiment directory.
        deduplicated_data: Loaded deduplicated_questions.json data.
        recreate: Whether to force regeneration.

    Returns:
        Tuple of (output_path, decomposed_data).
    """
    output_path = os.path.join(experiment_dir, "decomposed_synthesized.json")

    # Check for existing output (resume support)
    if not recreate and os.path.exists(output_path):
        print("    Skipping: decomposed_synthesized.json already exists")
        with open(output_path) as f:
            return output_path, json.load(f)

    # Count original FAQs
    total_original = sum(
        len(cluster.get("questions", []))
        for cluster in deduplicated_data.get("by_cluster", {}).values()
    )
    print(
        f"    Found {total_original} FAQs across {len(deduplicated_data.get('by_cluster', {}))} clusters"
    )

    # Get LLM config
    max_concurrent = config["llm"].get("max_concurrency", 15)
    max_retries = config["llm"].get("max_retries", 5)

    # Run decomposition using core module
    results_by_cluster = await decompose_all_faqs(
        deduplicated_data, max_concurrent, max_retries
    )

    # Count results
    total_atomic = sum(
        len(cluster_data.get("items", []))
        for cluster_data in results_by_cluster.values()
    )

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": "deduplicated_questions.json",
            "total_original_faqs": total_original,
            "total_atomic_faqs": total_atomic,
            "expansion_ratio": (
                round(total_atomic / total_original, 2) if total_original > 0 else 0
            ),
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Saved to {output_path}")
    print(f"    Original FAQs: {total_original} -> Atomic FAQs: {total_atomic}")

    return output_path, output


def _run_pass2_deduplication(
    experiment_dir: str,
    decomposed_synthesized_data: dict,
    threshold: float,
    recreate: bool,
) -> tuple[str, dict]:
    """Run Pass 2 deduplication on decomposed synthesized FAQs.

    Processes questions and presuppositions separately to avoid
    over-aggregation, then merges results.

    Args:
        experiment_dir: Path to experiment directory.
        decomposed_synthesized_data: Loaded decomposed_synthesized.json data.
        threshold: Similarity threshold for deduplication.
        recreate: Whether to force regeneration.

    Returns:
        Tuple of (output_path, deduplicated_data).
    """
    output_path = os.path.join(experiment_dir, "deduplicated_synthesized.json")

    # Check for existing output (resume support)
    if not recreate and os.path.exists(output_path):
        print("    Skipping: deduplicated_synthesized.json already exists")
        with open(output_path) as f:
            return output_path, json.load(f)

    dedup_config = get_dedup_config()

    print("    Initializing embedding model...")
    embeddings_model = get_embedding_model()
    print(f"      Model: {dedup_config['embedding_model']}")
    print(f"      Threshold: {threshold}")

    # Process each cluster - separate dedup for questions vs presuppositions
    results_by_cluster = {}
    total_original = 0
    total_unique = 0
    total_questions = 0
    total_presuppositions = 0

    clusters = decomposed_synthesized_data.get("by_cluster", {})
    for cluster_id, cluster_data in clusters.items():
        print(
            f"    Processing {cluster_id}: {cluster_data.get('cluster_title', '')}..."
        )

        # Split items by type
        items = cluster_data.get("items", [])
        questions_items = [i for i in items if not i.get("is_presupposition", False)]
        presup_items = [i for i in items if i.get("is_presupposition", False)]

        all_questions = []

        # Process questions
        if questions_items:
            print(f"      Deduplicating {len(questions_items)} questions...")
            questions_data = {
                "cluster_title": cluster_data.get("cluster_title", ""),
                "items": questions_items,
            }
            q_result = process_cluster_pass2(
                cluster_id,
                questions_data,
                embeddings_model,
                threshold,
                is_presupposition=False,
            )
            all_questions.extend(q_result.get("questions", []))
            total_questions += q_result["unique_count"]

        # Process presuppositions
        if presup_items:
            print(f"      Deduplicating {len(presup_items)} presuppositions...")
            presup_data = {
                "cluster_title": cluster_data.get("cluster_title", ""),
                "items": presup_items,
            }
            p_result = process_cluster_pass2(
                cluster_id,
                presup_data,
                embeddings_model,
                threshold,
                is_presupposition=True,
            )
            all_questions.extend(p_result.get("questions", []))
            total_presuppositions += p_result["unique_count"]

        # Merge results
        cluster_original = len(items)
        cluster_unique = len(all_questions)
        total_original += cluster_original
        total_unique += cluster_unique

        # Sort by unique_conversation_count descending
        all_questions.sort(
            key=lambda x: x.get("unique_conversation_count", 0), reverse=True
        )

        cluster_reduction_pct = (
            round((1 - cluster_unique / cluster_original) * 100, 1)
            if cluster_original > 0
            else 0
        )

        results_by_cluster[cluster_id] = {
            "cluster_title": cluster_data.get("cluster_title", ""),
            "original_count": cluster_original,
            "unique_count": cluster_unique,
            "reduction_pct": cluster_reduction_pct,
            "questions": all_questions,
        }

        q_count = len(
            [q for q in all_questions if not q.get("is_presupposition", False)]
        )
        p_count = len([q for q in all_questions if q.get("is_presupposition", False)])
        print(
            f"      {cluster_original} -> {cluster_unique} "
            f"({cluster_reduction_pct}% reduction, "
            f"{q_count} questions + {p_count} presuppositions)"
        )

    # Build final output
    overall_reduction_pct = (
        round((1 - total_unique / total_original) * 100, 1) if total_original > 0 else 0
    )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": "decomposed_synthesized.json",
            "similarity_threshold": threshold,
            "embedding_model": dedup_config["embedding_model"],
            "embedding_dimensions": dedup_config["dimensions"],
            "total_original_faqs": total_original,
            "total_unique_faqs": total_unique,
            "overall_reduction_pct": overall_reduction_pct,
            "separate_type_dedup": True,
            "question_count": total_questions,
            "presupposition_count": total_presuppositions,
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"    Saved to {output_path}")
    print(
        f"    Total: {total_original} -> {total_unique} ({overall_reduction_pct}% reduction)"
    )
    print(f"    Questions: {total_questions}, Presuppositions: {total_presuppositions}")

    return output_path, output


# =============================================================================
# Export Summary
# =============================================================================


def _export_summary(experiment_dir: str, second_pass: bool) -> tuple[str, str]:
    """Export FAQ summary files from the deduplication results.

    Creates two files:
    - faq_summary.json: Minimal format (question, type, unique_conversations)
    - final_deduplicated.json: Detailed format (with original_ids, variants)

    Args:
        experiment_dir: Path to experiment directory.
        second_pass: Whether second pass was run.

    Returns:
        Tuple of (summary_path, detailed_path).
    """
    # Determine which file to read
    if second_pass:
        input_file = "deduplicated_synthesized.json"
    else:
        input_file = "deduplicated_questions.json"

    input_path = os.path.join(experiment_dir, input_file)
    summary_path = os.path.join(experiment_dir, "faq_summary.json")
    detailed_path = os.path.join(experiment_dir, "final_deduplicated.json")

    print(f"    Reading from {input_file}...")

    with open(input_path) as f:
        data = json.load(f)

    # Build both outputs
    summary_clusters: list[dict[str, Any]] = []
    detailed_by_cluster: dict[str, Any] = {}

    total_questions_count = 0
    total_presuppositions_count = 0
    total_unique_conversations = set()

    for cluster_id, cluster_data in data.get("by_cluster", {}).items():
        cluster_title = cluster_data.get("cluster_title", "")

        summary_cluster = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_title,
            "questions": [],
        }

        detailed_cluster = {
            "cluster_title": cluster_title,
            "questions": [],
        }

        for q in cluster_data.get("questions", []):
            # Get question text
            question_text = q.get(
                "synthesized_question", q.get("representative_question", "")
            )

            # Determine type
            is_presup = q.get("is_presupposition", False)
            q_type = "presupposition" if is_presup else "question"

            if is_presup:
                total_presuppositions_count += 1
            else:
                total_questions_count += 1

            # Get conversation count
            unique_convs = q.get("unique_conversation_count", q.get("count", 0))

            # Get original IDs
            original_ids = q.get("original_ids", [])
            total_unique_conversations.update(original_ids)

            # Get variants (handle both Pass 1 and Pass 2 formats)
            variants = q.get("variants_merged", q.get("variants", []))

            # Add to summary (minimal)
            summary_cluster["questions"].append(
                {
                    "question": question_text,
                    "type": q_type,
                    "unique_conversations": unique_convs,
                }
            )

            # Add to detailed (with traceability)
            detailed_cluster["questions"].append(
                {
                    "question": question_text,
                    "type": q_type,
                    "unique_conversations": unique_convs,
                    "original_ids": original_ids,
                    "variants": variants,
                }
            )

        # Sort questions by unique_conversations descending
        summary_cluster["questions"].sort(
            key=lambda x: x["unique_conversations"], reverse=True
        )
        detailed_cluster["questions"].sort(
            key=lambda x: x["unique_conversations"], reverse=True
        )

        summary_clusters.append(summary_cluster)
        detailed_by_cluster[cluster_id] = detailed_cluster

    # Sort clusters by total questions descending
    summary_clusters.sort(key=lambda x: len(x["questions"]), reverse=True)

    # Build final outputs
    total_faqs = total_questions_count + total_presuppositions_count

    summary_output: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "conversations_analyzed": len(total_unique_conversations),
            "total_unique_faqs": total_faqs,
            "questions": total_questions_count,
            "presuppositions": total_presuppositions_count,
        },
        "clusters": summary_clusters,
    }

    detailed_output: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "conversations_analyzed": len(total_unique_conversations),
            "total_unique_faqs": total_faqs,
            "questions": total_questions_count,
            "presuppositions": total_presuppositions_count,
            "source_file": input_file,
            "second_pass": second_pass,
        },
        "by_cluster": detailed_by_cluster,
    }

    # Save summary (minimal)
    with open(summary_path, "w") as f:
        json.dump(summary_output, f, indent=2)

    # Save detailed (with traceability)
    with open(detailed_path, "w") as f:
        json.dump(detailed_output, f, indent=2)

    print(f"    Saved minimal summary to {summary_path}")
    print(f"    Saved detailed file to {detailed_path}")
    print(f"      Clusters: {len(summary_clusters)}")
    print(
        f"      Total FAQs: {total_faqs} ({total_questions_count} questions + {total_presuppositions_count} presuppositions)"
    )

    return summary_path, detailed_path


# =============================================================================
# Main Entry Point
# =============================================================================


async def _run_pipeline_async(
    experiment_dir: str,
    clusters_data: dict,
    threshold: float,
    second_pass_threshold: float,
    second_pass: bool,
    with_presuppositions: bool,
    recreate: bool,
) -> str:
    """Run the full deduplication pipeline with a single event loop.

    This function orchestrates all async operations to avoid multiple
    asyncio.run() calls, which prevents httpx client cleanup errors.

    Args:
        experiment_dir: Path to experiment directory.
        clusters_data: Loaded clusters_named.json data.
        threshold: Similarity threshold for Pass 1 deduplication.
        second_pass_threshold: Similarity threshold for Pass 2 deduplication.
        second_pass: Whether to run second pass.
        with_presuppositions: Whether to extract presuppositions.
        recreate: Whether to force regeneration.

    Returns:
        Path to the faq_summary.json file.
    """
    print()
    print("  " + "=" * 56)
    print("  STEP 1: Question Decomposition")
    print("  " + "=" * 56)
    _, decomposed_data = await _run_decomposition(
        experiment_dir,
        clusters_data,
        with_presuppositions,
        recreate,
    )

    print()
    print("  " + "=" * 56)
    print("  STEP 2: Question Deduplication (Pass 1)")
    print("  " + "=" * 56)
    _, deduplicated_data = await _run_deduplication(
        experiment_dir,
        decomposed_data,
        threshold,
        recreate,
        with_presuppositions,
    )

    if second_pass:
        print()
        print("  " + "=" * 56)
        print("  STEP 3: Pass 2 Decomposition (on synthesized FAQs)")
        print("  " + "=" * 56)
        _, decomposed_synthesized_data = await _run_pass2_decomposition(
            experiment_dir,
            deduplicated_data,
            recreate,
        )

        print()
        print("  " + "=" * 56)
        print("  STEP 4: Pass 2 Deduplication")
        print("  " + "=" * 56)
        _run_pass2_deduplication(
            experiment_dir,
            decomposed_synthesized_data,
            second_pass_threshold,
            recreate,
        )

    print()
    print("  " + "=" * 56)
    print("  FINAL: Export FAQ Summary")
    print("  " + "=" * 56)
    summary_path, _ = _export_summary(experiment_dir, second_pass)

    # Create visualizations
    print()
    print("  " + "=" * 56)
    print("  VISUALIZATION: FAQ Questions Charts")
    print("  " + "=" * 56)
    from src.vis.faq_questions_chart import create_faq_questions_chart

    # Questions only (no presuppositions)
    vis_path_questions = os.path.join(experiment_dir, "faq_questions.png")
    create_faq_questions_chart(
        summary_path, vis_path_questions, show_presuppositions=False
    )

    # Questions with presuppositions
    vis_path_all = os.path.join(
        experiment_dir, "faq_questions_with_presuppositions.png"
    )
    create_faq_questions_chart(summary_path, vis_path_all, show_presuppositions=True)

    return summary_path


def run() -> str:
    """Run the deduplicate_faqs pipeline step.

    Runs the full deduplication pipeline with resume support.
    Uses a single event loop to avoid httpx client cleanup errors.

    Returns:
        Path to the faq_summary.json file.
    """
    experiment_dir = config["experiment"]["dir"]
    recreate = config.get("recreate", False)

    # Get config
    dedup_config = get_dedup_config()
    decomp_config = get_decomposition_config()

    threshold = dedup_config["similarity_threshold"]
    second_pass_threshold = dedup_config["second_pass_threshold"]
    second_pass = dedup_config["second_pass"]
    with_presuppositions = decomp_config["with_presuppositions"]

    # Check if final output exists (skip if not recreating)
    summary_path = os.path.join(experiment_dir, "faq_summary.json")
    if not recreate and os.path.exists(summary_path):
        print(f"  Skipping: {summary_path} already exists")
        print("  Use --recreate to regenerate")
        return summary_path

    # Load input data
    clusters_path = os.path.join(experiment_dir, "clusters_named.json")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(
            f"clusters_named.json not found at {clusters_path}. "
            "Run the clustering pipeline first (steps 1-6)."
        )

    print(f"  Loading clusters from {clusters_path}...")
    clusters_data = read_json_file(clusters_path)

    # Run the full pipeline with a single event loop
    return asyncio.run(
        _run_pipeline_async(
            experiment_dir,
            clusters_data,
            threshold,
            second_pass_threshold,
            second_pass,
            with_presuppositions,
            recreate,
        )
    )


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment_from_latest

    # Set up experiment directory for standalone execution
    setup_experiment_from_latest(config)
    run()
