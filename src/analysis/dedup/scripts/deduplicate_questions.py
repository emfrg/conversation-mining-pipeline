"""De-duplicate FAQ Questions Per Cluster.

Uses vector similarity with fresh embeddings computed ONLY for
canonical_faq_question text to identify duplicate questions within
each cluster.

Supports two input modes:
1. Decomposed mode (default): Uses decomposed_questions.json if available,
   deduplicating on atomic questions (including presuppositions as questions).
2. Legacy mode (--no-decompose): Uses clusters_named.json directly.

Optionally uses LLM to synthesize clean FAQ questions from each group.

Reads:
    {output_dir}/decomposed_questions.json - Decomposed questions (preferred).
    OR config.experiment.dir/clusters_named.json - Named clusters with items.

Writes:
    {output_dir}/deduplicated_questions.json - Unique questions with counts.

Usage:
    python -m src.analysis.dedup.scripts.deduplicate_questions
    python -m src.analysis.dedup.scripts.deduplicate_questions --threshold 0.95
    python -m src.analysis.dedup.scripts.deduplicate_questions --no-synthesize
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from config import config
from src.analysis.dedup import (
    get_dedup_config,
    get_embedding_model,
    process_cluster,
    process_cluster_decomposed,
    synthesize_all_questions,
)
from src.utils.helpers import read_json_file


def run(
    similarity_threshold: float | None = None,
    output_dir: str | None = None,
    synthesize: bool = True,
    use_decomposed: bool = True,
    global_mode: bool = False,
) -> str:
    """Run the de-duplication analysis.

    Args:
        similarity_threshold: Cosine similarity threshold (default from config).
        output_dir: Output directory (default: src/vis/outputs/).
        synthesize: Whether to use LLM to synthesize clean FAQ questions.
        use_decomposed: Whether to use decomposed_questions.json if available.
        global_mode: If True, flatten all clusters for global deduplication (legacy mode only).

    Returns:
        Path to the output JSON file.
    """
    from src.utils.experiment_manager import setup_experiment_from_latest

    # Get deduplication config
    dedup_config = get_dedup_config()
    if similarity_threshold is None:
        similarity_threshold = dedup_config["similarity_threshold"]

    # Set up experiment directory
    setup_experiment_from_latest(config)
    experiment_dir = config["experiment"]["dir"]
    clusters_path = os.path.join(experiment_dir, "clusters_named.json")

    # Set output directory
    output_dir_path = (
        Path("src/vis/outputs") if output_dir is None else Path(output_dir)
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / "deduplicated_questions.json"

    # Check for decomposed input
    decomposed_path = output_dir_path / "decomposed_questions.json"
    input_mode = "legacy"

    if use_decomposed and decomposed_path.exists():
        print(f"Loading decomposed questions from {decomposed_path}...")
        input_data = read_json_file(str(decomposed_path))
        input_mode = "decomposed"
        with_presuppositions = input_data.get("metadata", {}).get(
            "with_presuppositions", False
        )
        decomp_global_mode = input_data.get("metadata", {}).get("global_mode", False)
        print(
            f"  Mode: decomposed (presuppositions: {with_presuppositions}, global: {decomp_global_mode})"
        )
    else:
        if use_decomposed:
            print("No decomposed_questions.json found, falling back to clusters...")
        print(f"Loading clusters from {clusters_path}...")
        input_data = read_json_file(clusters_path)
        clusters_dict = input_data.get("clusters", {})

        # If global mode, flatten all clusters into one
        if global_mode:
            all_items = []
            for cluster_data in clusters_dict.values():
                all_items.extend(cluster_data.get("items", []))
            clusters_dict = {
                "all": {
                    "cluster_title": "All Questions (Global Deduplication)",
                    "items": all_items,
                }
            }
            print(
                f"  Global mode: flattened to single cluster with {len(all_items)} items"
            )

        input_data = {"by_cluster": clusters_dict}

    print("Initializing embedding model...")
    embeddings_model = get_embedding_model()
    print(f"  Model: {dedup_config['embedding_model']}")
    print(f"  Dimensions: {dedup_config['dimensions']}")
    print(f"  Threshold: {similarity_threshold}")

    # Process each cluster
    results_by_cluster = {}
    total_original = 0
    total_unique = 0

    clusters = input_data.get("by_cluster", {})
    for cluster_id, cluster_data in clusters.items():
        print(f"\nProcessing {cluster_id}: {cluster_data.get('cluster_title', '')}...")

        if input_mode == "decomposed":
            result = process_cluster_decomposed(
                cluster_id, cluster_data, embeddings_model, similarity_threshold
            )
        else:
            result = process_cluster(
                cluster_id, cluster_data, embeddings_model, similarity_threshold
            )

        results_by_cluster[cluster_id] = result
        total_original += result["original_count"]
        total_unique += result["unique_count"]
        print(
            f"    {result['original_count']} -> {result['unique_count']} "
            f"({result['reduction_pct']}% reduction)"
        )

    # Synthesize FAQ questions using LLM
    if synthesize:
        max_concurrent = config["llm"].get("max_concurrency", 15)
        max_retries = config["llm"].get("max_retries", 5)
        results_by_cluster = asyncio.run(
            synthesize_all_questions(results_by_cluster, max_concurrent, max_retries)
        )

    # Build final output
    overall_reduction_pct = (
        round((1 - total_unique / total_original) * 100, 1) if total_original > 0 else 0
    )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "similarity_threshold": similarity_threshold,
            "embedding_model": dedup_config["embedding_model"],
            "embedding_dimensions": dedup_config["dimensions"],
            "total_original_questions": total_original,
            "total_unique_questions": total_unique,
            "overall_reduction_pct": overall_reduction_pct,
            "synthesized": synthesize,
            "input_mode": input_mode,
            "global_mode": global_mode,
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"Input mode: {input_mode}")
    print(
        f"Total: {total_original} -> {total_unique} ({overall_reduction_pct}% reduction)"
    )
    if synthesize:
        print("FAQ questions synthesized with LLM")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="De-duplicate FAQ questions per cluster using vector similarity."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold (default: from config.deduplication)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: src/vis/outputs/)",
    )
    parser.add_argument(
        "--no-synthesize",
        action="store_true",
        help="Skip LLM synthesis of FAQ questions",
    )
    parser.add_argument(
        "--no-decompose",
        action="store_true",
        help="Skip using decomposed_questions.json, use clusters_named.json directly",
    )
    parser.add_argument(
        "--global",
        dest="global_mode",
        action="store_true",
        help="Flatten all clusters for global deduplication (legacy mode only)",
    )
    args = parser.parse_args()

    result_path = run(
        similarity_threshold=args.threshold,
        output_dir=args.output_dir,
        synthesize=not args.no_synthesize,
        use_decomposed=not args.no_decompose,
        global_mode=args.global_mode,
    )
