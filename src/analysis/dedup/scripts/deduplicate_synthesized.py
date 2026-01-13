"""Deduplicate Decomposed Synthesized FAQs (Pass 2).

Runs deduplication on atomic FAQs from decompose_synthesized.py.
Uses second_pass_threshold from config.

This module is called by run_deduplication.py as part of the second pass.
"""

import json
from datetime import datetime
from pathlib import Path

from src.analysis.dedup import (
    get_dedup_config,
    get_embedding_model,
    process_cluster_pass2,
)


def run(
    input_path: str,
    output_dir: str,
    similarity_threshold: float | None = None,
) -> str:
    """Run the deduplication analysis on decomposed synthesized FAQs.

    Args:
        input_path: Path to decomposed_synthesized.json from pass 2 decomposition.
        output_dir: Output directory for deduplicated_synthesized.json.
        similarity_threshold: Cosine similarity threshold (default from config).

    Returns:
        Path to the output JSON file.
    """
    # Get deduplication config
    dedup_config = get_dedup_config()
    if similarity_threshold is None:
        similarity_threshold = dedup_config["second_pass_threshold"]

    input_path_obj = Path(input_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / "deduplicated_synthesized.json"

    print(f"Loading decomposed FAQs from {input_path_obj}...")
    with open(input_path_obj) as f:
        input_data = json.load(f)

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

        result = process_cluster_pass2(
            cluster_id, cluster_data, embeddings_model, similarity_threshold
        )

        results_by_cluster[cluster_id] = result
        total_original += result["original_count"]
        total_unique += result["unique_count"]
        print(
            f"    {result['original_count']} -> {result['unique_count']} "
            f"({result['reduction_pct']}% reduction)"
        )

    # Build final output
    overall_reduction_pct = (
        round((1 - total_unique / total_original) * 100, 1) if total_original > 0 else 0
    )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path_obj),
            "similarity_threshold": similarity_threshold,
            "embedding_model": dedup_config["embedding_model"],
            "embedding_dimensions": dedup_config["dimensions"],
            "total_original_faqs": total_original,
            "total_unique_faqs": total_unique,
            "overall_reduction_pct": overall_reduction_pct,
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(
        f"Total: {total_original} -> {total_unique} ({overall_reduction_pct}% reduction)"
    )

    return str(output_path)
