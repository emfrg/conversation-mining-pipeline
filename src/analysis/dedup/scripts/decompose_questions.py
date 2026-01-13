"""Decompose FAQ Questions into Atomic Units.

Breaks compound questions into atomic sub-questions and optionally extracts
context-relevant presuppositions. This step runs BEFORE deduplication.

Reads:
    config.experiment.dir/clusters_named.json - Named clusters with items.

Writes:
    {output_dir}/decomposed_questions.json - Decomposed questions with atomic units.

Usage:
    python -m src.analysis.dedup.scripts.decompose_questions
    python -m src.analysis.dedup.scripts.decompose_questions --with-presuppositions
    python -m src.analysis.dedup.scripts.decompose_questions --global
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from config import config
from src.analysis.dedup import count_atomic_questions, decompose_all_questions
from src.utils.helpers import read_json_file


def run(
    with_presuppositions: bool = False,
    output_dir: str | None = None,
    global_mode: bool = False,
) -> str:
    """Run the question decomposition analysis.

    Args:
        with_presuppositions: Whether to extract context-relevant presuppositions.
        output_dir: Output directory (default: src/vis/outputs/).
        global_mode: If True, flatten all clusters into a single pseudo-cluster.

    Returns:
        Path to the output JSON file.
    """
    from src.utils.experiment_manager import setup_experiment_from_latest

    # Set up experiment directory
    setup_experiment_from_latest(config)
    experiment_dir = config["experiment"]["dir"]
    clusters_path = os.path.join(experiment_dir, "clusters_named.json")

    # Set output directory
    output_dir_path = (
        Path("src/vis/outputs") if output_dir is None else Path(output_dir)
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / "decomposed_questions.json"

    print(f"Loading clusters from {clusters_path}...")
    clusters_data = read_json_file(clusters_path)

    # If global mode, flatten all clusters into one
    if global_mode:
        all_items = []
        for cluster_data in clusters_data.get("clusters", {}).values():
            all_items.extend(cluster_data.get("items", []))
        clusters_data = {
            "clusters": {
                "all": {
                    "cluster_title": "All Questions (Global Deduplication)",
                    "items": all_items,
                }
            }
        }
        print(f"  Global mode: flattened to single cluster with {len(all_items)} items")

    # Count original questions
    total_original = sum(
        len(cluster.get("items", []))
        for cluster in clusters_data.get("clusters", {}).values()
    )
    print(
        f"  Found {total_original} questions across {len(clusters_data.get('clusters', {}))} clusters"
    )

    # Get LLM config
    max_concurrent = config["llm"].get("max_concurrency", 15)
    max_retries = config["llm"].get("max_retries", 5)

    mode = "with presuppositions" if with_presuppositions else "decomposition only"
    print(f"\nMode: {mode}")
    print(f"Concurrency: {max_concurrent}")

    # Run decomposition using core module
    results_by_cluster = asyncio.run(
        decompose_all_questions(
            clusters_data,
            with_presuppositions,
            max_concurrent,
            max_retries,
        )
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
            "global_mode": global_mode,
            "total_original_questions": total_original,
            "total_atomic_questions": total_atomic,
            "explicit_questions": explicit_count,
            "presupposition_questions": presupposition_count,
            "expansion_ratio": round(total_atomic / total_original, 2)
            if total_original > 0
            else 0,
        },
        "by_cluster": results_by_cluster,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"Original questions: {total_original}")
    print(
        f"Atomic questions: {total_atomic} ({output['metadata']['expansion_ratio']}x expansion)"
    )
    if with_presuppositions:
        print(f"  - Explicit: {explicit_count}")
        print(f"  - Presuppositions: {presupposition_count}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decompose FAQ questions into atomic sub-questions."
    )
    parser.add_argument(
        "--with-presuppositions",
        action="store_true",
        help="Also extract context-relevant presuppositions as questions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: src/vis/outputs/)",
    )
    parser.add_argument(
        "--global",
        dest="global_mode",
        action="store_true",
        help="Flatten all clusters into a single pseudo-cluster for global deduplication",
    )
    args = parser.parse_args()

    result_path = run(
        with_presuppositions=args.with_presuppositions,
        output_dir=args.output_dir,
        global_mode=args.global_mode,
    )
