"""Decompose Synthesized FAQ Questions (Pass 2).

Runs decomposition on synthesized FAQs to split any compound questions
that slipped through pass 1.

This module is called by run_deduplication.py as part of the second pass.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from config import config
from src.analysis.dedup import decompose_all_faqs


def count_atomic_faqs(results_by_cluster: dict) -> int:
    """Count total atomic FAQs after decomposition."""
    total = 0
    for cluster_data in results_by_cluster.values():
        total += len(cluster_data.get("items", []))
    return total


def run(
    input_path: str,
    output_dir: str,
) -> str:
    """Run the decomposition analysis on synthesized FAQs.

    Args:
        input_path: Path to deduplicated_questions.json from pass 1.
        output_dir: Output directory for decomposed_synthesized.json.

    Returns:
        Path to the output JSON file.
    """
    input_path_obj = Path(input_path)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / "decomposed_synthesized.json"

    print(f"Loading synthesized FAQs from {input_path_obj}...")
    with open(input_path_obj) as f:
        input_data = json.load(f)

    # Count original FAQs
    total_original = sum(
        len(cluster.get("questions", []))
        for cluster in input_data.get("by_cluster", {}).values()
    )
    print(
        f"  Found {total_original} FAQs across {len(input_data.get('by_cluster', {}))} clusters"
    )

    # Get LLM config
    max_concurrent = config["llm"].get("max_concurrency", 15)
    max_retries = config["llm"].get("max_retries", 5)

    print(f"\nConcurrency: {max_concurrent}")

    # Run decomposition using core module
    results_by_cluster = asyncio.run(
        decompose_all_faqs(input_data, max_concurrent, max_retries)
    )

    # Count results
    total_atomic = count_atomic_faqs(results_by_cluster)

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(input_path_obj),
            "total_original_faqs": total_original,
            "total_atomic_faqs": total_atomic,
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
    print(f"Original FAQs: {total_original}")
    print(
        f"Atomic FAQs: {total_atomic} ({output['metadata']['expansion_ratio']}x expansion)"
    )

    return str(output_path)
