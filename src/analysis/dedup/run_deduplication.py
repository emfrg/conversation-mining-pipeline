"""Deduplication Pipeline Orchestrator.

Orchestrates the full deduplication pipeline:
1. Question decomposition (with optional presuppositions)
2. Question deduplication (with synthesis)
3. Optional second pass on synthesized FAQs

Supports two modes:
- within-cluster: Deduplicate within each cluster separately (default)
- global: Flatten all clusters and deduplicate globally

Each run creates a timestamped output directory with all artifacts.
"""

import argparse
import os
from typing import Any

from config import config
from src.analysis.dedup.dedup_experiment_manager import (
    save_config_snapshot,
    setup_run,
    update_latest_symlink,
)
from src.analysis.dedup.scripts.decompose_questions import run as decompose_run
from src.analysis.dedup.scripts.deduplicate_questions import run as deduplicate_run
from src.utils.experiment_manager import setup_experiment_from_latest


def run(
    mode: str = "within-cluster",
    with_presuppositions: bool = True,
    second_pass: bool = True,
    threshold: float | None = None,
    second_pass_threshold: float | None = None,
    output_dir: str | None = None,
) -> str:
    """Orchestrate the full deduplication pipeline.

    Args:
        mode: "within-cluster" or "global"
        with_presuppositions: Extract presuppositions during decomposition (default ON)
        second_pass: Run second pass on synthesized FAQs
        threshold: Similarity threshold for pass 1 (from config if None)
        second_pass_threshold: Similarity threshold for pass 2 (from config if None)
        output_dir: Output directory (auto-generate timestamped if None)

    Returns:
        Path to the output directory
    """
    # Get thresholds from config if not provided
    dedup_config = config.get("deduplication", {})
    if threshold is None:
        threshold = dedup_config.get("similarity_threshold", 0.95)
    if second_pass_threshold is None:
        second_pass_threshold = dedup_config.get("second_pass_threshold", 0.85)

    # Set up experiment from latest clustering run
    setup_experiment_from_latest(config)
    experiment_dir = config.get("experiment", {}).get("dir", "")
    clusters_path = os.path.join(experiment_dir, "clusters_named.json")

    if not os.path.exists(clusters_path):
        raise FileNotFoundError(
            f"clusters_named.json not found at {clusters_path}. "
            "Run the clustering pipeline first."
        )

    print("=" * 60)
    print("DEDUPLICATION PIPELINE")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Presuppositions: {'ON' if with_presuppositions else 'OFF'}")
    print(f"Second pass: {'ON' if second_pass else 'OFF'}")
    print(f"Threshold (pass 1): {threshold}")
    if second_pass:
        print(f"Threshold (pass 2): {second_pass_threshold}")
    print(f"Input: {clusters_path}")

    # Set up run directory
    run_dir: str
    if output_dir is None:
        setup_run(config, mode)
        run_dir = str(config["dedup_run"]["dir"])
    else:
        run_dir = output_dir
        os.makedirs(run_dir, exist_ok=True)
        config["dedup_run"] = {"id": "custom", "dir": run_dir, "mode": mode}

    print(f"Output: {run_dir}")
    print("=" * 60)

    # Save config snapshot
    save_config_snapshot(
        config=config,
        run_dir=run_dir,
        with_presuppositions=with_presuppositions,
        second_pass=second_pass,
        threshold=threshold,
        second_pass_threshold=second_pass_threshold,
        input_clusters_path=clusters_path,
    )

    global_mode = mode == "global"

    # Step 1: Decomposition
    print("\n" + "=" * 60)
    print("STEP 1: Question Decomposition")
    print("=" * 60)
    decompose_run(
        with_presuppositions=with_presuppositions,
        output_dir=run_dir,
        global_mode=global_mode,
    )

    # Step 2: Deduplication
    print("\n" + "=" * 60)
    print("STEP 2: Question Deduplication")
    print("=" * 60)
    deduplicate_run(
        similarity_threshold=threshold,
        output_dir=run_dir,
        synthesize=True,
        use_decomposed=True,
        global_mode=global_mode,
    )

    # Step 3: Optional second pass
    if second_pass:
        print("\n" + "=" * 60)
        print("STEP 3: Second Pass (on synthesized FAQs)")
        print("=" * 60)

        from src.analysis.dedup.scripts.decompose_synthesized import (
            run as decompose_pass2,
        )
        from src.analysis.dedup.scripts.deduplicate_synthesized import (
            run as deduplicate_pass2,
        )

        pass1_output = os.path.join(run_dir, "deduplicated_questions.json")
        decomposed_output = os.path.join(run_dir, "decomposed_synthesized.json")

        decompose_pass2(
            input_path=pass1_output,
            output_dir=run_dir,
        )
        deduplicate_pass2(
            input_path=decomposed_output,
            output_dir=run_dir,
            similarity_threshold=second_pass_threshold,
        )

    # Step 4: Export summary
    print("\n" + "=" * 60)
    print("STEP 4: Export FAQ Summary")
    print("=" * 60)
    _export_summary(run_dir, second_pass)

    # Update latest symlink
    update_latest_symlink(run_dir, mode)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output: {run_dir}")

    return run_dir


def _export_summary(run_dir: str, second_pass: bool) -> str:
    """Export a clean FAQ summary from the deduplication results.

    Args:
        run_dir: Path to the run directory
        second_pass: Whether second pass was run

    Returns:
        Path to the summary file
    """
    import json

    # Determine which file to read
    if second_pass:
        input_file = "deduplicated_synthesized.json"
    else:
        input_file = "deduplicated_questions.json"

    input_path = os.path.join(run_dir, input_file)
    output_path = os.path.join(run_dir, "faq_summary.json")

    print(f"Reading from {input_file}...")

    with open(input_path) as f:
        data = json.load(f)

    # Build clean output
    output: dict[str, Any] = {"clusters": []}

    for cluster_id, cluster_data in data.get("by_cluster", {}).items():
        cluster_entry = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_data.get("cluster_title", ""),
            "questions": [],
        }

        for q in cluster_data.get("questions", []):
            question_entry = {
                "question": q.get(
                    "synthesized_question", q.get("representative_question", "")
                ),
                "unique_conversations": q.get(
                    "unique_conversation_count", q.get("count", 0)
                ),
            }
            cluster_entry["questions"].append(question_entry)

        # Sort questions by unique_conversations descending
        cluster_entry["questions"].sort(
            key=lambda x: x["unique_conversations"], reverse=True
        )
        output["clusters"].append(cluster_entry)

    # Sort clusters by total questions descending
    output["clusters"].sort(key=lambda x: len(x["questions"]), reverse=True)

    # Add summary stats
    total_questions = sum(len(c["questions"]) for c in output["clusters"])
    output["summary"] = {
        "total_clusters": len(output["clusters"]),
        "total_unique_questions": total_questions,
        "second_pass": second_pass,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")
    print(f"  Clusters: {len(output['clusters'])}")
    print(f"  Total questions: {total_questions}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full deduplication pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: within-cluster, presuppositions ON, second pass ON
  uv run python -m src.analysis.dedup.run_deduplication

  # Global mode (ignore clusters)
  uv run python -m src.analysis.dedup.run_deduplication --mode global

  # Disable presuppositions
  uv run python -m src.analysis.dedup.run_deduplication --no-presuppositions

  # Disable second pass (run pass 1 only)
  uv run python -m src.analysis.dedup.run_deduplication --no-second-pass

  # Custom thresholds
  uv run python -m src.analysis.dedup.run_deduplication \\
    --threshold 0.95 \\
    --second-pass-threshold 0.85
""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["within-cluster", "global"],
        default="within-cluster",
        help="Deduplication mode (default: within-cluster)",
    )
    parser.add_argument(
        "--with-presuppositions",
        action="store_true",
        default=True,
        help="Extract presuppositions during decomposition (default: ON)",
    )
    parser.add_argument(
        "--no-presuppositions",
        action="store_true",
        help="Disable presupposition extraction",
    )
    parser.add_argument(
        "--no-second-pass",
        action="store_true",
        help="Disable second pass on synthesized FAQs (default: ON)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for pass 1 (default: from config)",
    )
    parser.add_argument(
        "--second-pass-threshold",
        type=float,
        default=None,
        help="Similarity threshold for pass 2 (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generate timestamped)",
    )

    args = parser.parse_args()

    # Handle flags (both default ON, use --no-* to disable)
    with_presuppositions = not args.no_presuppositions
    second_pass = not args.no_second_pass

    result_dir = run(
        mode=args.mode,
        with_presuppositions=with_presuppositions,
        second_pass=second_pass,
        threshold=args.threshold,
        second_pass_threshold=args.second_pass_threshold,
        output_dir=args.output_dir,
    )
