#!/usr/bin/env python3
"""
FAQ Analytics Pipeline Orchestrator

Runs the complete pipeline to analyze FAQ conversations:
1. clean_data            - Parse raw chat data into clean conversations
2. extract_issues        - Extract issue summaries using LLM
3. build_embeddings      - Generate embeddings and store in Qdrant
4. reduce_dimensions     - Apply PCA dimensionality reduction
5. cluster_embeddings    - Cluster the reduced embeddings
6. name_clusters         - Name clusters using LLM
7. top_issues            - Rank issues by proximity to cluster centroid
8. compute_cluster_stats - Pre-compute cluster statistics for visualizations
9. visualize_clusters    - Create visualizations
10. statistical_analysis - Compute usage stats, correlations, chi-square (skips in snippet mode)
11. deduplicate_faqs     - Deduplicate FAQ questions and synthesize clean FAQs

Usage:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --steps clean_data extract_issues
    python run_pipeline.py --from build_embeddings
    python run_pipeline.py --dry-run
    python run_pipeline.py --list
    python run_pipeline.py --snippet 100      # Run on 100 random conversations
    python run_pipeline.py --enrich-only      # Add sentiment/resolution/steps (parallel, fast)
    python run_pipeline.py --tag-only         # Add tags with pooling (sequential)
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Step names for validation (before importing modules)
STEP_NAMES = [
    "clean_data",
    "extract_issues",
    "build_embeddings",
    "reduce_dimensions",
    "cluster_embeddings",
    "name_clusters",
    "top_issues",
    "compute_cluster_stats",
    "visualize_clusters",
    "statistical_analysis",
    "deduplicate_faqs",
]

STEP_DESCRIPTIONS = {
    "clean_data": "Parse raw chat data into clean conversations",
    "extract_issues": "Extract issue summaries using LLM",
    "build_embeddings": "Generate embeddings and store in Qdrant",
    "reduce_dimensions": "Apply PCA dimensionality reduction",
    "cluster_embeddings": "Cluster the reduced embeddings",
    "name_clusters": "Name clusters using LLM",
    "top_issues": "Rank issues by proximity to cluster centroid",
    "compute_cluster_stats": "Pre-compute cluster statistics for visualizations",
    "visualize_clusters": "Create visualizations",
    "statistical_analysis": "Compute usage stats, correlations, chi-square (skips in snippet)",
    "deduplicate_faqs": "Deduplicate FAQ questions and synthesize clean FAQs",
}


@dataclass
class PipelineStep:
    """Definition of a pipeline step."""

    name: str
    description: str
    run: Callable[[], Any]


def get_pipeline_steps() -> list[PipelineStep]:
    """
    Import and return pipeline steps.

    This function delays importing pipeline modules until after config
    has been set up (including --snippet override).
    """
    from src.analysis import top_issues
    from src.pipeline import (
        build_embeddings,
        clean_data,
        cluster_embeddings,
        compute_cluster_stats,
        deduplicate_faqs,
        extract_issues,
        name_clusters,
        reduce_dimensions,
        statistical_analysis,
        visualize_clusters,
    )

    return [
        PipelineStep(
            "clean_data",
            "Parse raw chat data into clean conversations",
            clean_data.run,
        ),
        PipelineStep(
            "extract_issues",
            "Extract issue summaries using LLM",
            extract_issues.run,
        ),
        PipelineStep(
            "build_embeddings",
            "Generate embeddings and store in Qdrant",
            build_embeddings.run,
        ),
        PipelineStep(
            "reduce_dimensions",
            "Apply PCA dimensionality reduction",
            reduce_dimensions.run,
        ),
        PipelineStep(
            "cluster_embeddings",
            "Cluster the reduced embeddings",
            cluster_embeddings.run,
        ),
        PipelineStep(
            "name_clusters",
            "Name clusters using LLM",
            name_clusters.run,
        ),
        PipelineStep(
            "top_issues",
            "Rank issues by proximity to cluster centroid",
            top_issues.run,
        ),
        PipelineStep(
            "compute_cluster_stats",
            "Pre-compute cluster statistics for visualizations",
            compute_cluster_stats.run,
        ),
        PipelineStep(
            "visualize_clusters",
            "Create visualizations",
            visualize_clusters.run,
        ),
        PipelineStep(
            "statistical_analysis",
            "Compute usage stats, correlations, chi-square (skips in snippet)",
            statistical_analysis.run,
        ),
        PipelineStep(
            "deduplicate_faqs",
            "Deduplicate FAQ questions and synthesize clean FAQs",
            deduplicate_faqs.run,
        ),
    ]


def run_pipeline(steps: list[str], dry_run: bool = False) -> None:
    """Run the specified pipeline steps."""
    import config as config_module
    from src.utils.experiment_manager import (
        find_matching_experiment,
        setup_experiment,
        setup_experiment_from_existing,
        setup_experiment_from_latest,
        update_latest_symlink,
    )

    print("\n" + "=" * 60)
    print("FAQ ANALYTICS PIPELINE")
    print("=" * 60)

    # Validate steps
    for step_name in steps:
        if step_name not in STEP_NAMES:
            print(f"\nError: Unknown step '{step_name}'")
            print(f"Valid steps: {', '.join(STEP_NAMES)}")
            sys.exit(1)

    # Get current config
    current_config = config_module.config

    # Show configuration summary
    print("\nConfiguration:")
    print(f"  Data: {current_config['paths']['raw_data']}")
    print(
        f"  Qdrant: {current_config['qdrant']['host']}:{current_config['qdrant']['port']}"
    )
    print(f"  Clustering: {current_config['clustering']['method']}")
    print(f"  LLM: {current_config['llm']['model_name']}")

    # Show snippet mode if enabled
    num_conv = current_config.get("snippet", {}).get("num_conversations")
    if num_conv:
        print(f"  Mode: SNIPPET ({num_conv} conversations)")
    else:
        print("  Mode: FULL DATASET")

    print(f"\nSteps to run ({len(steps)}):")
    for i, step_name in enumerate(steps, 1):
        print(f"  {i}. {step_name}: {STEP_DESCRIPTIONS[step_name]}")

    if dry_run:
        print("\n[DRY RUN] No steps executed.")
        return

    print("\n" + "-" * 60)

    # Set up experiment directory if running clustering steps
    clustering_steps = {
        "cluster_embeddings",
        "name_clusters",
        "top_issues",
        "compute_cluster_stats",
        "visualize_clusters",
        "deduplicate_faqs",
    }
    steps_set = set(steps)
    will_run_clustering = bool(clustering_steps & steps_set)

    # Determine if we should create new experiment or reuse existing
    creates_new_experiment = {"cluster_embeddings", "name_clusters"}
    only_visualize = steps_set == {"visualize_clusters"}

    if will_run_clustering:
        print("\n[Experiment Setup]")
        recreate = current_config.get("recreate", False)

        if only_visualize:
            # Only visualize_clusters - reuse existing experiment
            experiment_id = setup_experiment_from_latest(current_config)
            if experiment_id is None:
                print("  ERROR: No existing experiment found (no _latest symlink)")
                print("  Run cluster_embeddings first, or run full pipeline.")
                sys.exit(1)
        elif not (creates_new_experiment & steps_set):
            # Only downstream steps (no cluster_embeddings/name_clusters)
            experiment_id = setup_experiment_from_latest(current_config)
            if experiment_id is None:
                print("  ERROR: No existing experiment found (no _latest symlink)")
                print("  Run cluster_embeddings first, or run full pipeline.")
                sys.exit(1)
        elif recreate:
            # Force new experiment when --recreate is passed
            setup_experiment(current_config)
        else:
            # Try to find matching experiment, create new if none found
            existing = find_matching_experiment(current_config)
            if existing:
                setup_experiment_from_existing(current_config, existing)
            else:
                setup_experiment(current_config)

    # Import pipeline modules NOW (after config is set up)
    pipeline_steps = get_pipeline_steps()

    # Execute steps
    total_start = time.time()
    for i, step_name in enumerate(steps, 1):
        step = next(s for s in pipeline_steps if s.name == step_name)

        # Skip top_issues for HDBSCAN (centroid-based ranking not meaningful for density-based clustering)
        if (
            step.name == "top_issues"
            and current_config["clustering"]["method"] == "hdbscan"
        ):
            print(f"\n[{i}/{len(steps)}] {step.name}")
            print(f"  {step.description}")
            print(
                "  Skipped: Centroid-based ranking not applicable for HDBSCAN (density-based clustering)"
            )
            continue

        print(f"\n[{i}/{len(steps)}] {step.name}")
        print(f"  {step.description}")

        step_start = time.time()
        try:
            step.run()
            elapsed = time.time() - step_start
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print(f"\nPipeline failed at step '{step.name}'")
            sys.exit(1)

    # Update latest symlink after successful completion of clustering steps
    if will_run_clustering and "visualize_clusters" in steps:
        update_latest_symlink(current_config)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE ({total_elapsed:.1f}s)")
    print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FAQ Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run full pipeline
  python run_pipeline.py --steps clean_data extract_issues
  python run_pipeline.py --from build_embeddings
  python run_pipeline.py --dry-run
  python run_pipeline.py --list
  python run_pipeline.py --snippet 100      # Run on 100 conversations
  python run_pipeline.py --enrich-only      # Add sentiment/resolution/steps (parallel)
  python run_pipeline.py --tag-only         # Add tags with pooling (sequential)
        """,
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        help="Specific steps to run",
    )

    parser.add_argument(
        "--from",
        dest="from_step",
        help="Run from this step onwards",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available steps",
    )

    parser.add_argument(
        "--snippet",
        type=int,
        metavar="N",
        help="Run on a snippet of N random conversations (overrides config)",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Force recreation of all outputs (ignore existing)",
    )

    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Add sentiment/resolution/steps to existing reports (parallel, fast)",
    )

    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Add tags to existing reports with pooling (sequential, consistent)",
    )

    parser.add_argument(
        "--skip-deduplication",
        action="store_true",
        help="Skip the deduplicate_faqs step (useful for faster iteration)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle --snippet override BEFORE importing any pipeline modules
    if args.snippet:
        import config as config_module

        config_module.config = config_module.get_config_with_snippet(args.snippet)

    # Handle --recreate flag
    import config as config_module

    config_module.config["recreate"] = args.recreate

    if args.list:
        print("\nAvailable pipeline steps:")
        for i, name in enumerate(STEP_NAMES, 1):
            print(f"  {i}. {name}: {STEP_DESCRIPTIONS[name]}")
        print()
        return

    # Handle --enrich-only flag
    if args.enrich_only:
        from src.pipeline import extract_issues

        print("\n" + "=" * 60)
        print("ENRICH ONLY MODE (parallel)")
        print("=" * 60)
        print("\nAdding sentiment/resolution/steps to existing reports...")

        extract_issues.enrich_only()

        print("\n" + "-" * 60)
        print("Regenerating stats and visualizations...")
        print("-" * 60)

        # Auto-run stats and visualization with recreate to reflect new fields
        config_module.config["recreate"] = True
        run_pipeline(["compute_cluster_stats", "visualize_clusters"])

        print("\n" + "=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60 + "\n")
        return

    # Handle --tag-only flag
    if args.tag_only:
        from src.pipeline import extract_issues

        print("\n" + "=" * 60)
        print("TAG ONLY MODE (sequential with pooling)")
        print("=" * 60)
        print("\nAdding tags to existing reports...")

        extract_issues.tag_only()

        print("\n" + "-" * 60)
        print("Regenerating stats and visualizations...")
        print("-" * 60)

        # Auto-run stats and visualization with recreate to reflect new fields
        config_module.config["recreate"] = True
        run_pipeline(["compute_cluster_stats", "visualize_clusters"])

        print("\n" + "=" * 60)
        print("TAGGING COMPLETE")
        print("=" * 60 + "\n")
        return

    # Determine which steps to run
    if args.steps:
        steps = args.steps
    elif args.from_step:
        if args.from_step not in STEP_NAMES:
            print(f"Error: Unknown step '{args.from_step}'")
            sys.exit(1)
        start_idx = STEP_NAMES.index(args.from_step)
        steps = STEP_NAMES[start_idx:]
    else:
        steps = STEP_NAMES  # Run all

    # Handle --skip-deduplication flag
    if args.skip_deduplication and "deduplicate_faqs" in steps:
        steps = [s for s in steps if s != "deduplicate_faqs"]
        print("Note: Skipping deduplicate_faqs step (--skip-deduplication)")

    run_pipeline(steps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
