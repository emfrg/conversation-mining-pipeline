"""Step 6: Name Clusters - Generate human-readable cluster names using LLM.

Uses LLM agent to generate meaningful names and descriptions for each cluster
based on the issues it contains. Processes clusters sequentially so each
cluster can see previously named clusters to avoid duplicate/similar titles.

Features:
- Sequential processing for consistent naming
- Resume support: skips already-named clusters
- Intermediate saves after each cluster

Reads:
    config.experiment.dir/clusters_readable.json - Cluster contents from Step 5.

Writes:
    config.experiment.dir/clusters_named.json - Named clusters with descriptions.
"""

import asyncio
import json
import os

from dotenv import load_dotenv

from config import config
from src.agents.cluster_naming_agent import cluster_namer
from src.schemas import ClusterName
from src.utils.helpers import format_cluster_context, read_json_file

load_dotenv()


def format_cluster_issues(items: list[dict]) -> str:
    """Format cluster items as text for the LLM.

    Args:
        items: List of issue dicts with issue_title and canonical_faq_question.

    Returns:
        Formatted text string for LLM input.
    """
    lines = []
    for item in items:
        lines.append(f"- Title: {item['issue_title']}")
        lines.append(f"  FAQ: {item['canonical_faq_question']}")
    return "\n".join(lines)


async def async_name_single_cluster(
    cluster_key: str,
    cluster_data: dict,
    max_retries: int,
    named_clusters: dict[str, dict],
) -> tuple[str, dict]:
    """Call agent to name a single cluster with context of prior clusters.

    Args:
        cluster_key: Cluster identifier (e.g., "cluster_0").
        cluster_data: Dict with count and items for this cluster.
        max_retries: Maximum retries for LLM failures.
        named_clusters: Already-named clusters for context.

    Returns:
        Tuple of (cluster_key, result dict with title and descriptions).
    """
    issues_text = format_cluster_issues(cluster_data["items"])
    already_named_clusters = format_cluster_context(named_clusters)

    for attempt in range(max_retries):
        try:
            response: ClusterName = await cluster_namer.ainvoke(
                {
                    "already_named_clusters": already_named_clusters,
                    "issues_text": issues_text,
                }
            )

            result = {
                "cluster_title": response.cluster_title,
                "cluster_description": response.cluster_description,
                "cluster_detailed_description": response.cluster_detailed_description,
                "count": cluster_data["count"],
                "items": cluster_data["items"],
            }
            return (cluster_key, result)

        except Exception as e:
            print(f"  {cluster_key}: Retry {attempt + 1}/{max_retries} - Error: {e}")

    # Return original data if all retries fail
    print(f"  {cluster_key}: Failed after {max_retries} retries, keeping original")
    return (cluster_key, cluster_data)


async def process_clusters_sequential(
    clusters: list[tuple[str, dict]],
    max_retries: int,
    named_clusters: dict[str, dict],
    output_path: str,
    full_output_data: dict,
) -> dict[str, dict]:
    """Process clusters sequentially with intermediate saving.

    Args:
        clusters: List of (cluster_key, cluster_data) tuples to name.
        max_retries: Maximum retries per cluster.
        named_clusters: Dict of already-named clusters (modified in place).
        output_path: Path to save intermediate results.
        full_output_data: Full output dict structure for saving.

    Returns:
        Updated named_clusters dict.
    """
    for i, (cluster_key, cluster_data) in enumerate(clusters):
        print(f"  Processing cluster {i + 1}/{len(clusters)}: {cluster_key}")

        _, result = await async_name_single_cluster(
            cluster_key, cluster_data, max_retries, named_clusters
        )

        named_clusters[cluster_key] = result

        if "cluster_title" in result:
            print(f"    -> {result['cluster_title']}")

        # Save intermediate results after each cluster
        full_output_data["clusters"] = named_clusters
        with open(output_path, "w") as f:
            json.dump(full_output_data, f, indent=2)

    return named_clusters


def run() -> int:
    """Run the name_clusters pipeline step.

    Names clusters using LLM with sequential processing and resumption.

    Returns:
        Number of clusters named in this run.
    """
    max_retries = config["llm"]["max_retries"]
    force = config.get("recreate", False)

    experiment_dir = config["experiment"]["dir"]
    input_path = os.path.join(experiment_dir, "clusters_readable.json")
    output_path = os.path.join(experiment_dir, "clusters_named.json")

    data = read_json_file(input_path)

    # Load existing results if output file exists (for resumption)
    if not force and os.path.exists(output_path):
        existing_output = read_json_file(output_path)
        named_clusters = existing_output.get("clusters", {})
        print(f"  Resuming: Found {len(named_clusters)} existing named clusters")

        # Remove stale clusters that don't exist in current input
        current_keys = set(data["clusters"].keys())
        stale_keys = [k for k in named_clusters if k not in current_keys]
        for key in stale_keys:
            del named_clusters[key]
            print(f"  Removed stale cluster: {key}")
    else:
        if force:
            print("  Force mode: Re-naming all clusters")
        named_clusters = {}

    # Separate noise cluster and identify clusters that need naming
    clusters_to_name = []

    for cluster_key, cluster_data in data["clusters"].items():
        if cluster_key == "noise":
            # Name the noise cluster
            named_clusters[cluster_key] = {
                "cluster_title": "Uncategorized Issues (Noise cluster)",
                "cluster_description": "Issues that don't fit into any identified cluster pattern",
                "cluster_detailed_description": "Issues that don't fit into any identified cluster pattern",
                "count": cluster_data["count"],
                "items": cluster_data["items"],
            }
            print(f"  Named noise cluster ({cluster_data['count']} items)")
        elif (
            cluster_key in named_clusters
            and "cluster_title" in named_clusters[cluster_key]
        ):
            # Skip already-named clusters (for resumption)
            print(f"  Skipping already-named: {cluster_key}")
        else:
            clusters_to_name.append((cluster_key, cluster_data))

    if not clusters_to_name:
        print("  No clusters to name")
        return 0

    print(f"  Naming {len(clusters_to_name)} clusters (sequential mode)")

    # Prepare output structure for intermediate saving
    output = {
        "method": data["method"],
        "total_items": data["total_items"],
        "n_clusters": data["n_clusters"],
        "clusters": named_clusters,
    }

    # Process all clusters sequentially
    asyncio.run(
        process_clusters_sequential(
            clusters_to_name,
            max_retries,
            named_clusters,
            output_path,
            output,
        )
    )

    print(f"  Saved to {output_path}")

    return len(clusters_to_name)


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment

    # Set up experiment directory for standalone execution
    setup_experiment(config)
    run()
