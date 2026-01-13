"""Step 9: Visualize Clusters - Create cluster visualizations.

Creates visualizations for cluster analysis:
1. Horizontal bar chart showing cluster counts
2. 2D scatter plot with UMAP dimensionality reduction
3. Resolution status by cluster (from pre-computed stats)
4. Sentiment analysis by cluster (from pre-computed stats)
5. Tag word cloud (requires extra agent)
6. Tool usage by cluster (from pre-computed stats)

Reads:
    config.experiment.dir/clusters_named.json - Named clusters from Step 6.
    config.experiment.dir/cluster_stats.json - Pre-computed stats from Step 8.
    config.experiment.dir/labels.json - Cluster assignments from Step 5.
    config.paths.issue_reports - Issue reports (for tag wordcloud only).
    Qdrant collection (config.qdrant.reduced_collection) - PCA-reduced embeddings.

Writes:
    config.experiment.dir/vis/ - All visualization outputs.
"""

import os
from typing import Any

from qdrant_client import QdrantClient

from config import config
from src.analysis.umap_reduction import reduce_to_2d
from src.utils.helpers import (
    fetch_embeddings_from_qdrant,
    filter_by_confidence,
    read_json_file,
)
from src.vis import (
    create_cluster_bar_chart,
    create_cluster_scatter_plot,
    create_resolution_chart,
    create_sentiment_chart,
    create_tag_wordcloud,
    create_tool_use_chart,
)


def has_extra_agent_fields(issue_reports: dict) -> dict[str, bool]:
    """Check which extra agent fields exist in issue reports.

    Args:
        issue_reports: Parsed issue_reports.json data.

    Returns:
        Dict with boolean flags for each field type.
    """
    sample_issue: dict[str, Any] = next(iter(issue_reports.values()), {})
    return {
        "tags": bool("tags" in sample_issue and sample_issue.get("tags")),
    }


def run() -> str:
    """Run the visualize_clusters pipeline step.

    Creates bar chart and scatter plot visualizations for clusters.
    Optionally creates additional visualizations if extra agent fields exist.

    For HDBSCAN with noise cluster, generates two versions of each chart:
    - Without noise (default)
    - With noise (*_with_noise.png)

    Returns:
        Path to the visualization output directory.
    """
    method = config["clustering"]["method"]
    experiment_dir = config["experiment"]["dir"]
    vis_dir = os.path.join(experiment_dir, "vis")
    recreate = config.get("recreate", False)

    # Skip if visualizations already exist (unless --recreate)
    bar_chart_path = os.path.join(vis_dir, f"faq_clusters_{method}.png")
    if not recreate and os.path.exists(bar_chart_path):
        print(f"  Skipping: Visualizations already exist in {vis_dir}")
        return vis_dir

    os.makedirs(vis_dir, exist_ok=True)

    input_path = os.path.join(experiment_dir, "clusters_named.json")
    labels_path = os.path.join(experiment_dir, "labels.json")

    # Get chart config
    chart_config = config.get("visualization", {}).get("charts", {})

    # Load cluster data
    print(f"  Loading clusters from {input_path}...")
    cluster_data = read_json_file(input_path)

    # Check if HDBSCAN with noise cluster
    is_hdbscan = method == "hdbscan"
    has_noise = "noise" in cluster_data.get("clusters", {})

    # Extract clustering metadata for chart titles
    n_clusters = cluster_data.get("n_clusters", 0)
    total_items = cluster_data.get("total_items", 0)
    noise_count = cluster_data["clusters"].get("noise", {}).get("count", 0)

    # === Core visualizations (always available) ===

    # 1. Bar chart
    if chart_config.get("bar_chart", True):
        print("  Creating bar chart...")
        # Default version (without noise)
        bar_chart = create_cluster_bar_chart(cluster_data, include_noise=False)
        bar_output_path = os.path.join(vis_dir, f"faq_clusters_{method}.png")
        bar_chart.save(bar_output_path, ppi=200)
        print(f"    Saved: {bar_output_path}")

        # Additional version with noise for HDBSCAN
        if is_hdbscan and has_noise:
            bar_chart_noise = create_cluster_bar_chart(cluster_data, include_noise=True)
            bar_noise_path = os.path.join(
                vis_dir, f"faq_clusters_{method}_with_noise.png"
            )
            bar_chart_noise.save(bar_noise_path, ppi=200)
            print(f"    Saved: {bar_noise_path}")

    # 2. Scatter plot (requires UMAP reduction)
    if chart_config.get("scatter_plot", True):
        print(f"  Loading labels from {labels_path}...")
        labels = read_json_file(labels_path)

        print("  Fetching embeddings from Qdrant...")
        client = QdrantClient(
            host=config["qdrant"]["host"],
            port=config["qdrant"]["port"],
        )
        collection_name = config["qdrant"]["reduced_collection"]
        ids, embeddings, _ = fetch_embeddings_from_qdrant(client, collection_name)

        print("  Reducing to 2D with UMAP...")
        embeddings_2d = reduce_to_2d(embeddings)

        print("  Creating scatter plot...")
        scatter_chart = create_cluster_scatter_plot(
            embeddings_2d, ids, labels, cluster_data
        )
        scatter_output_path = os.path.join(
            vis_dir, f"faq_clusters_scatter_{method}.png"
        )
        scatter_chart.save(scatter_output_path, ppi=200)
        print(f"    Saved: {scatter_output_path}")

    # === Extra visualizations (from pre-computed stats) ===

    # Load pre-computed cluster stats from Step 8
    cluster_stats_path = os.path.join(experiment_dir, "cluster_stats.json")
    cluster_stats = read_json_file(cluster_stats_path)

    # 3. Resolution status chart
    if chart_config.get("resolution_status", True):
        resolution_stats = cluster_stats.get("resolution_by_cluster", [])
        if resolution_stats:
            print("  Creating resolution status chart...")
            # Default version (without noise)
            resolution_path = os.path.join(vis_dir, "resolution_status.png")
            create_resolution_chart(
                resolution_stats,
                resolution_path,
                include_noise=False,
                method=method,
                n_clusters=n_clusters,
                total_items=total_items,
                noise_count=noise_count,
            )
            print(f"    Saved: {resolution_path}")

            # Additional version with noise for HDBSCAN
            if is_hdbscan and has_noise:
                resolution_noise_path = os.path.join(
                    vis_dir, "resolution_status_with_noise.png"
                )
                create_resolution_chart(
                    resolution_stats,
                    resolution_noise_path,
                    include_noise=True,
                    method=method,
                    n_clusters=n_clusters,
                    total_items=total_items,
                    noise_count=noise_count,
                )
                print(f"    Saved: {resolution_noise_path}")
        else:
            print("  Skipping resolution status (no pre-computed stats)")

    # 4. Sentiment analysis chart
    if chart_config.get("sentiment_analysis", True):
        sentiment_stats = cluster_stats.get("sentiment_by_cluster", [])
        if sentiment_stats:
            print("  Creating sentiment analysis chart...")
            # Default version (without noise)
            sentiment_path = os.path.join(vis_dir, "sentiment_analysis.png")
            create_sentiment_chart(
                sentiment_stats,
                sentiment_path,
                include_noise=False,
                method=method,
                n_clusters=n_clusters,
                total_items=total_items,
                noise_count=noise_count,
            )
            print(f"    Saved: {sentiment_path}")

            # Additional version with noise for HDBSCAN
            if is_hdbscan and has_noise:
                sentiment_noise_path = os.path.join(
                    vis_dir, "sentiment_analysis_with_noise.png"
                )
                create_sentiment_chart(
                    sentiment_stats,
                    sentiment_noise_path,
                    include_noise=True,
                    method=method,
                    n_clusters=n_clusters,
                    total_items=total_items,
                    noise_count=noise_count,
                )
                print(f"    Saved: {sentiment_noise_path}")
        else:
            print("  Skipping sentiment analysis (no pre-computed stats)")

    # 5. Tag word cloud (still needs issue_reports for tags)
    if chart_config.get("tag_wordcloud", True):
        issue_reports_path = config["paths"]["issue_reports"]
        issue_reports = read_json_file(issue_reports_path)

        # Apply confidence filtering if configured
        min_confidence = config.get("filtering", {}).get("min_confidence")
        if min_confidence:
            issue_reports = filter_by_confidence(issue_reports, min_confidence)

        extra_fields = has_extra_agent_fields(issue_reports)

        if extra_fields["tags"]:
            print("  Creating tag word cloud...")
            wordcloud_path = os.path.join(vis_dir, "tag_wordcloud.png")
            result = create_tag_wordcloud(issue_reports, wordcloud_path)
            if result:
                print(f"    Saved: {wordcloud_path}")
            else:
                print("    Skipped (no tags found)")
        else:
            print("  Skipping tag word cloud (field not available)")

    # 6. Tool use chart
    if chart_config.get("tool_use", True):
        tool_use_stats = cluster_stats.get("tool_use_by_cluster", [])
        if tool_use_stats:
            print("  Creating tool use chart...")
            # Default version (without noise)
            tool_use_path = os.path.join(vis_dir, "tool_use.png")
            create_tool_use_chart(
                tool_use_stats,
                tool_use_path,
                include_noise=False,
                method=method,
                n_clusters=n_clusters,
                total_items=total_items,
                noise_count=noise_count,
            )
            print(f"    Saved: {tool_use_path}")

            # Additional version with noise for HDBSCAN
            if is_hdbscan and has_noise:
                tool_use_noise_path = os.path.join(vis_dir, "tool_use_with_noise.png")
                create_tool_use_chart(
                    tool_use_stats,
                    tool_use_noise_path,
                    include_noise=True,
                    method=method,
                    n_clusters=n_clusters,
                    total_items=total_items,
                    noise_count=noise_count,
                )
                print(f"    Saved: {tool_use_noise_path}")
        else:
            print("  Skipping tool use (no pre-computed stats)")

    print(f"  All visualizations saved to {vis_dir}/")
    return vis_dir


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment

    # Set up experiment directory for standalone execution
    setup_experiment(config)
    run()
