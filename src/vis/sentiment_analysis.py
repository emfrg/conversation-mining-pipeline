"""Sentiment Analysis by Cluster Visualization.

Stacked horizontal bar chart showing distribution of user sentiment
across all clusters, highlighting negative sentiment.

Input:
    Pre-computed sentiment stats from cluster_stats.json (via compute_cluster_stats).

Output:
    PNG chart with sentiment breakdown per cluster.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_sentiment_chart(
    sentiment_stats: list[dict],
    output_path: Path | str,
    include_noise: bool = False,
    method: str = "kmeans",
    n_clusters: int = 0,
    total_items: int = 0,
    noise_count: int = 0,
) -> Path:
    """Create and save a stacked horizontal bar chart of sentiment.

    Args:
        sentiment_stats: List of dicts with title, negative, other, total.
            Pre-computed by compute_cluster_stats.
        output_path: Path to save the chart.
        include_noise: Whether to include noise cluster in chart (HDBSCAN only).
        method: Clustering method ("kmeans" or "hdbscan").
        n_clusters: Number of clusters.
        total_items: Total items clustered.
        noise_count: Number of noise points (HDBSCAN only).

    Returns:
        Path to the saved chart.
    """
    output_path = Path(output_path)

    # Filter out noise cluster if not included
    if not include_noise:
        sentiment_stats = [
            s for s in sentiment_stats if "Noise" not in s.get("title", "")
        ]

    # Stats are already sorted by total descending
    cluster_names = [item["title"] for item in sentiment_stats]
    other = [item["other"] for item in sentiment_stats]
    negative = [item["negative"] for item in sentiment_stats]

    # Truncate long cluster names for display
    display_names = [
        name[:40] + "..." if len(name) > 40 else name for name in cluster_names
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(cluster_names))
    bar_height = 0.7

    # Colors
    colors = {
        "other": "#cccccc",  # Gray
        "negative": "#e74c3c",  # Red
    }

    # Create stacked bars (other first, then negative on top)
    ax.barh(y_pos, other, bar_height, label="Neutral/Positive", color=colors["other"])
    ax.barh(
        y_pos,
        negative,
        bar_height,
        left=other,
        label="Negative",
        color=colors["negative"],
    )

    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()  # Largest at top
    ax.set_xlabel("Number of Issues")

    # Create title with clustering info
    if method == "kmeans":
        subtitle = f"K-means clustering (k={n_clusters}) • {total_items} total items"
    else:
        subtitle = (
            f"HDBSCAN clustering • {n_clusters} clusters • "
            f"{total_items} total items • {noise_count} noise"
        )
    ax.set_title(
        f"Sentiment Analysis by Cluster\n{subtitle}", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right")

    # Add total count labels at end of bars
    for i, item in enumerate(sentiment_stats):
        total = item["total"]
        ax.text(total + 2, i, str(total), va="center", fontsize=9, color="gray")

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
