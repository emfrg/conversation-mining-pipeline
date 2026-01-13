"""Resolution Status by Cluster Visualization.

Stacked horizontal bar chart showing distribution of resolution statuses
(resolved, partially_resolved, unresolved) across all clusters.

Input:
    Pre-computed resolution stats from cluster_stats.json (via compute_cluster_stats).

Output:
    PNG chart with resolution status breakdown per cluster.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_resolution_chart(
    resolution_stats: list[dict],
    output_path: Path | str,
    include_noise: bool = False,
    method: str = "kmeans",
    n_clusters: int = 0,
    total_items: int = 0,
    noise_count: int = 0,
) -> Path:
    """Create and save a stacked horizontal bar chart of resolution status.

    Args:
        resolution_stats: List of dicts with title, resolved, partially_resolved,
            unresolved, total. Pre-computed by compute_cluster_stats.
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
        resolution_stats = [
            s for s in resolution_stats if "Noise" not in s.get("title", "")
        ]

    # Stats are already sorted by total descending
    cluster_names = [item["title"] for item in resolution_stats]
    resolved = [item["resolved"] for item in resolution_stats]
    partial = [item["partially_resolved"] for item in resolution_stats]
    unresolved = [item["unresolved"] for item in resolution_stats]

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
        "resolved": "#2ecc71",  # Green
        "partial": "#f39c12",  # Orange/Yellow
        "unresolved": "#e74c3c",  # Red
    }

    # Create stacked bars
    ax.barh(y_pos, resolved, bar_height, label="Resolved", color=colors["resolved"])
    ax.barh(
        y_pos,
        partial,
        bar_height,
        left=resolved,
        label="Partially Resolved",
        color=colors["partial"],
    )
    ax.barh(
        y_pos,
        unresolved,
        bar_height,
        left=np.array(resolved) + np.array(partial),
        label="Unresolved",
        color=colors["unresolved"],
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
        f"Resolution Status by Cluster\n{subtitle}", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right")

    # Add total count labels at end of bars
    for i, item in enumerate(resolution_stats):
        total = item["total"]
        ax.text(total + 2, i, str(total), va="center", fontsize=9, color="gray")

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
