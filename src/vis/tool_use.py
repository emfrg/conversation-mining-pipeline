"""Tool Usage by Cluster Visualization.

Stacked horizontal bar chart showing distribution of tool usage
across all clusters, with weighted contribution per conversation.

Input:
    Pre-computed tool use stats from cluster_stats.json (via compute_cluster_stats).

Output:
    PNG chart with tool usage breakdown per cluster.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# CUSTOMIZE FOR YOUR CHATBOT
# =============================================================================
# These tool names are specific to your chatbot implementation.
# Edit these dictionaries to match your chatbot's tool names.
# The keys should match the tool names in your chat_history.json data.
# =============================================================================

# Tool colors (key = tool name from your data, value = hex color)
TOOL_COLORS = {
    "no_tool": "#cccccc",  # Gray - keep this for conversations with no tool use
    "rag_tool": "#3498db",  # Blue
    "zone_checker": "#2ecc71",  # Green
    "unanswered_question_tool": "#f39c12",  # Orange
    "feedback_tool": "#9b59b6",  # Purple
}

# Tool display labels (key = tool name, value = human-readable label for charts)
TOOL_LABELS = {
    "no_tool": "No Tool",
    "rag_tool": "RAG Tool",
    "zone_checker": "Zone Checker",
    "unanswered_question_tool": "Unanswered Question",
    "feedback_tool": "Feedback Tool",
}

# Order for stacking in charts (no_tool first as gray base)
TOOL_ORDER = [
    "no_tool",
    "rag_tool",
    "zone_checker",
    "unanswered_question_tool",
    "feedback_tool",
]


def create_tool_use_chart(
    tool_use_stats: list[dict],
    output_path: Path | str,
    include_noise: bool = False,
    method: str = "kmeans",
    n_clusters: int = 0,
    total_items: int = 0,
    noise_count: int = 0,
) -> Path:
    """Create and save a stacked horizontal bar chart of tool usage.

    Args:
        tool_use_stats: List of dicts with title, no_tool, rag_tool, zone_checker,
            unanswered_question_tool, feedback_tool, total.
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
        tool_use_stats = [
            s for s in tool_use_stats if "Noise" not in s.get("title", "")
        ]

    # Stats are already sorted by total descending
    cluster_names = [item["title"] for item in tool_use_stats]

    # Truncate long cluster names for display
    display_names = [
        name[:40] + "..." if len(name) > 40 else name for name in cluster_names
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    y_pos = np.arange(len(cluster_names))
    bar_height = 0.7

    # Stack bars in order
    left = np.zeros(len(cluster_names))

    for tool in TOOL_ORDER:
        values = [item[tool] for item in tool_use_stats]
        ax.barh(
            y_pos,
            values,
            bar_height,
            left=left,
            label=TOOL_LABELS[tool],
            color=TOOL_COLORS[tool],
        )
        left += np.array(values)

    # Customize chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()  # Largest at top
    ax.set_xlabel("Number of Conversations (tool usage weighted)")

    # Create title with clustering info
    if method == "kmeans":
        subtitle = f"K-means clustering (k={n_clusters}) • {total_items} total items"
    else:
        subtitle = (
            f"HDBSCAN clustering • {n_clusters} clusters • "
            f"{total_items} total items • {noise_count} noise"
        )
    ax.set_title(f"Tool Usage by Cluster\n{subtitle}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")

    # Add total count labels at end of bars
    for i, item in enumerate(tool_use_stats):
        total = item["total"]
        ax.text(total + 2, i, str(int(total)), va="center", fontsize=9, color="gray")

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
