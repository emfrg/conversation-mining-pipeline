"""FAQ Questions Chart Visualization.

Horizontal bar chart showing top FAQ questions per cluster,
color-coded by type (question vs presupposition).

Input:
    faq_summary.json - FAQ summary with questions and types.

Output:
    PNG file with the chart.
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def create_faq_questions_chart(
    summary_path: str,
    output_path: str,
    top_n: int = 5,
    show_presuppositions: bool = True,
) -> None:
    """Create horizontal bar chart of top questions per cluster.

    Args:
        summary_path: Path to faq_summary.json.
        output_path: Path to save the PNG output.
        top_n: Number of top questions to show per cluster.
        show_presuppositions: Whether to include presuppositions.
    """
    # Load data
    with open(summary_path) as f:
        data = json.load(f)

    clusters = data.get("clusters", [])
    if not clusters:
        print("    No clusters found in summary data")
        return

    # Filter and prepare data
    chart_data = []
    for cluster in clusters:
        cluster_name = cluster.get("cluster_name", "Unknown")
        questions = cluster.get("questions", [])

        # Filter by type if needed
        if not show_presuppositions:
            questions = [q for q in questions if q.get("type") != "presupposition"]

        # Get top N questions
        top_questions = questions[:top_n]
        if top_questions:
            chart_data.append(
                {
                    "cluster_name": cluster_name,
                    "questions": top_questions,
                }
            )

    if not chart_data:
        print("    No questions to visualize")
        return

    # Calculate figure size based on data
    n_clusters = len(chart_data)
    questions_per_cluster = [len(c["questions"]) for c in chart_data]
    total_bars = sum(questions_per_cluster)

    # Create figure with subplots
    fig_height = max(6, total_bars * 0.5 + n_clusters * 0.5)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(14, fig_height))

    # Handle single cluster case
    if n_clusters == 1:
        axes = [axes]

    # Colors for question types
    question_color = "#1f77b4"  # Blue
    presup_color = "#7f7f7f"  # Gray

    # Plot each cluster
    for _idx, (cluster_info, ax) in enumerate(zip(chart_data, axes, strict=False)):
        cluster_name = cluster_info["cluster_name"]
        questions = cluster_info["questions"]

        # Prepare bar data
        labels = []
        values = []
        colors = []

        for q in reversed(questions):  # Reverse so highest is at top
            question_text = q.get("question", "")
            q_type = q.get("type", "question")
            unique_convs = q.get("unique_conversations", 0)

            # Wrap long text
            wrapped = textwrap.fill(question_text, width=60)
            labels.append(wrapped)
            values.append(unique_convs)
            colors.append(
                presup_color if q_type == "presupposition" else question_color
            )

        # Create horizontal bar chart
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=colors)

        # Add value labels on bars
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=9,
            )

        # Set labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Unique Conversations")

        # Wrap cluster name for title
        wrapped_title = textwrap.fill(cluster_name, width=80)
        ax.set_title(wrapped_title, fontsize=11, fontweight="bold", loc="left")

        # Add some padding to x-axis
        max_val = max(values) if values else 1
        ax.set_xlim(0, max_val * 1.15)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=question_color, label="Question"),
        Patch(facecolor=presup_color, label="Presupposition"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )

    # Add title
    metadata = data.get("metadata", {})
    total_faqs = metadata.get("total_unique_faqs", 0)
    q_count = metadata.get("questions", 0)
    p_count = metadata.get("presuppositions", 0)

    fig.suptitle(
        f"Top FAQ Questions by Cluster\n({total_faqs} total: {q_count} questions + {p_count} presuppositions)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Adjust layout
    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Chart saved to {output_path}")
