"""Chi-Square Proportions Visualization.

Bar charts showing proportions for chi-square analysis relationships.

Input:
    DataFrame with binary flags from chi_square_analysis.py.

Output:
    PNG with three-panel bar chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_proportion_charts(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Create bar charts showing proportions for each relationship.

    Three panels:
    1. Unanswered tool usage rate by resolution status
    2. Negative sentiment rate by resolution status
    3. Negative sentiment rate by tool usage

    Args:
        df: DataFrame with binary columns (is_unresolved, used_unanswered_tool,
            is_negative_sentiment).
        output_path: Path to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Tool usage by resolution status
    tool_by_resolution = df.groupby("is_unresolved")["used_unanswered_tool"].mean()
    ax1 = axes[0]
    x_labels = ["Resolved/Partial", "Unresolved"]
    values = [
        tool_by_resolution.get(False, 0),
        tool_by_resolution.get(True, 0),
    ]
    bars1 = ax1.bar(x_labels, values, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax1.set_ylabel("Proportion Using Unanswered Tool", fontsize=11)
    ax1.set_title(
        "Unanswered Tool Usage\nby Resolution Status", fontsize=12, fontweight="bold"
    )
    ax1.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 0.1)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Panel 2: Sentiment by resolution status
    sentiment_by_resolution = df.groupby("is_unresolved")[
        "is_negative_sentiment"
    ].mean()
    ax2 = axes[1]
    values2 = [
        sentiment_by_resolution.get(False, 0),
        sentiment_by_resolution.get(True, 0),
    ]
    bars2 = ax2.bar(x_labels, values2, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax2.set_ylabel("Proportion with Negative Sentiment", fontsize=11)
    ax2.set_title(
        "Negative Sentiment\nby Resolution Status", fontsize=12, fontweight="bold"
    )
    ax2.set_ylim(0, max(values2) * 1.3 if max(values2) > 0 else 0.1)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Panel 3: Sentiment by tool usage
    sentiment_by_tool = df.groupby("used_unanswered_tool")[
        "is_negative_sentiment"
    ].mean()
    ax3 = axes[2]
    x_labels3 = ["No Unanswered Tool", "Used Unanswered Tool"]
    values3 = [
        sentiment_by_tool.get(False, 0),
        sentiment_by_tool.get(True, 0),
    ]
    bars3 = ax3.bar(x_labels3, values3, color=["#3498db", "#f39c12"], edgecolor="black")
    ax3.set_ylabel("Proportion with Negative Sentiment", fontsize=11)
    ax3.set_title(
        "Negative Sentiment\nby Unanswered Tool Usage", fontsize=12, fontweight="bold"
    )
    ax3.set_ylim(0, max(values3) * 1.3 if max(values3) > 0 else 0.1)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.suptitle(
        "Chi-Square Analysis: Proportional Breakdowns",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
