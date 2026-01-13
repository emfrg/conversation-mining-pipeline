"""Correlation Heatmap Visualization.

Heatmap showing Pearson correlations between binary variables
with significance stars.

Input:
    Correlation data from correlation_analysis.py.

Output:
    PNG heatmap with significance annotations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Display labels for visualization
DISPLAY_LABELS = {
    "is_unresolved": "Unresolved",
    "used_unanswered_tool": "Used Unanswered Tool",
    "is_negative_sentiment": "Negative Sentiment",
}


def create_correlation_heatmap(
    correlations: dict,
    output_path: Path | str,
) -> Path:
    """Create and save correlation heatmap with significance stars.

    Args:
        correlations: Dict with columns, correlation_matrix, and pvalue_matrix.
        output_path: Path to save the PNG.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    display_labels = [DISPLAY_LABELS[c] for c in correlations["columns"]]
    corr_matrix = np.array(correlations["correlation_matrix"])
    pval_matrix = np.array(correlations["pvalue_matrix"])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        annot_kws={"size": 14},
        cbar_kws={"label": "Correlation Coefficient"},
    )

    # Add significance stars below the correlation values
    n = len(correlations["columns"])
    for i in range(n):
        for j in range(n):
            if i != j:
                pval = pval_matrix[i, j]
                if pval < 0.001:
                    star = "***"
                elif pval < 0.01:
                    star = "**"
                elif pval < 0.05:
                    star = "*"
                else:
                    star = ""
                if star:
                    ax.text(
                        j + 0.5,
                        i + 0.72,
                        star,
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="black",
                    )

    ax.set_title(
        "Correlation Matrix: Resolution, Tool Usage, and Sentiment\n"
        "* p<0.05, ** p<0.01, *** p<0.001",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
