"""Correlation Analysis for FAQ Analytics.

Computes Pearson/point-biserial correlations between:
- Unresolved status (resolution_status == "unresolved")
- Unanswered question tool usage
- Negative sentiment (user_sentiment == "negative")

For binary variables, Pearson correlation equals point-biserial correlation.

Reads:
    config.paths.issue_reports - Resolution status and sentiment.
    config.paths.clean_conversations - Tool usage data.

Writes:
    src/vis/outputs/correlation_analysis.json - Correlation matrix with p-values.
    src/vis/outputs/correlation_heatmap.png - Visualization.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import config
from src.utils.helpers import filter_by_confidence, read_json_file
from src.vis.correlation_heatmap import create_correlation_heatmap

# Column names for analysis
COLUMNS = ["is_unresolved", "used_unanswered_tool", "is_negative_sentiment"]

# Display labels for visualization
DISPLAY_LABELS = {
    "is_unresolved": "Unresolved",
    "used_unanswered_tool": "Used Unanswered Tool",
    "is_negative_sentiment": "Negative Sentiment",
}


def create_analysis_dataframe(
    issue_reports: dict,
    clean_conversations: dict,
) -> pd.DataFrame:
    """Join issue reports with conversations and create binary flags.

    Args:
        issue_reports: Dict of issue reports keyed by conversation_id.
        clean_conversations: Dict of clean conversations keyed by conversation_id.

    Returns:
        DataFrame with columns:
        - conversation_id: str
        - is_unresolved: bool (resolution_status == "unresolved")
        - used_unanswered_tool: bool (any turn has tool=="unanswered_question_tool")
        - is_negative_sentiment: bool (user_sentiment == "negative")
    """
    rows = []
    for conv_id, issue in issue_reports.items():
        # Get resolution status
        resolution = issue.get("resolution_status", "")
        is_unresolved = resolution == "unresolved"

        # Get sentiment
        sentiment = issue.get("user_sentiment", "")
        is_negative = sentiment == "negative"

        # Check for unanswered tool usage (configured in config.yaml tools.unanswered_tool)
        unanswered_tool = config.get("tools", {}).get("unanswered_tool")
        used_unanswered = False
        if unanswered_tool and conv_id in clean_conversations:
            for turn in clean_conversations[conv_id].get("turns", []):
                if turn.get("type") == "tool" and turn.get("tool") == unanswered_tool:
                    used_unanswered = True
                    break

        rows.append(
            {
                "conversation_id": conv_id,
                "is_unresolved": is_unresolved,
                "used_unanswered_tool": used_unanswered,
                "is_negative_sentiment": is_negative,
            }
        )

    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute pairwise Pearson correlations with p-values.

    Args:
        df: DataFrame with binary columns.

    Returns:
        Dict with correlation matrix, p-value matrix, and summary.
    """
    n = len(COLUMNS)

    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))

    for i, col1 in enumerate(COLUMNS):
        for j, col2 in enumerate(COLUMNS):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                corr, pval = stats.pearsonr(df[col1].astype(int), df[col2].astype(int))
                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

    return {
        "columns": COLUMNS,
        "correlation_matrix": corr_matrix.tolist(),
        "pvalue_matrix": pval_matrix.tolist(),
    }


def interpret_correlation(corr: float, pval: float) -> str:
    """Generate human-readable interpretation of correlation.

    Args:
        corr: Correlation coefficient.
        pval: P-value.

    Returns:
        Interpretation string.
    """
    # Strength
    abs_corr = abs(corr)
    if abs_corr < 0.1:
        strength = "Negligible"
    elif abs_corr < 0.3:
        strength = "Weak"
    elif abs_corr < 0.5:
        strength = "Moderate"
    elif abs_corr < 0.7:
        strength = "Strong"
    else:
        strength = "Very strong"

    # Direction
    direction = "positive" if corr > 0 else "negative"

    # Significance
    if pval < 0.001:
        sig = "p<0.001"
    elif pval < 0.01:
        sig = "p<0.01"
    elif pval < 0.05:
        sig = "p<0.05"
    else:
        sig = f"p={pval:.3f} (not significant)"

    return f"{strength} {direction} correlation (r={corr:.3f}, {sig})"


def run(output_dir: str | None = None) -> tuple[str, str]:
    """Run correlation analysis.

    Args:
        output_dir: Output directory (default: src/vis/outputs/).

    Returns:
        Tuple of (json_path, png_path).
    """
    # Set output directory
    output_dir_path = (
        Path("src/vis/outputs") if output_dir is None else Path(output_dir)
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    json_path = output_dir_path / "correlation_analysis.json"
    png_path = output_dir_path / "correlation_heatmap.png"

    # Load data
    print(f"Loading issue reports from {config['paths']['issue_reports']}...")
    issue_reports = read_json_file(config["paths"]["issue_reports"])

    # Apply confidence filtering if configured
    min_confidence = config.get("filtering", {}).get("min_confidence")
    if min_confidence:
        original_count = len(issue_reports)
        issue_reports = filter_by_confidence(issue_reports, min_confidence)
        print(
            f"  Filtered to {min_confidence}+ confidence: {len(issue_reports)}/{original_count}"
        )
    else:
        print(f"  Loaded {len(issue_reports)} issue reports")

    print(f"Loading conversations from {config['paths']['clean_conversations']}...")
    clean_conversations = read_json_file(config["paths"]["clean_conversations"])
    print(f"  Loaded {len(clean_conversations)} conversations")

    # Create analysis dataframe
    print("Creating analysis dataframe...")
    df = create_analysis_dataframe(issue_reports, clean_conversations)
    print(f"  Created dataframe with {len(df)} rows")

    # Compute summary statistics
    summary_stats = {}
    for col in COLUMNS:
        count = df[col].sum()
        proportion = count / len(df) if len(df) > 0 else 0
        summary_stats[col] = {"count": int(count), "proportion": round(proportion, 4)}
        print(f"  {DISPLAY_LABELS[col]}: {count} ({proportion:.1%})")

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(df)

    # Generate interpretations
    interpretations = {}
    corr_matrix = correlations["correlation_matrix"]
    pval_matrix = correlations["pvalue_matrix"]

    pairs = [
        ("is_unresolved", "used_unanswered_tool", "unresolved_vs_tool"),
        ("is_unresolved", "is_negative_sentiment", "unresolved_vs_sentiment"),
        ("used_unanswered_tool", "is_negative_sentiment", "tool_vs_sentiment"),
    ]

    for col1, col2, key in pairs:
        i = COLUMNS.index(col1)
        j = COLUMNS.index(col2)
        corr = corr_matrix[i][j]
        pval = pval_matrix[i][j]
        interpretation = interpret_correlation(corr, pval)
        interpretations[key] = interpretation
        print(f"  {DISPLAY_LABELS[col1]} vs {DISPLAY_LABELS[col2]}: {interpretation}")

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_conversations": len(df),
            "method": "pearson",
        },
        "summary_statistics": summary_stats,
        "correlations": correlations,
        "interpretations": interpretations,
    }

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Create heatmap using vis module
    print("Creating heatmap...")
    create_correlation_heatmap(correlations, png_path)
    print(f"Heatmap saved to: {png_path}")

    return str(json_path), str(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute correlations between resolution, tool usage, and sentiment."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: src/vis/outputs/)",
    )
    args = parser.parse_args()

    json_path, png_path = run(output_dir=args.output_dir)
