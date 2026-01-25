"""Chi-Square Analysis for FAQ Analytics.

Performs chi-square tests for independence between:
- Unresolved status (resolution_status == "unresolved")
- Unanswered question tool usage
- Negative sentiment (user_sentiment == "negative")

Reads:
    config.paths.issue_reports - Resolution status and sentiment.
    config.paths.clean_conversations - Tool usage data.

Writes:
    src/vis/outputs/chi_square_analysis.json - Test results for all pairs.
    src/vis/outputs/chi_square_proportions.png - Bar charts showing proportions.
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
from src.vis.chi_square_proportions import create_proportion_charts

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
        DataFrame with binary flags for each variable.
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


def perform_chi_square_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> dict:
    """Perform chi-square test between two binary variables.

    Args:
        df: DataFrame with binary columns.
        var1: First variable name.
        var2: Second variable name.

    Returns:
        Dict with test results and contingency table.
    """
    contingency = pd.crosstab(df[var1], df[var2])
    chi2, pval, dof, expected = stats.chi2_contingency(contingency)

    # Calculate effect size (Cramér's V for 2x2 table = phi coefficient)
    n = len(df)
    cramers_v = np.sqrt(chi2 / n)

    # Interpret effect size
    if cramers_v < 0.1:
        effect_size_interpretation = "negligible"
    elif cramers_v < 0.3:
        effect_size_interpretation = "small"
    elif cramers_v < 0.5:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"

    return {
        "variables": [var1, var2],
        "variable_labels": [DISPLAY_LABELS[var1], DISPLAY_LABELS[var2]],
        "chi2_statistic": float(round(chi2, 4)),
        "p_value": float(pval),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(round(cramers_v, 4)),
        "effect_size_interpretation": effect_size_interpretation,
        "contingency_table": {
            str(k1): {str(k2): int(v) for k2, v in row.items()}
            for k1, row in contingency.to_dict().items()
        },
        "expected_frequencies": expected.tolist(),
        "significant_at_05": bool(pval < 0.05),
        "significant_at_01": bool(pval < 0.01),
        "significant_at_001": bool(pval < 0.001),
    }


def run(output_dir: str | None = None) -> tuple[str, str]:
    """Run chi-square analysis.

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

    json_path = output_dir_path / "chi_square_analysis.json"
    png_path = output_dir_path / "chi_square_proportions.png"

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

    # Perform chi-square tests for all pairs
    print("\nPerforming chi-square tests...")
    tests = []

    pairs = [
        ("is_unresolved", "used_unanswered_tool"),
        ("is_unresolved", "is_negative_sentiment"),
        ("used_unanswered_tool", "is_negative_sentiment"),
    ]

    for var1, var2 in pairs:
        result = perform_chi_square_test(df, var1, var2)
        tests.append(result)

        sig = (
            "***"
            if result["significant_at_001"]
            else (
                "**"
                if result["significant_at_01"]
                else ("*" if result["significant_at_05"] else "")
            )
        )
        print(
            f"  {result['variable_labels'][0]} vs {result['variable_labels'][1]}: "
            f"χ²={result['chi2_statistic']:.2f}, p={result['p_value']:.4f}{sig}, "
            f"φ={result['cramers_v']:.3f} ({result['effect_size_interpretation']})"
        )

    # Build output
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_conversations": len(df),
        },
        "tests": tests,
    }

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Create proportion charts using vis module
    print("Creating proportion charts...")
    create_proportion_charts(df, png_path)
    print(f"Charts saved to: {png_path}")

    return str(json_path), str(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform chi-square tests between resolution, tool usage, and sentiment."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: src/vis/outputs/)",
    )
    args = parser.parse_args()

    json_path, png_path = run(output_dir=args.output_dir)
