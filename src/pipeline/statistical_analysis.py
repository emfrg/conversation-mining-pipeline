"""Step 10: Statistical Analysis - Compute usage stats, correlations, and chi-square tests.

Runs three analyses in sequence:
1. Usage Statistics - conversation counts, message distributions, temporal patterns
2. Correlation Analysis - Pearson correlations between resolution/sentiment/tool usage
3. Chi-Square Analysis - statistical independence tests for the same variables

This step only runs on FULL data and skips automatically in snippet mode.
Statistical analysis on partial data would not be meaningful.

Reads:
    config.paths.raw_data - Raw chat history for usage statistics.
    config.paths.issue_reports - Issue reports with resolution/sentiment.
    config.paths.clean_conversations - Clean conversations for tool usage.

Writes:
    data/output/chat_history_stats.json - Usage statistics.
    src/vis/outputs/correlation_analysis.json - Correlation matrix with p-values.
    src/vis/outputs/correlation_heatmap.png - Heatmap visualization.
    src/vis/outputs/chi_square_analysis.json - Chi-square test results.
    src/vis/outputs/chi_square_proportions.png - Bar chart visualization.
"""

import os

from config import config

# Output files to check for --recreate logic
OUTPUT_FILES = [
    "data/output/chat_history_stats.json",
    "src/vis/outputs/correlation_analysis.json",
    "src/vis/outputs/chi_square_analysis.json",
]


def run() -> None:
    """Run statistical analysis pipeline step.

    Skips if:
    - Running in snippet mode (statistical analysis on partial data not meaningful)
    - All output files exist and --recreate not specified
    """
    # Check snippet mode - skip entirely
    if config.get("snippet", {}).get("num_conversations"):
        print("  Skipping: statistical analysis not meaningful on snippet data")
        return

    # Check recreate flag
    recreate = config.get("recreate", False)
    if not recreate and all(os.path.exists(f) for f in OUTPUT_FILES):
        print("  Skipping: outputs already exist (use --recreate to force)")
        return

    # Import analysis modules here to avoid circular imports
    from src.analysis import (
        chi_square_analysis,
        correlation_analysis,
        get_usage_statistics,
    )

    # Run usage statistics
    print("\n  --- Usage Statistics ---")
    get_usage_statistics.run()

    # Run correlation analysis
    print("\n  --- Correlation Analysis ---")
    correlation_analysis.run()

    # Run chi-square analysis
    print("\n  --- Chi-Square Analysis ---")
    chi_square_analysis.run()

    print("\n  Statistical analysis complete.")


if __name__ == "__main__":
    run()
