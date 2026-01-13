"""Tag Word Cloud Visualization.

Word cloud showing tag frequency across all issues,
where tag size corresponds to frequency.

Input:
    issue_reports.json - Issue reports with tags field.

Output:
    PNG word cloud image.
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def clean_tag(tag: str) -> str:
    """Clean a tag: remove underscores and convert to uppercase.

    Args:
        tag: Raw tag string.

    Returns:
        Cleaned tag string.
    """
    return tag.replace("_", " ").upper()


def compute_tag_frequencies(issues_data: dict) -> dict:
    """Compute frequency of each tag across all issues.

    Args:
        issues_data: Parsed issue_reports.json data.

    Returns:
        Dict mapping tag to count.
    """
    all_tags = []

    for _issue_id, issue in issues_data.items():
        tags = issue.get("tags", [])
        cleaned_tags = [clean_tag(tag) for tag in tags]
        all_tags.extend(cleaned_tags)

    return dict(Counter(all_tags))


def create_tag_wordcloud(
    issues_data: dict,
    output_path: Path | str,
) -> Path | None:
    """Create and save a word cloud visualization of tags.

    Args:
        issues_data: Parsed issue_reports.json data.
        output_path: Path to save the chart.

    Returns:
        Path to the saved chart, or None if no tags found.
    """
    output_path = Path(output_path)
    tag_frequencies = compute_tag_frequencies(issues_data)

    if not tag_frequencies:
        return None

    # Create word cloud
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        max_words=200,
        min_font_size=10,
        max_font_size=150,
        colormap="viridis",
        prefer_horizontal=0.7,
        relative_scaling=0.5,
        margin=10,
    )

    # Generate from frequencies
    wc.generate_from_frequencies(tag_frequencies)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path
