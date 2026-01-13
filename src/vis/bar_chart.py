"""Cluster Bar Chart Visualization.

Horizontal bar chart showing cluster counts sorted by size.

Input:
    clusters_named.json - Named cluster data with counts.

Output:
    Altair LayerChart object (caller saves to file).
"""

from typing import cast

import altair as alt
import pandas as pd

from config import config


def create_cluster_bar_chart(
    clusters_data: dict,
    include_noise: bool = False,
) -> alt.LayerChart:
    """Create a horizontal bar chart of cluster counts.

    Args:
        clusters_data: Parsed clusters_named.json data.
        include_noise: Whether to include noise cluster in chart.

    Returns:
        Altair LayerChart with bars and count labels.
    """
    method = clusters_data["method"]
    total_items = clusters_data["total_items"]
    n_clusters = clusters_data["n_clusters"]

    # Extract cluster names and counts
    rows = []
    for cluster_key, cluster_info in clusters_data["clusters"].items():
        if cluster_key == "noise" and not include_noise:
            continue
        rows.append(
            {
                "cluster_title": cluster_info["cluster_title"],
                "count": cluster_info["count"],
            }
        )

    df = pd.DataFrame(rows)

    # Create subtitle based on method
    if method == "kmeans":
        subtitle = f"K-means clustering (k={n_clusters}) • {total_items} total items"
    else:
        # Get noise count for HDBSCAN
        noise_count = clusters_data["clusters"].get("noise", {}).get("count", 0)
        subtitle = (
            f"HDBSCAN clustering • {n_clusters} clusters • "
            f"{total_items} total items • {noise_count} noise"
        )

    # Base bar chart
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count", axis=alt.Axis(format="d", tickMinStep=1)),
            y=alt.Y(
                "cluster_title:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=400),
            ),
            tooltip=["cluster_title", "count"],
        )
    )

    # Text labels on bars (integer format)
    text = (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=3,
        )
        .encode(
            x=alt.X("count:Q"),
            y=alt.Y("cluster_title:N", sort="-x"),
            text=alt.Text("count:Q", format="d"),
        )
    )

    chart = (bars + text).properties(
        title=alt.Title(
            text=f"{config['domain']['project_name']} Clusters",
            subtitle=subtitle,
        ),
        width=400,
        height=alt.Step(50),
    )

    return cast(alt.LayerChart, chart)
