"""Cluster Scatter Plot Visualization.

2D scatter plot of clusters using UMAP-reduced embeddings.

Input:
    embeddings_2d - 2D coordinates from UMAP (n_samples, 2).
    labels.json - Cluster assignments.
    clusters_named.json - Named cluster data.

Output:
    Altair Chart object (caller saves to file).
"""

from typing import cast

import altair as alt
import numpy as np
import pandas as pd

from config import config


def create_cluster_scatter_plot(
    embeddings_2d: np.ndarray,
    ids: list[str],
    labels: dict[str, int],
    cluster_data: dict,
) -> alt.Chart:
    """Create a 2D scatter plot of clusters with colored points and legend.

    Args:
        embeddings_2d: 2D coordinates from UMAP (N x 2).
        ids: List of point IDs matching embeddings order.
        labels: Mapping of point ID to cluster label.
        cluster_data: Parsed clusters_named.json data.

    Returns:
        Altair Chart with scatter plot.
    """
    method = cluster_data["method"]
    n_clusters = cluster_data["n_clusters"]
    total_items = cluster_data["total_items"]

    # Check if there are noise points (only HDBSCAN has noise)
    has_noise = "noise" in cluster_data["clusters"]

    # Build cluster_id -> cluster_title mapping
    cluster_titles = {}
    for cluster_key, cluster_info in cluster_data["clusters"].items():
        if cluster_key == "noise":
            cluster_titles[-1] = "Noise"
        else:
            cluster_id = int(cluster_key.replace("cluster_", ""))
            cluster_titles[cluster_id] = cluster_info["cluster_title"]

    # Build DataFrame
    rows = []
    for i, point_id in enumerate(ids):
        cluster_id = labels.get(point_id, -1)
        cluster_name = cluster_titles.get(cluster_id, "Unknown")
        is_noise = cluster_id == -1
        rows.append(
            {
                "x": embeddings_2d[i, 0],
                "y": embeddings_2d[i, 1],
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "is_noise": is_noise,
            }
        )

    df = pd.DataFrame(rows)

    # Sort so noise points are rendered first (behind cluster points)
    df = df.sort_values("is_noise", ascending=False)

    # Compute axis ranges with padding to zoom into data
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # Get unique cluster names for color scale (only include Noise if it exists)
    cluster_names = [
        cluster_titles[i] for i in sorted(cluster_titles.keys()) if i != -1
    ]

    # Build color palette
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
    ]

    if has_noise:
        cluster_names.append("Noise")
        colors = color_palette[: len(cluster_names) - 1] + ["#cccccc"]
    else:
        colors = color_palette[: len(cluster_names)]

    # Create color scale
    color_scale = alt.Scale(domain=cluster_names, range=colors)

    # Create subtitle
    if method == "kmeans":
        subtitle = f"K-means clustering (k={n_clusters}) • {total_items} total items"
    else:
        noise_count = cluster_data["clusters"].get("noise", {}).get("count", 0)
        subtitle = f"HDBSCAN • {n_clusters} clusters • {noise_count} noise points • {total_items} total"

    # Create scatter plot with adaptive axis scaling
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "x:Q",
                title="UMAP 1",
                scale=alt.Scale(domain=[x_min - x_padding, x_max + x_padding]),
                axis=alt.Axis(grid=True),
            ),
            y=alt.Y(
                "y:Q",
                title="UMAP 2",
                scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding]),
                axis=alt.Axis(grid=True),
            ),
            color=alt.Color(
                "cluster_name:N",
                title="Cluster",
                scale=color_scale,
                legend=alt.Legend(
                    orient="right",
                    titleFontSize=12,
                    labelFontSize=10,
                    labelLimit=200,
                ),
            ),
            opacity=alt.condition(
                alt.datum.is_noise,
                alt.value(0.3),
                alt.value(0.8),
            ),
            tooltip=["cluster_name", "x", "y"],
        )
        .properties(
            title=alt.Title(
                text=f"{config['domain']['project_name']} Clusters (2D)",
                subtitle=subtitle,
            ),
            width=600,
            height=500,
        )
    )

    return cast(alt.Chart, chart)
