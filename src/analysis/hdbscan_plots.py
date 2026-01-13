"""HDBSCAN Parameter Analysis Visualization.

Generates DBCV and silhouette score plots for min_cluster_size analysis.
Uses Altair for visualization.

Input:
    analysis: Parameter analysis from hdbscan_selection.compute_parameter_analysis().

Output:
    PNG chart showing DBCV and silhouette scores with selected size marked.
"""

import altair as alt
import pandas as pd


def create_hdbscan_analysis_chart(
    analysis: dict,
    selected_size: int,
    output_path: str,
) -> None:
    """
    Create and save HDBSCAN DBCV/silhouette analysis chart.

    Generates a two-panel chart:
    - Top: DBCV score vs min_cluster_size
    - Bottom: Silhouette score vs min_cluster_size

    Key points are marked:
    - Best DBCV (red)
    - Best silhouette (green)
    - Selected size (blue dashed line)

    Args:
        analysis: Dict with all_results, best_dbcv_size, best_silhouette_size.
        selected_size: The min_cluster_size that was selected/used.
        output_path: Path to save PNG.
    """
    all_results = analysis["all_results"]
    best_dbcv_size = analysis["best_dbcv_size"]
    best_sil_size = analysis["best_silhouette_size"]

    # Build dataframe from analysis results
    df = pd.DataFrame(
        [
            {
                "size": int(k),
                "dbcv": v["dbcv"],
                "silhouette": v["silhouette"] if v["silhouette"] != -1 else None,
            }
            for k, v in all_results.items()
        ]
    )
    df = df.sort_values("size")

    # Create marker dataframes for special points
    dbcv_point = df[df["size"] == best_dbcv_size]
    best_sil_point = df[df["size"] == best_sil_size]
    selected_line = pd.DataFrame({"size": [selected_size]})

    # Base DBCV chart (line + points)
    dbcv_line = (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("size:Q", title="min_cluster_size", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("dbcv:Q", title="DBCV Score"),
        )
    )

    dbcv_points = (
        alt.Chart(df)
        .mark_circle(size=50, color="#1f77b4")
        .encode(
            x="size:Q",
            y="dbcv:Q",
        )
    )

    # Best DBCV point marker (red)
    dbcv_marker = (
        alt.Chart(dbcv_point)
        .mark_circle(size=150, color="red", strokeWidth=2)
        .encode(
            x="size:Q",
            y="dbcv:Q",
            tooltip=[
                alt.Tooltip("size:Q", title="Best DBCV size"),
                alt.Tooltip("dbcv:Q", title="DBCV", format=".4f"),
            ],
        )
    )

    # Selected size vertical rule (blue dashed)
    selected_rule = (
        alt.Chart(selected_line)
        .mark_rule(color="blue", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="size:Q")
    )

    dbcv_chart = alt.layer(
        dbcv_line, dbcv_points, dbcv_marker, selected_rule
    ).properties(
        title="DBCV Score",
        width=500,
        height=200,
    )

    # Base silhouette chart (line + points) - filter out None values
    df_sil = df.dropna(subset=["silhouette"])

    sil_line = (
        alt.Chart(df_sil)
        .mark_line(color="#ff7f0e")
        .encode(
            x=alt.X("size:Q", title="min_cluster_size", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("silhouette:Q", title="Silhouette Score"),
        )
    )

    sil_points = (
        alt.Chart(df_sil)
        .mark_circle(size=50, color="#ff7f0e")
        .encode(
            x="size:Q",
            y="silhouette:Q",
        )
    )

    # Best silhouette point marker (green)
    best_sil_point_filtered = best_sil_point.dropna(subset=["silhouette"])
    if not best_sil_point_filtered.empty:
        best_sil_marker = (
            alt.Chart(best_sil_point_filtered)
            .mark_circle(size=150, color="green", strokeWidth=2)
            .encode(
                x="size:Q",
                y="silhouette:Q",
                tooltip=[
                    alt.Tooltip("size:Q", title="Best size"),
                    alt.Tooltip("silhouette:Q", title="Silhouette", format=".4f"),
                ],
            )
        )
        sil_chart = alt.layer(sil_line, sil_points, best_sil_marker, selected_rule)
    else:
        sil_chart = alt.layer(sil_line, sil_points, selected_rule)

    sil_chart = sil_chart.properties(
        title="Silhouette Score",
        width=500,
        height=200,
    )

    # Combine charts vertically
    combined = (
        alt.vconcat(
            dbcv_chart,
            sil_chart,
        )
        .properties(
            title=alt.TitleParams(
                text="HDBSCAN Parameter Analysis",
                subtitle=f"Selected size={selected_size} | Best DBCV size={best_dbcv_size} (red) | Best Silhouette size={best_sil_size} (green)",
                fontSize=16,
            ),
        )
        .resolve_scale(x="shared")
    )

    # Save to PNG
    combined.save(output_path)
    print(f"  Saved HDBSCAN analysis chart to {output_path}")
