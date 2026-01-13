"""K-Means K-Selection Visualization.

Generates elbow curve and silhouette score plots for cluster analysis.
Uses Altair for visualization.

Input:
    analysis: K-analysis results from kmeans_selection.compute_k_analysis().

Output:
    PNG chart showing inertia curve and silhouette scores with selected k marked.
"""

import altair as alt
import pandas as pd


def create_k_analysis_chart(
    analysis: dict,
    selected_k: int,
    output_path: str,
) -> None:
    """
    Create and save elbow/silhouette analysis chart.

    Generates a two-panel chart:
    - Top: Elbow curve (inertia vs k)
    - Bottom: Silhouette score vs k

    Key points are marked:
    - Detected elbow point
    - Best silhouette score
    - Selected k (blue dashed line)

    Args:
        analysis: Dict with all_results, elbow_k, best_silhouette_k.
        selected_k: The k value that was selected/used.
        output_path: Path to save PNG.
    """
    all_results = analysis["all_results"]
    elbow_k = analysis["elbow_k"]
    best_sil_k = analysis["best_silhouette_k"]

    # Build dataframe from analysis results
    df = pd.DataFrame(
        [
            {"k": int(k), "inertia": v["inertia"], "silhouette": v["silhouette"]}
            for k, v in all_results.items()
        ]
    )
    df = df.sort_values("k")

    # Create marker dataframes for special points
    elbow_point = df[df["k"] == elbow_k]
    best_sil_point = df[df["k"] == best_sil_k]
    selected_line = pd.DataFrame({"k": [selected_k]})

    # Base elbow chart (line + points)
    elbow_line = (
        alt.Chart(df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X(
                "k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)
            ),
            y=alt.Y("inertia:Q", title="Inertia (SSE)"),
        )
    )

    elbow_points = (
        alt.Chart(df)
        .mark_circle(size=50, color="#1f77b4")
        .encode(
            x="k:Q",
            y="inertia:Q",
        )
    )

    # Elbow point marker (red)
    elbow_marker = (
        alt.Chart(elbow_point)
        .mark_circle(size=150, color="red", strokeWidth=2)
        .encode(
            x="k:Q",
            y="inertia:Q",
            tooltip=[
                alt.Tooltip("k:Q", title="Elbow k"),
                alt.Tooltip("inertia:Q", title="Inertia", format=".2f"),
            ],
        )
    )

    # Selected k vertical rule (blue dashed)
    selected_rule = (
        alt.Chart(selected_line)
        .mark_rule(color="blue", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="k:Q")
    )

    elbow_chart = alt.layer(
        elbow_line, elbow_points, elbow_marker, selected_rule
    ).properties(
        title="Elbow Method",
        width=500,
        height=200,
    )

    # Base silhouette chart (line + points)
    sil_line = (
        alt.Chart(df)
        .mark_line(color="#ff7f0e")
        .encode(
            x=alt.X(
                "k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)
            ),
            y=alt.Y("silhouette:Q", title="Silhouette Score"),
        )
    )

    sil_points = (
        alt.Chart(df)
        .mark_circle(size=50, color="#ff7f0e")
        .encode(
            x="k:Q",
            y="silhouette:Q",
        )
    )

    # Best silhouette point marker (green)
    best_sil_marker = (
        alt.Chart(best_sil_point)
        .mark_circle(size=150, color="green", strokeWidth=2)
        .encode(
            x="k:Q",
            y="silhouette:Q",
            tooltip=[
                alt.Tooltip("k:Q", title="Best k"),
                alt.Tooltip("silhouette:Q", title="Silhouette", format=".4f"),
            ],
        )
    )

    # Selected k vertical rule (blue dashed)
    selected_rule_sil = (
        alt.Chart(selected_line)
        .mark_rule(color="blue", strokeDash=[5, 5], strokeWidth=2)
        .encode(x="k:Q")
    )

    sil_chart = alt.layer(
        sil_line, sil_points, best_sil_marker, selected_rule_sil
    ).properties(
        title="Silhouette Score",
        width=500,
        height=200,
    )

    # Combine charts vertically
    combined = (
        alt.vconcat(
            elbow_chart,
            sil_chart,
        )
        .properties(
            title=alt.TitleParams(
                text="KMeans K-Selection Analysis",
                subtitle=f"Selected k={selected_k} | Elbow k={elbow_k} (red) | Best Silhouette k={best_sil_k} (green)",
                fontSize=16,
            ),
        )
        .resolve_scale(x="shared")
    )

    # Save to PNG
    combined.save(output_path)
    print(f"  Saved k-analysis chart to {output_path}")
