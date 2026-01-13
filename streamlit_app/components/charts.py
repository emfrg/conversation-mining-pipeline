"""Altair chart components for the Streamlit app."""

from typing import Any, cast

import altair as alt
import pandas as pd

# Consistent blue color palette
CHART_COLOR = "#1f77b4"


def create_horizontal_bar_chart(
    data: dict[str, int | float],
    x_title: str = "Value",
    y_title: str = "Category",
    title: str | None = None,
    color: str = CHART_COLOR,
    sort: str | None = "-x",
) -> alt.Chart:
    """Create a horizontal bar chart.

    Args:
        data: Dictionary with category names as keys and values.
        x_title: Label for the x-axis (values) - also used in tooltip.
        y_title: Label for the y-axis (categories) - also used in tooltip.
        title: Optional chart title.
        color: Bar color.
        sort: Sort order for y-axis ("-x" for descending by value, None for original order).

    Returns:
        Altair Chart object.
    """
    df = pd.DataFrame({y_title: list(data.keys()), x_title: list(data.values())})

    chart = (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x_title}:Q", title=x_title),
            y=alt.Y(f"{y_title}:N", title=y_title, sort=sort),
            tooltip=[
                alt.Tooltip(f"{y_title}:N", title=y_title),
                alt.Tooltip(f"{x_title}:Q", title=x_title),
            ],
        )
    )

    if title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


def create_line_chart(
    data: dict[str, int | float],
    x_title: str = "X",
    y_title: str = "Y",
    title: str | None = None,
    color: str = CHART_COLOR,
) -> alt.Chart:
    """Create a line chart.

    Args:
        data: Dictionary with x values as keys and y values.
        x_title: Label for the x-axis - also used in tooltip.
        y_title: Label for the y-axis - also used in tooltip.
        title: Optional chart title.
        color: Line color.

    Returns:
        Altair Chart object.
    """
    df = pd.DataFrame({x_title: list(data.keys()), y_title: list(data.values())})

    chart = (
        alt.Chart(df)
        .mark_line(color=color, point=True)
        .encode(
            x=alt.X(f"{x_title}:O", title=x_title),
            y=alt.Y(f"{y_title}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x_title}:O", title=x_title),
                alt.Tooltip(f"{y_title}:Q", title=y_title),
            ],
        )
    )

    if title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


def create_area_chart(
    data: dict[str, int | float],
    x_title: str = "X",
    y_title: str = "Y",
    title: str | None = None,
    color: str = CHART_COLOR,
    opacity: float = 0.7,
) -> alt.Chart:
    """Create an area chart.

    Args:
        data: Dictionary with x values as keys and y values.
        x_title: Label for the x-axis - also used in tooltip.
        y_title: Label for the y-axis - also used in tooltip.
        title: Optional chart title.
        color: Area fill color.
        opacity: Fill opacity.

    Returns:
        Altair Chart object.
    """
    df = pd.DataFrame({x_title: list(data.keys()), y_title: list(data.values())})

    chart = (
        alt.Chart(df)
        .mark_area(color=color, opacity=opacity, line=True)
        .encode(
            x=alt.X(f"{x_title}:O", title=x_title),
            y=alt.Y(f"{y_title}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x_title}:O", title=x_title),
                alt.Tooltip(f"{y_title}:Q", title=y_title),
            ],
        )
    )

    if title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


def create_cluster_bar_chart(
    df: "pd.DataFrame",
    title: str = "FAQ Clusters",
    subtitle: str | None = None,
    color: str = CHART_COLOR,
    selected_title: str | None = None,
) -> alt.Chart:
    """Create a horizontal bar chart for cluster visualization.

    Args:
        df: DataFrame with columns: 'title', 'count', 'description'
        title: Chart title.
        subtitle: Optional subtitle (e.g., "K-means clustering (k=12) • 1039 items")
        color: Bar color.
        selected_title: Title of the currently selected cluster (others will be grayed out).

    Returns:
        Altair Chart object.
    """
    # Sort by count descending
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    # Add selection state column
    if selected_title:
        df = df.copy()
        df["is_selected"] = df["title"] == selected_title

    # Base bar chart with conditional coloring
    color_encoding: Any
    if selected_title:
        color_encoding = alt.condition(
            alt.datum.is_selected,
            alt.value(color),
            alt.value("#cccccc"),
        )
    else:
        color_encoding = alt.value(color)

    # Define point selection for click interactivity (name is required for Streamlit)
    point_selection = alt.selection_point(name="cluster_select", fields=["title"])

    chart = (
        alt.Chart(df)
        .mark_bar(cursor="pointer")
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y("title:N", title=None, sort=None, axis=alt.Axis(labelLimit=400)),
            color=color_encoding,
            tooltip=[
                alt.Tooltip("title:N", title="Cluster"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("description:N", title="Description"),
            ],
        )
        .add_params(point_selection)
        .properties(
            width="container",
            height=alt.Step(50),
            padding={"right": 150},  # Reserve space for legend consistency
        )
    )

    # Add title and subtitle
    if subtitle:
        chart = chart.properties(title=alt.TitleParams(text=title, subtitle=subtitle))
    elif title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


# Resolution status colors
RESOLUTION_COLORS = {
    "resolved": "#2ecc71",  # Green
    "partially_resolved": "#f39c12",  # Orange/Yellow
    "unresolved": "#e74c3c",  # Red
}


def create_stacked_resolution_chart(
    df: "pd.DataFrame",
    title: str = "Resolution Status by Cluster",
    subtitle: str | None = None,
    selected_title: str | None = None,
) -> alt.Chart:
    """Create a stacked horizontal bar chart showing resolution status per cluster.

    Args:
        df: DataFrame with columns: 'title', 'resolved', 'partially_resolved', 'unresolved', 'total'
        title: Chart title.
        subtitle: Optional subtitle.
        selected_title: Title of the currently selected cluster (others will be grayed out).

    Returns:
        Altair Chart object.
    """
    # Sort by total descending
    df = df.sort_values("total", ascending=False).reset_index(drop=True)

    # Melt the dataframe for stacked bar chart
    df_melted = df.melt(
        id_vars=["title", "total"],
        value_vars=["resolved", "partially_resolved", "unresolved"],
        var_name="status",
        value_name="count",
    )

    # Compute percentage for each status
    df_melted["pct"] = (df_melted["count"] / df_melted["total"] * 100).round(1)

    # Define status order for stacking (resolved first, then partial, then unresolved)
    status_order = ["resolved", "partially_resolved", "unresolved"]

    # Create color scale
    color_scale = alt.Scale(
        domain=status_order,
        range=[RESOLUTION_COLORS[s] for s in status_order],
    )

    # Status labels for legend
    status_labels = {
        "resolved": "Resolved",
        "partially_resolved": "Partially Resolved",
        "unresolved": "Unresolved",
    }
    df_melted["status_label"] = df_melted["status"].map(status_labels)

    # Add sort order column for stacking (0=resolved first, 1=partial, 2=unresolved)
    status_sort_order = {"resolved": 0, "partially_resolved": 1, "unresolved": 2}
    df_melted["sort_order"] = df_melted["status"].map(status_sort_order)

    # Add selection state for graying out non-selected bars
    if selected_title:
        df_melted = df_melted.copy()
        df_melted["is_selected"] = df_melted["title"] == selected_title

    # Define point selection for click interactivity
    point_selection = alt.selection_point(name="resolution_select", fields=["title"])

    # Conditional opacity based on selection
    opacity_encoding: Any
    if selected_title:
        opacity_encoding = alt.condition(
            alt.datum.is_selected,
            alt.value(1.0),
            alt.value(0.3),
        )
    else:
        opacity_encoding = alt.value(1.0)

    chart = (
        alt.Chart(df_melted)
        .mark_bar(cursor="pointer")
        .encode(
            x=alt.X("count:Q", title="Count", stack="zero"),
            y=alt.Y("title:N", title=None, sort=None, axis=alt.Axis(labelLimit=400)),
            color=alt.Color(
                "status:N",
                scale=color_scale,
                legend=alt.Legend(title="Resolution Status"),
                sort=status_order,
            ),
            opacity=opacity_encoding,
            order=alt.Order("sort_order:Q"),
            tooltip=[
                alt.Tooltip("title:N", title="Cluster"),
                alt.Tooltip("status_label:N", title="Status"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("total:Q", title="Total in Cluster"),
                alt.Tooltip("pct:Q", title="Percentage", format=".1f"),
            ],
        )
        .add_params(point_selection)
        .properties(
            width="container",
            height=alt.Step(50),
        )
    )

    # Add title and subtitle
    if subtitle:
        chart = chart.properties(title=alt.TitleParams(text=title, subtitle=subtitle))
    elif title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


# Sentiment colors
SENTIMENT_COLORS = {
    "negative": "#e74c3c",  # Red
    "other": "#cccccc",  # Gray
}


def create_stacked_sentiment_chart(
    df: "pd.DataFrame",
    title: str = "Sentiment Analysis by Cluster",
    subtitle: str | None = None,
    selected_title: str | None = None,
) -> alt.Chart:
    """Create a stacked horizontal bar chart showing sentiment per cluster.

    Args:
        df: DataFrame with columns: 'title', 'negative', 'other', 'total'
        title: Chart title.
        subtitle: Optional subtitle.
        selected_title: Title of the currently selected cluster (others will be grayed out).

    Returns:
        Altair Chart object.
    """
    # Sort by total descending
    df = df.sort_values("total", ascending=False).reset_index(drop=True)

    # Melt the dataframe for stacked bar chart
    df_melted = df.melt(
        id_vars=["title", "total"],
        value_vars=["other", "negative"],  # other first, then negative on top
        var_name="sentiment",
        value_name="count",
    )

    # Compute percentage for each sentiment
    df_melted["pct"] = (df_melted["count"] / df_melted["total"] * 100).round(1)

    # Define sentiment order for stacking (other first, negative on top/right)
    sentiment_order = ["other", "negative"]

    # Create color scale
    color_scale = alt.Scale(
        domain=sentiment_order,
        range=[SENTIMENT_COLORS[s] for s in sentiment_order],
    )

    # Sentiment labels for legend
    sentiment_labels = {
        "negative": "Negative",
        "other": "Neutral/Positive",
    }
    df_melted["sentiment_label"] = df_melted["sentiment"].map(sentiment_labels)

    # Add sort order column for stacking (0=other first, 1=negative)
    sentiment_sort_order = {"other": 0, "negative": 1}
    df_melted["sort_order"] = df_melted["sentiment"].map(sentiment_sort_order)

    # Add selection state for graying out non-selected bars
    if selected_title:
        df_melted = df_melted.copy()
        df_melted["is_selected"] = df_melted["title"] == selected_title

    # Define point selection for click interactivity
    point_selection = alt.selection_point(name="sentiment_select", fields=["title"])

    # Conditional opacity based on selection
    opacity_encoding: Any
    if selected_title:
        opacity_encoding = alt.condition(
            alt.datum.is_selected,
            alt.value(1.0),
            alt.value(0.3),
        )
    else:
        opacity_encoding = alt.value(1.0)

    chart = (
        alt.Chart(df_melted)
        .mark_bar(cursor="pointer")
        .encode(
            x=alt.X("count:Q", title="Count", stack="zero"),
            y=alt.Y("title:N", title=None, sort=None, axis=alt.Axis(labelLimit=400)),
            color=alt.Color(
                "sentiment:N",
                scale=color_scale,
                legend=alt.Legend(title="Sentiment"),
                sort=sentiment_order,
            ),
            opacity=opacity_encoding,
            order=alt.Order("sort_order:Q"),
            tooltip=[
                alt.Tooltip("title:N", title="Cluster"),
                alt.Tooltip("sentiment_label:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("total:Q", title="Total in Cluster"),
                alt.Tooltip("pct:Q", title="Percentage", format=".1f"),
            ],
        )
        .add_params(point_selection)
        .properties(
            width="container",
            height=alt.Step(50),
        )
    )

    # Add title and subtitle
    if subtitle:
        chart = chart.properties(title=alt.TitleParams(text=title, subtitle=subtitle))
    elif title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)


# =============================================================================
# CUSTOMIZE FOR YOUR CHATBOT
# =============================================================================
# These tool names are specific to your chatbot implementation.
# Edit these to match your chatbot's tool names from chat_history.json.
# =============================================================================

# Tool use colors (key = tool name, value = hex color)
TOOL_USE_COLORS = {
    "no_tool": "#cccccc",  # Gray - keep this for conversations with no tool use
    "rag_tool": "#3498db",  # Blue
    "zone_checker": "#2ecc71",  # Green
    "unanswered_question_tool": "#f39c12",  # Orange
    "feedback_tool": "#9b59b6",  # Purple
}

# Tool use order for stacking in charts
TOOL_USE_ORDER = [
    "no_tool",
    "rag_tool",
    "zone_checker",
    "unanswered_question_tool",
    "feedback_tool",
]


def create_stacked_tool_use_chart(
    df: "pd.DataFrame",
    title: str = "Tool Usage by Cluster",
    subtitle: str | None = None,
    selected_title: str | None = None,
) -> alt.Chart:
    """Create a stacked horizontal bar chart showing tool usage per cluster.

    Args:
        df: DataFrame with columns: 'title', 'no_tool', 'rag_tool', 'zone_checker',
            'unanswered_question_tool', 'feedback_tool', 'total'
        title: Chart title.
        subtitle: Optional subtitle.
        selected_title: Title of the currently selected cluster (others will be grayed out).

    Returns:
        Altair Chart object.
    """
    # Sort by total descending
    df = df.sort_values("total", ascending=False).reset_index(drop=True)

    # Melt the dataframe for stacked bar chart
    df_melted = df.melt(
        id_vars=["title", "total"],
        value_vars=TOOL_USE_ORDER,
        var_name="tool",
        value_name="count",
    )

    # Compute percentage for each tool
    df_melted["pct"] = (df_melted["count"] / df_melted["total"] * 100).round(1)

    # Create color scale
    color_scale = alt.Scale(
        domain=TOOL_USE_ORDER,
        range=[TOOL_USE_COLORS[t] for t in TOOL_USE_ORDER],
    )

    # Tool labels for legend (customize these for your chatbot's tools)
    tool_labels = {
        "no_tool": "No Tool",
        "rag_tool": "RAG Tool",
        "zone_checker": "Zone Checker",
        "unanswered_question_tool": "Unanswered Question",
        "feedback_tool": "Feedback Tool",
    }
    df_melted["tool_label"] = df_melted["tool"].map(tool_labels)

    # Add sort order column for stacking
    tool_sort_order = {t: i for i, t in enumerate(TOOL_USE_ORDER)}
    df_melted["sort_order"] = df_melted["tool"].map(tool_sort_order)

    # Add selection state for graying out non-selected bars
    if selected_title:
        df_melted = df_melted.copy()
        df_melted["is_selected"] = df_melted["title"] == selected_title

    # Define point selection for click interactivity
    point_selection = alt.selection_point(name="tool_use_select", fields=["title"])

    # Conditional opacity based on selection
    opacity_encoding: Any
    if selected_title:
        opacity_encoding = alt.condition(
            alt.datum.is_selected,
            alt.value(1.0),
            alt.value(0.3),
        )
    else:
        opacity_encoding = alt.value(1.0)

    chart = (
        alt.Chart(df_melted)
        .mark_bar(cursor="pointer")
        .encode(
            x=alt.X("count:Q", title="Count", stack="zero"),
            y=alt.Y("title:N", title=None, sort=None, axis=alt.Axis(labelLimit=400)),
            color=alt.Color(
                "tool:N",
                scale=color_scale,
                legend=alt.Legend(title="Tool"),
                sort=TOOL_USE_ORDER,
            ),
            opacity=opacity_encoding,
            order=alt.Order("sort_order:Q"),
            tooltip=[
                alt.Tooltip("title:N", title="Cluster"),
                alt.Tooltip("tool_label:N", title="Tool"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("total:Q", title="Total in Cluster"),
                alt.Tooltip("pct:Q", title="Percentage", format=".1f"),
            ],
        )
        .add_params(point_selection)
        .properties(
            width="container",
            height=alt.Step(50),
        )
    )

    # Add title and subtitle
    if subtitle:
        chart = chart.properties(title=alt.TitleParams(text=title, subtitle=subtitle))
    elif title:
        chart = chart.properties(title=title)

    return cast(alt.Chart, chart)
