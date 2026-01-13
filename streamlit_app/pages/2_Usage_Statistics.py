"""Usage Statistics page - displays chatbot usage metrics and trends."""

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from streamlit_app.components.charts import create_horizontal_bar_chart
from streamlit_app.utils.data_loader import load_statistics

st.title("Usage Statistics")

# Load data
stats = load_statistics()

if stats is None:
    st.error(
        "Statistics data not found. Please run `uv run python -m src.analysis.get_usage_statistics` first to generate the data."
    )
    st.stop()

# Extract sections
overview = stats.get("overview", {})
conversation_length = stats.get("conversation_length", {})
message_lengths = stats.get("message_lengths", {})
features = stats.get("features", {})
temporal = stats.get("temporal_distribution", {})

# =============================================================================
# Overview - KPI Cards (single row)
# =============================================================================
st.header("Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Conversations",
        value=f"{overview.get('total_conversations', 0):,}",
    )

with col2:
    st.metric(
        label="Total Messages",
        value=f"{overview.get('total_messages', 0):,}",
    )

with col3:
    st.metric(
        label="Date Range",
        value=f"{overview.get('date_range_days', 0)} days",
    )
    # Format date range as "3 Jun to 5 Aug"
    first = overview.get("first_conversation", "")[:10]
    last = overview.get("last_conversation", "")[:10]
    if first and last:
        first_fmt = datetime.strptime(first, "%Y-%m-%d").strftime("%-d %b")
        last_fmt = datetime.strptime(last, "%Y-%m-%d").strftime("%-d %b")
        st.caption(f"{first_fmt} to {last_fmt}")

with col4:
    st.metric(
        label="Avg Conversations/Day",
        value=f"{overview.get('avg_conversations_per_day', 0):.1f}",
    )

with col5:
    st.metric(
        label="Avg Messages/Conversation",
        value=f"{conversation_length.get('median_messages_per_conversation', 0):.0f}",
    )

st.divider()

# =============================================================================
# Temporal Distribution
# =============================================================================

# Day of week and Busiest dates (same row)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversations by Day of Week")
    by_weekday = temporal.get("by_weekday", {})
    if by_weekday:
        weekday_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        ordered_weekday = {day: by_weekday.get(day, 0) for day in weekday_order}
        weekday_chart = create_horizontal_bar_chart(
            ordered_weekday,
            x_title="Conversations",
            y_title="Day",
            sort=None,
        ).properties(height=300)
        st.altair_chart(weekday_chart, width="stretch")

with col2:
    st.subheader("Busiest Dates")
    busiest_dates = temporal.get("busiest_dates", {})
    if busiest_dates:
        sorted_dates = sorted(busiest_dates.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        # Format dates as "Sun, 25 Dec"
        formatted_dates = {
            datetime.strptime(date, "%Y-%m-%d").strftime("%a, %d %b"): count
            for date, count in sorted_dates
        }
        dates_chart = create_horizontal_bar_chart(
            formatted_dates,
            x_title="Conversations",
            y_title="Date",
            sort=None,  # Preserve descending order by count
        ).properties(height=300)
        st.altair_chart(dates_chart, width="stretch")

# Hourly distribution
st.subheader("Conversations by Hour of Day")
by_hour = temporal.get("by_hour", {})
if by_hour:
    # Create dataframe with time format labels
    hour_labels = [f"{h}:00" for h in range(24)]
    df_hours = pd.DataFrame(
        {
            "Hour": hour_labels,
            "Conversations": [by_hour.get(str(h), 0) for h in range(24)],
        }
    )
    hourly_chart = (
        alt.Chart(df_hours)
        .mark_area(color="#1f77b4", opacity=0.7, line=True)
        .encode(
            x=alt.X("Hour:O", title="Hour", sort=hour_labels),
            y=alt.Y("Conversations:Q", title="Conversations"),
            tooltip=[
                alt.Tooltip("Hour:O", title="Hour"),
                alt.Tooltip("Conversations:Q", title="Conversations"),
            ],
        )
    )
    st.altair_chart(hourly_chart, width="stretch")

st.divider()

# =============================================================================
# Tools Used + Message Length Pie Chart (same row)
# =============================================================================
st.header("Technical Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tools Used")
    top_tools = features.get("top_tools_used", {})
    if top_tools:
        tools_chart = create_horizontal_bar_chart(
            top_tools,
            x_title="Usage Count",
            y_title="Tool",
        ).properties(height=250)
        st.altair_chart(tools_chart, width="stretch")
    else:
        st.info("No tool usage data available.")

    # Tool and citation usage info
    st.caption(
        f"Out of {overview.get('total_conversations', 0):,} conversations, "
        f"{features.get('conversations_with_tool_use', 0):,} used tools and "
        f"{features.get('conversations_with_citations', 0):,} included citations."
    )

with col2:
    st.subheader("Avg Message Length")
    user_median = message_lengths.get("median_user_message_chars", 0)
    assistant_median = message_lengths.get("median_assistant_message_chars", 0)

    df_pie = pd.DataFrame(
        {
            "Type": ["User", "Assistant"],
            "Characters": [user_median, assistant_median],
        }
    )

    pie_chart = (
        alt.Chart(df_pie)
        .mark_arc(innerRadius=30)
        .encode(
            theta=alt.Theta("Characters:Q"),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["User", "Assistant"],
                    range=["#1f77b4", "#6baed6"],
                ),
                legend=alt.Legend(title="Message Type"),
            ),
            tooltip=[
                alt.Tooltip("Type:N", title="Type"),
                alt.Tooltip("Characters:Q", title="Characters"),
            ],
        )
        .properties(height=200)
    )
    st.altair_chart(pie_chart, width="stretch")

# =============================================================================
# Footer
# =============================================================================
st.divider()

st.caption(
    f"Data period: {overview.get('first_conversation', 'N/A')[:10]} to "
    f"{overview.get('last_conversation', 'N/A')[:10]}"
)
