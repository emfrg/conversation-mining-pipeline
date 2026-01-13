"""Cluster Analysis page - Interactive visualization of FAQ clusters."""

import pandas as pd
import streamlit as st

from streamlit_app.components.charts import (
    create_cluster_bar_chart,
    create_stacked_resolution_chart,
    create_stacked_sentiment_chart,
    create_stacked_tool_use_chart,
)
from streamlit_app.config import PROJECT_NAME
from streamlit_app.utils.data_loader import (
    load_cluster_data,
    load_cluster_stats,
    load_final_deduplicated,
)

# Prevent scroll-to-top on rerun + custom link styling
st.markdown(
    """
    <style>
    * {
        overflow-anchor: none !important;
    }
    .question-link {
        color: inherit !important;
        text-decoration: none !important;
    }
    .question-link:hover {
        color: #0068c9 !important;
        text-decoration: underline !important;
    }
    .presupposition-label {
        color: #7f7f7f !important;
        font-style: italic;
    }
    .original-question {
        margin-left: 1rem;
        padding: 0.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cluster Analysis")

# Load cluster data
cluster_data = load_cluster_data()

if cluster_data is None:
    st.error("Cluster data not found. Please run the clustering pipeline first.")
    st.stop()

# Extract metadata
method = cluster_data.get("method", "kmeans")
total_items = cluster_data.get("total_items", 0)
n_clusters = cluster_data.get("n_clusters", 0)
clusters = cluster_data.get("clusters", {})

if not clusters:
    st.warning("No clusters found in the data.")
    st.stop()

# Check if HDBSCAN with noise cluster
is_hdbscan = method == "hdbscan"
has_noise = "noise" in clusters

# Noise cluster toggle (only for HDBSCAN with noise)
show_noise = False
if is_hdbscan and has_noise:
    show_noise = st.checkbox("Show noise cluster", value=False)

# Filter out noise cluster if not showing
if not show_noise and has_noise:
    clusters = {k: v for k, v in clusters.items() if k != "noise"}

# Build DataFrame for chart
cluster_list = []
for cluster_key, cluster_info in clusters.items():
    cluster_list.append(
        {
            "key": cluster_key,
            "title": cluster_info.get("cluster_title", cluster_key),
            "count": cluster_info.get("count", 0),
            "description": cluster_info.get("cluster_description", ""),
            "detailed_description": cluster_info.get(
                "cluster_detailed_description", ""
            ),
        }
    )

# Sort by count descending
cluster_list = sorted(cluster_list, key=lambda x: x["count"], reverse=True)
df = pd.DataFrame(cluster_list)
titles = [row["title"] for row in cluster_list]

# Initialize session state (None = no selection, all bars blue)
if "selected_cluster_title" not in st.session_state:
    st.session_state.selected_cluster_title = None

# Radio button group for view mode selection
view_mode = st.radio(
    "View mode",
    options=["Simple View", "Resolution Status", "Sentiment Analysis", "Tool Use"],
    index=1,  # Default to Resolution Status
    horizontal=True,
)

# Display chart with click selection
subtitle = f"K-means clustering (k={n_clusters}) • {total_items} total conversations"

# Load pre-computed cluster stats for resolution/sentiment/tool use views
cluster_stats = None
if view_mode in ["Resolution Status", "Sentiment Analysis", "Tool Use"]:
    cluster_stats = load_cluster_stats()
    if cluster_stats is None:
        st.warning(
            "Cluster stats not found. Run the pipeline with compute_cluster_stats step."
        )
        view_mode = "Simple View"

if view_mode == "Resolution Status":
    assert cluster_stats is not None  # Guarded by check above
    resolution_data = cluster_stats.get("resolution_by_cluster", [])

    # Filter out noise cluster if not showing
    if not show_noise:
        resolution_data = [
            r for r in resolution_data if "Noise" not in r.get("title", "")
        ]

    if not resolution_data:
        st.warning("Resolution stats not available.")
    else:
        df_resolution = pd.DataFrame(resolution_data)

        # Create stacked resolution chart with selection support
        chart = create_stacked_resolution_chart(
            df=df_resolution,
            title=f"{PROJECT_NAME} Clusters - Resolution Status",
            subtitle=subtitle,
            selected_title=st.session_state.selected_cluster_title,
        )

        # Render chart with selection enabled
        event = st.altair_chart(
            chart,
            width="stretch",
            on_select="rerun",
            key="resolution_chart",
        )

        # Handle bar click selection
        if event and event.selection and "resolution_select" in event.selection:
            points = event.selection.resolution_select
            if points and len(points) > 0:
                clicked_title = points[0].get("title")
                if (
                    clicked_title
                    and clicked_title != st.session_state.selected_cluster_title
                ):
                    st.session_state.selected_cluster_title = clicked_title
                    st.rerun()

elif view_mode == "Sentiment Analysis":
    assert cluster_stats is not None  # Guarded by check above
    sentiment_data = cluster_stats.get("sentiment_by_cluster", [])

    # Filter out noise cluster if not showing
    if not show_noise:
        sentiment_data = [
            s for s in sentiment_data if "Noise" not in s.get("title", "")
        ]

    if not sentiment_data:
        st.warning("Sentiment stats not available.")
    else:
        df_sentiment = pd.DataFrame(sentiment_data)

        # Create stacked sentiment chart with selection support
        chart = create_stacked_sentiment_chart(
            df=df_sentiment,
            title=f"{PROJECT_NAME} Clusters - Sentiment Analysis",
            subtitle=subtitle,
            selected_title=st.session_state.selected_cluster_title,
        )

        # Render chart with selection enabled
        event = st.altair_chart(
            chart,
            width="stretch",
            on_select="rerun",
            key="sentiment_chart",
        )

        # Handle bar click selection
        if event and event.selection and "sentiment_select" in event.selection:
            points = event.selection.sentiment_select
            if points and len(points) > 0:
                clicked_title = points[0].get("title")
                if (
                    clicked_title
                    and clicked_title != st.session_state.selected_cluster_title
                ):
                    st.session_state.selected_cluster_title = clicked_title
                    st.rerun()

elif view_mode == "Tool Use":
    assert cluster_stats is not None  # Guarded by check above
    tool_use_data = cluster_stats.get("tool_use_by_cluster", [])

    # Filter out noise cluster if not showing
    if not show_noise:
        tool_use_data = [t for t in tool_use_data if "Noise" not in t.get("title", "")]

    if not tool_use_data:
        st.warning("Tool use stats not available.")
    else:
        df_tool_use = pd.DataFrame(tool_use_data)

        # Create stacked tool use chart with selection support
        chart = create_stacked_tool_use_chart(
            df=df_tool_use,
            title=f"{PROJECT_NAME} Clusters - Tool Usage",
            subtitle=subtitle,
            selected_title=st.session_state.selected_cluster_title,
        )

        # Render chart with selection enabled
        event = st.altair_chart(
            chart,
            width="stretch",
            on_select="rerun",
            key="tool_use_chart",
        )

        # Handle bar click selection
        if event and event.selection and "tool_use_select" in event.selection:
            points = event.selection.tool_use_select
            if points and len(points) > 0:
                clicked_title = points[0].get("title")
                if (
                    clicked_title
                    and clicked_title != st.session_state.selected_cluster_title
                ):
                    st.session_state.selected_cluster_title = clicked_title
                    st.rerun()

else:  # Simple View
    # Simple bar chart with selection
    chart = create_cluster_bar_chart(
        df=df[["title", "count", "description"]],
        title=f"{PROJECT_NAME} Clusters",
        subtitle=subtitle,
        selected_title=st.session_state.selected_cluster_title,
    )

    # Render chart with selection enabled
    event = st.altair_chart(
        chart,
        width="stretch",
        on_select="rerun",
        key="cluster_chart",
    )

    # Handle bar click selection
    if event and event.selection and "cluster_select" in event.selection:
        points = event.selection.cluster_select
        if points and len(points) > 0:
            clicked_title = points[0].get("title")
            if (
                clicked_title
                and clicked_title != st.session_state.selected_cluster_title
            ):
                st.session_state.selected_cluster_title = clicked_title
                st.rerun()

st.divider()

# Cluster selection dropdown
st.subheader("Cluster Details")

# Add placeholder option for "no selection"
options_with_placeholder = ["-- Select a cluster --"] + titles

# Get current index
if st.session_state.selected_cluster_title in titles:
    current_index = titles.index(st.session_state.selected_cluster_title) + 1
else:
    current_index = 0

selected_option = st.selectbox(
    "Select a cluster to view details:",
    options=options_with_placeholder,
    index=current_index,
)

# Update session state based on selectbox
if selected_option == "-- Select a cluster --":
    if st.session_state.selected_cluster_title is not None:
        st.session_state.selected_cluster_title = None
        st.rerun()
elif selected_option != st.session_state.selected_cluster_title:
    st.session_state.selected_cluster_title = selected_option
    st.rerun()

# Show cluster details only if one is selected
if st.session_state.selected_cluster_title:
    selected_cluster = next(
        (
            c
            for c in cluster_list
            if c["title"] == st.session_state.selected_cluster_title
        ),
        None,
    )

    if selected_cluster:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"### {selected_cluster['title']}")
            st.markdown(f"**Summary:** {selected_cluster['description']}")

        with col2:
            st.metric("Items in cluster", selected_cluster["count"])

        with st.expander("View detailed description", expanded=True):
            st.markdown(selected_cluster["detailed_description"])

        # Show deduplicated questions from cluster
        dedup_data = load_final_deduplicated()
        if dedup_data is None:
            st.info(
                "💡 **Deduplicated FAQ questions not available.**\n\n"
                "To generate them, run:\n"
                "```bash\n"
                "uv run python run_pipeline.py --steps deduplicate_faqs\n"
                "```"
            )
        elif dedup_data:
            cluster_key = selected_cluster["key"]
            cluster_dedup = dedup_data.get("by_cluster", {}).get(cluster_key, {})
            cluster_questions = cluster_dedup.get("questions", [])

            if cluster_questions:
                # Build lookup: conv_id -> canonical_faq_question from cluster_data
                conv_lookup: dict[str, str] = {}
                cluster_items = clusters.get(cluster_key, {}).get("items", [])
                for item in cluster_items:
                    conv_id = item.get("id", "")
                    question = item.get("canonical_faq_question", "")
                    if conv_id and question:
                        conv_lookup[conv_id] = question

                # Configuration
                QUESTIONS_PER_BATCH = 5

                # Checkbox to show presuppositions (default: off)
                show_presuppositions = st.checkbox(
                    "Show presuppositions",
                    value=False,
                    key=f"show_presup_{cluster_key}",
                    help="Presuppositions are implicit assumptions extracted from user questions",
                )

                # Filter questions based on checkbox
                filtered_questions = [
                    q
                    for q in cluster_questions
                    if show_presuppositions or q.get("type") != "presupposition"
                ]

                # Initialize session state for pagination
                state_key = f"questions_shown_{cluster_key}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = QUESTIONS_PER_BATCH

                total_questions = len(filtered_questions)
                questions_to_show = min(st.session_state[state_key], total_questions)

                # Count questions vs presuppositions
                q_count = sum(
                    1 for q in filtered_questions if q.get("type") != "presupposition"
                )
                p_count = sum(
                    1 for q in filtered_questions if q.get("type") == "presupposition"
                )

                if show_presuppositions:
                    st.markdown(
                        f"#### Top questions for selected cluster ({q_count} questions + {p_count} presuppositions)"
                    )
                else:
                    st.markdown(
                        f"#### Top questions for selected cluster ({q_count} questions)"
                    )
                st.caption(
                    "Note: questions are decomposed and deduplicated. Some merging errors are expected."
                )

                # Display deduplicated questions as expanders
                for _idx, q in enumerate(filtered_questions[:questions_to_show]):
                    question_text = q.get("question", "")
                    q_type = q.get("type", "question")
                    unique_convs = q.get("unique_conversations", 0)
                    original_ids = q.get("original_ids", [])

                    # Build expander label with count and type indicator
                    if q_type == "presupposition":
                        label = f"🔹 {question_text} ({unique_convs} conversations)"
                    else:
                        label = f"{question_text} ({unique_convs} conversations)"

                    with st.expander(label, expanded=False):
                        if q_type == "presupposition":
                            st.markdown(
                                '<span class="presupposition-label">This is a presupposition (implicit assumption)</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("**Original questions:**")
                        for conv_id in original_ids:
                            # Look up original question from cluster_data
                            original_question = conv_lookup.get(
                                conv_id, f"Conversation {conv_id[:8]}..."
                            )
                            link = f"/Conversation_Viewer?id={conv_id}"
                            st.markdown(
                                f'<div class="original-question">• <a href="{link}" target="_blank" class="question-link">{original_question} ↗</a></div>',
                                unsafe_allow_html=True,
                            )

                # Show "Load more" button if there are more questions
                if questions_to_show < total_questions:
                    remaining = total_questions - questions_to_show
                    if st.button(
                        f"Load more ({remaining} remaining)",
                        key=f"load_more_{cluster_key}",
                    ):
                        st.session_state[state_key] += QUESTIONS_PER_BATCH
                        st.rerun()
else:
    st.info(
        "Click on a bar in the chart above or select from the dropdown to view cluster details."
    )
