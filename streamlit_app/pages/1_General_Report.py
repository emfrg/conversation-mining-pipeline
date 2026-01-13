"""General Report page - LLM-generated executive summary."""

import streamlit as st

from streamlit_app.config import CLUSTER_DATA_PATH, VIS_OUTPUTS_PATH
from streamlit_app.utils.data_loader import load_executive_report

st.title("General Report")

# Load report data
report = load_executive_report()

if report is None:
    st.warning(
        "Executive report not found. Please run the report generator first:\n\n"
        "```bash\nuv run python -m src.analysis.executive_report\n```"
    )

    # Show available visualizations as fallback
    st.divider()
    st.header("Useful Visualizations")

    # Word cloud
    wordcloud_path = CLUSTER_DATA_PATH / "vis" / "tag_wordcloud.png"
    if wordcloud_path.exists():
        st.subheader("Issue Tags Word Cloud")
        st.image(str(wordcloud_path), width="stretch")
    else:
        st.info(
            "Word cloud not available. Run the tagger to generate tags:\n\n"
            "```bash\nuv run python -m run_pipeline --tag-only\n```"
        )

    # Clustering scatterplot
    scatter_path = CLUSTER_DATA_PATH / "vis" / "faq_clusters_scatter_kmeans.png"
    if scatter_path.exists():
        st.subheader("Cluster Visualization")
        st.image(str(scatter_path), width="stretch")

    # Chi-square proportions
    chi_square_path = VIS_OUTPUTS_PATH / "chi_square_proportions.png"
    if chi_square_path.exists():
        st.subheader("Chi-Square Proportions")
        st.image(str(chi_square_path), width="stretch")

    # Correlation heatmap
    heatmap_path = VIS_OUTPUTS_PATH / "correlation_heatmap.png"
    if heatmap_path.exists():
        st.subheader("Correlation Heatmap")
        st.image(str(heatmap_path), width="stretch")

    st.stop()

# Display generation timestamp
metadata = report.get("_metadata", {})
if metadata.get("generated_at"):
    st.caption(f"Report generated: {metadata['generated_at'][:19].replace('T', ' ')}")

st.divider()

# =============================================================================
# Executive Summary
# =============================================================================
st.header("Executive Summary")
st.markdown(report.get("executive_summary", "No summary available."))

st.divider()

# =============================================================================
# Issue Tags Word Cloud (static image)
# =============================================================================
wordcloud_path = CLUSTER_DATA_PATH / "vis" / "tag_wordcloud.png"
clusters_scatter_path = CLUSTER_DATA_PATH / "vis" / "faq_clusters_scatter_kmeans.png"

if wordcloud_path.exists():
    st.image(str(wordcloud_path), width="stretch")
    st.caption("Issue tags")
else:
    st.info(
        "Word cloud not available. Run the tagger to generate tags:\n\n"
        "```bash\nuv run python -m run_pipeline --tag-only\n```"
    )

st.divider()

# =============================================================================
# Key Findings
# =============================================================================
st.header("Key Findings")
findings = report.get("key_findings", [])
for finding in findings:
    st.markdown(f"- {finding}")

st.divider()

# =============================================================================
# Top Clusters Analysis
# =============================================================================
st.header("Top Clusters Analysis")

# Image of clusters scatter plot
if clusters_scatter_path.exists():
    st.image(str(clusters_scatter_path), width="stretch")


clusters_analysis = report.get("top_clusters_analysis", [])

for cluster in clusters_analysis:
    with st.expander(f"**{cluster.get('cluster_title', 'Untitled')}**", expanded=False):
        st.markdown(f"**Insight:** {cluster.get('insight', 'N/A')}")
        st.markdown(f"**Recommendation:** {cluster.get('recommendation', 'N/A')}")


st.divider()

# =============================================================================
# Recommendations
# =============================================================================
st.header("Recommendations")

recommendations = report.get("recommendations", [])
for i, rec in enumerate(recommendations, 1):
    st.markdown(f"**{i}.** {rec}")

st.divider()

# =============================================================================
# Developer Insights
# =============================================================================
st.header("Developer Insights")
metrics = report.get("metrics_summary", {})

st.subheader("Strengths")
for strength in metrics.get("strengths", []):
    st.success(strength)

st.subheader("Areas of Concern")
for concern in metrics.get("concerns", []):
    st.warning(concern)
