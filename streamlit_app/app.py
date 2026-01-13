"""Main entry point for the FAQ Analytics Dashboard."""

import sys
from pathlib import Path

# Add project root to path so imports work correctly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from streamlit_app.config import APP_ICON, APP_TITLE  # noqa: E402

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages with custom labels (st.Page requires actual emojis or :material/ icons, not shortcodes)
welcome_page = st.Page("pages/0_Welcome.py", title="Welcome", default=True)
general_report = st.Page(
    "pages/1_General_Report.py", title="General Report", icon=":material/description:"
)
usage_stats = st.Page(
    "pages/2_Usage_Statistics.py", title="Usage Statistics", icon=":material/bar_chart:"
)
cluster_analysis = st.Page(
    "pages/3_Cluster_Analysis.py", title="Cluster Analysis", icon=":material/hub:"
)
conversation_viewer = st.Page(
    "pages/4_Conversation_Viewer.py",
    title="Conversation Viewer",
    icon=":material/chat:",
)

# Set up navigation
pg = st.navigation(
    [welcome_page, general_report, usage_stats, cluster_analysis, conversation_viewer]
)

# Run the selected page
pg.run()
