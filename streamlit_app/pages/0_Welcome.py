"""Welcome page for the FAQ Analytics Dashboard."""

import streamlit as st

from streamlit_app.config import APP_TITLE, PROJECT_NAME

# Main page content
st.title(APP_TITLE)

st.markdown(
    f"""
Welcome to the FAQ Analytics Dashboard. This interactive report provides insights into
1) how visitors used the {PROJECT_NAME} chatbot and
2) most common user queries grouped by topic

**Pages:**

- **General Report** - Overview report and insights
- **Usage Statistics** - Usage metrics and trends
- **Cluster Analysis** - Interactive cluster visualization
- **Conversation Viewer** - Browse individual conversations

Use the sidebar to navigate between pages.
"""
)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    st.markdown("Select a page from above to explore the analytics.")
