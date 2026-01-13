"""Conversation Viewer page - View individual conversations with chat UI."""

import streamlit as st

from streamlit_app.utils.data_loader import load_clean_conversations

st.title("Conversation Viewer")

# Load conversations
conversations = load_clean_conversations()

if conversations is None:
    st.error("Conversations data not found.")
    st.stop()

# Get conversation ID from query params
query_params = st.query_params
conversation_id = query_params.get("id", None)


def is_tool_output(content: str) -> bool:
    """Check if content looks like raw tool output (XML/document tags)."""
    if not content:
        return False
    # Check for common tool output patterns
    tool_patterns = [
        "<document index=",
        "<source>",
        "<document_content>",
        "<document_context>",
    ]
    return any(pattern in content for pattern in tool_patterns)


if conversation_id:
    conversation = conversations.get(conversation_id)

    if conversation is None:
        st.error(f"Conversation not found: {conversation_id}")
        if st.button("← Back to Conversation Viewer"):
            st.query_params.clear()
            st.rerun()
        st.stop()

    # Header with back button and download
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Back to Conversation Viewer"):
            st.query_params.clear()
            st.rerun()

    with col2:
        timestamp = conversation.get("timestamp", "N/A")
        if timestamp != "N/A":
            timestamp_display = (
                timestamp.split("T")[0] if "T" in timestamp else timestamp
            )
            st.caption(f"Date: {timestamp_display}")

    with col3:
        # Generate markdown transcript for download
        def generate_markdown_transcript(conv: dict) -> str:
            lines = [
                "# Conversation Transcript",
                "",
                f"**Timestamp:** {conv.get('timestamp', 'N/A')}",
                "",
                "---",
                "",
            ]
            for turn in conv.get("turns", []):
                role = turn.get("role", "")
                turn_type = turn.get("type", "")
                content = turn.get("content", "")

                if role == "user":
                    lines.append(f"### User\n{content}\n")
                elif role == "assistant":
                    # Include tool call indicator
                    if turn_type == "tool":
                        tool_name = turn.get("tool", "unknown")
                        lines.append(f"*[Tool: {tool_name}]*\n")
                    # Skip raw tool output content
                    elif is_tool_output(content):
                        continue
                    elif content:
                        lines.append(f"### Assistant\n{content}\n")
            return "\n".join(lines)

        markdown_content = generate_markdown_transcript(conversation)
        # Use full timestamp for filename (replace colons for filesystem compatibility)
        ts = conversation.get("timestamp", "unknown")
        file_ts = ts.replace(":", "-") if ts else "unknown"
        st.download_button(
            label="Download",
            data=markdown_content,
            file_name=f"conversation_{file_ts}.md",
            mime="text/markdown",
        )

    st.divider()

    # Display conversation using chat UI
    for turn in conversation.get("turns", []):
        role = turn.get("role", "")
        turn_type = turn.get("type", "")
        content = turn.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            # Show tool call indicator with gear icon
            if turn_type == "tool":
                tool_name = turn.get("tool", "unknown")
                with st.chat_message("assistant", avatar="🔧"):
                    st.caption(f"Tool called: {tool_name}")
            # Skip raw tool output content (XML documents)
            elif is_tool_output(content):
                continue
            elif content:
                with st.chat_message("assistant"):
                    st.markdown(content)

else:
    # No conversation selected - show info and example links
    st.info(
        "Select a conversation from the **Cluster Analysis** page, or choose an example below:"
    )

    st.markdown("#### Example Conversations")
    example_ids = list(conversations.keys())[:5]
    for i, cid in enumerate(example_ids):
        conv = conversations[cid]
        # Get first user message as preview
        first_user_msg = ""
        for turn in conv.get("turns", []):
            if turn.get("role") == "user":
                first_user_msg = turn.get("content", "")[:60]
                if len(turn.get("content", "")) > 60:
                    first_user_msg += "..."
                break

        if st.button(first_user_msg or f"Conversation {i + 1}", key=f"example_{cid}"):
            st.query_params["id"] = cid
            st.rerun()
