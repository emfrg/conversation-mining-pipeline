"""Step 1: Clean Data - Parse raw chat history into structured conversations.

Parses raw chat history JSON and extracts structured conversation data:
- Handles multiple content types: string messages, document lists (RAG), tool usage
- Extracts FAQ IDs from document sources and citations
- Generates conversation transcripts in human-readable format

Reads:
    config.paths.raw_data - Raw chat history JSON file.

Writes:
    config.paths.clean_conversations - Structured conversation data.
    config.paths.transcripts_dir - Human-readable transcript files.
"""

import json
import os
import re
from typing import Any

from config import config
from src.utils.helpers import read_json_file

# UUID v4 pattern (for validity check).
# NOTE: this could be different for different datasets.
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def is_valid_uuid(value: str) -> bool:
    """Check if a string is a valid UUID v4.

    Args:
        value: String to validate.

    Returns:
        True if string matches UUID v4 pattern, False otherwise.
    """
    return bool(UUID_PATTERN.match(value))


def extract_faq_id_from_source(source: dict) -> str | None:
    """Extract FAQ ID from a RAG document source object.

    Args:
        source: Document source dict with nested content[0].text containing JSON.

    Returns:
        FAQ ID string if found, None otherwise.
    """
    try:
        content = source.get("content", [])
        if content and len(content) > 0:
            text = content[0].get("text", "")
            faq_data = json.loads(text)
            faq_id: str | None = faq_data.get("id")
            return faq_id
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def extract_faq_id_from_citation(citation: dict) -> str | None:
    """Extract FAQ ID from a citation object.

    Args:
        citation: Citation dict with text field containing JSON.

    Returns:
        FAQ ID string if found, None otherwise.
    """
    try:
        text = citation.get("text", "")
        faq_data = json.loads(text)
        faq_id: str | None = faq_data.get("id")
        return faq_id
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def extract_conversation(conversation_id: str, conversation: dict) -> dict:
    """Parse a single conversation into a clean structure.

    Handles content types: string, document list, tool_use list.
    Extracts FAQ IDs from document sources and citations.

    Args:
        conversation_id: UUID of the conversation.
        conversation: Raw conversation dict with messages array.

    Returns:
        Clean conversation dict with id, timestamp, turns, and metadata.
    """
    turns: list[dict[str, Any]] = []
    faq_ids_retrieved: set[str] = set()
    faq_ids_cited: set[str] = set()
    num_user_messages = 0

    messages = conversation.get("messages", [])

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        citations = message.get("citations", [])

        # Extract cited FAQ IDs
        for citation in citations:
            faq_id = extract_faq_id_from_citation(citation)
            if faq_id:
                faq_ids_cited.add(faq_id)

        if role == "user":
            # User messages: content is a string except is it's tool result
            if isinstance(content, str):
                # Add to user message to turns list
                turns.append({"role": "user", "content": content})
                num_user_messages += 1
            # Handle tool result (pass for now)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "tool_result":
                        # NOTE: Skip tool results in output
                        pass

        elif role == "assistant":
            if isinstance(content, str):
                # Add assistant response to turns list
                turns.append(
                    {"role": "assistant", "type": "response", "content": content}
                )
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type")

                    if item_type == "document":
                        # RAG retrieved document - extract FAQ ID
                        source = item.get("source", {})
                        faq_id = extract_faq_id_from_source(source)
                        if faq_id:
                            faq_ids_retrieved.add(faq_id)
                        # NOTE: Ignore actual retrievals for now

                    elif item_type == "tool_use":
                        # Tool usage
                        tool_name = item.get("name", "unknown")
                        # Add tool use to turns list
                        turns.append(
                            {"role": "assistant", "type": "tool", "tool": tool_name}
                        )

    return {
        "id": conversation_id,
        "timestamp": conversation.get("timestamp"),
        "turns": turns,
        "metadata": {
            "faq_ids_retrieved": list(faq_ids_retrieved),
            "faq_ids_cited": list(faq_ids_cited),
            "num_turns": len(turns),
            "num_user_messages": num_user_messages,
            # NOTE: this can be extended with more metadata in the future
        },
    }


def extract_all_conversations(raw_data: dict) -> dict:
    """Extract all conversations from raw data.

    Filters out conversations with invalid (non-UUID) IDs.

    Args:
        raw_data: Dict of raw conversations keyed by conversation_id.

    Returns:
        Dict of clean conversations keyed by conversation_id.
    """
    result = {}
    skipped = []
    for conversation_id, conversation in raw_data.items():
        if not is_valid_uuid(conversation_id):
            skipped.append(conversation_id)
            continue
        result[conversation_id] = extract_conversation(conversation_id, conversation)

    if skipped:
        print(f"  Skipped {len(skipped)} conversations with invalid IDs: {skipped}")

    return result


def conversation_to_string(conversation: dict) -> str:
    """Convert a clean conversation to human-readable string format.

    Args:
        conversation: Clean conversation dict with turns array.

    Returns:
        Formatted string with conversation transcript.
    """
    # NOTE: not strictly needed,
    # # but could be a place to adjust if the item is not a "conversation" per se.
    lines = ["--- START OF CONVERSATION ---"]

    for turn in conversation["turns"]:
        role = turn["role"].upper()
        if turn.get("type") == "tool":
            lines.append(f"{role}: [Used tool: {turn['tool']}]")
        else:
            lines.append(f"{role}: {turn.get('content', '')}")

    return "\n\n".join(lines) + "\n\n--- END OF CONVERSATION ---"


def save_clean_data(data: dict, output_path: str) -> None:
    """Save extracted data to JSON file.

    Args:
        data: Dict of clean conversations to save.
        output_path: Path to output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_transcript(conversation: dict, transcripts_dir: str) -> tuple[str, bool]:
    """Save conversation transcript to a text file. Skips if already exists.

    Args:
        conversation: Clean conversation dict.
        transcripts_dir: Directory to save transcript files.

    Returns:
        Tuple of (filepath, was_created) where was_created is True if newly created.
    """
    os.makedirs(transcripts_dir, exist_ok=True)
    filename = f"{conversation['id']}_transcript.txt"
    filepath = os.path.join(transcripts_dir, filename)

    # Skip if transcript already exists
    if os.path.exists(filepath):
        return filepath, False

    transcript_text = conversation_to_string(conversation)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return filepath, True


def run() -> dict:
    """Run the clean_data pipeline step.

    Reads raw chat history, extracts conversations, saves transcripts.

    Returns:
        Dict of clean conversations keyed by conversation_id.
    """
    raw_data_path = config["paths"]["raw_data"]
    clean_data_path = config["paths"]["clean_conversations"]
    transcripts_dir = config["paths"]["transcripts_dir"]
    recreate = config.get("recreate", False)

    # Skip if output already exists
    if not recreate and os.path.exists(clean_data_path):
        print(f"  Skipping: {clean_data_path} already exists")
        return read_json_file(clean_data_path)

    # Read raw data
    raw_data = read_json_file(raw_data_path)
    print(f"  Loaded {len(raw_data)} conversations")

    # Extract all conversations
    clean_data = extract_all_conversations(raw_data)

    # Save transcripts and add path to metadata
    created_count = 0
    skipped_count = 0
    for _conv_id, conversation in clean_data.items():
        transcript_path, was_created = save_transcript(conversation, transcripts_dir)
        conversation["metadata"]["transcript_path"] = transcript_path
        if was_created:
            created_count += 1
        else:
            skipped_count += 1

    if created_count > 0:
        print(f"  Created {created_count} new transcripts in {transcripts_dir}/")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} existing transcripts")

    # Save clean data
    save_clean_data(clean_data, clean_data_path)
    print(f"  Saved clean data to {clean_data_path}")

    return clean_data


if __name__ == "__main__":
    clean_data = run()

    # Print sample for debugging
    sample_id = list(clean_data.keys())[0]
    sample = clean_data[sample_id]
    print("\n### Sample conversation (JSON) ###\n")
    print(json.dumps(sample, indent=2, ensure_ascii=False))
