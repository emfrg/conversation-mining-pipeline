"""Usage Statistics - Generate comprehensive statistics from raw chat history.

Standalone script for analyzing chat history data. Not part of the main pipeline.
Run directly: python -m src.analysis.get_usage_statistics

Analyzes raw chat history and produces statistics including:
- Conversation counts and message distributions
- Message length analysis
- Feature usage (tools, citations)
- Temporal patterns (by date, hour, weekday)

Reads:
    config.paths.raw_data - Raw chat history JSON file.

Writes:
    data/output/chat_history_stats.json - Statistics JSON file.
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any

from config import config
from src.utils.helpers import read_json_file

# UUID v4 pattern
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


def parse_timestamp(ts_string: str) -> datetime:
    """Parse ISO format timestamp string to datetime.

    Args:
        ts_string: ISO format timestamp string.

    Returns:
        Parsed datetime object.
    """
    try:
        return datetime.fromisoformat(ts_string.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromisoformat(ts_string[:19])


def safe_avg(lst: list) -> float:
    """Calculate average, returning 0 for empty lists.

    Args:
        lst: List of numbers.

    Returns:
        Average value, or 0 if list is empty.
    """
    return sum(lst) / len(lst) if lst else 0


def safe_median(lst: list) -> float:
    """Calculate median, returning 0 for empty lists.

    Args:
        lst: List of numbers.

    Returns:
        Median value, or 0 if list is empty.
    """
    if not lst:
        return 0
    sorted_lst: list[float] = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    return sorted_lst[mid]


def analyze_messages(messages: list) -> dict[str, Any]:
    """Analyze a list of messages from a single conversation.

    Args:
        messages: List of message dicts with role and content.

    Returns:
        Dict with message counts, lengths, and feature flags.
    """
    stats: dict[str, Any] = {
        "total": 0,
        "user": 0,
        "assistant": 0,
        "user_message_lengths": [],
        "assistant_message_lengths": [],
        "has_tool_use": False,
        "has_citations": False,
        "tool_names": [],
    }

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "user":
            # Only count user messages with string content (actual user input)
            # Skip tool_result containers - they're not real user messages
            if isinstance(content, str):
                stats["total"] += 1
                stats["user"] += 1
                stats["user_message_lengths"].append(len(content))

        elif role == "assistant":
            has_text_response = False
            text_len = 0

            if isinstance(content, str):
                has_text_response = True
                stats["assistant_message_lengths"].append(len(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_len += len(item.get("text", ""))
                        elif item.get("type") == "tool_use":
                            # Track tool usage (keep this for feature stats)
                            stats["has_tool_use"] = True
                            stats["tool_names"].append(item.get("name", "unknown"))
                if text_len > 0:
                    has_text_response = True
                    stats["assistant_message_lengths"].append(text_len)

            # Only count as assistant message if it has actual response text
            # Skip messages that are just tool_use or document blocks
            if has_text_response:
                stats["total"] += 1
                stats["assistant"] += 1

            if msg.get("citations"):
                stats["has_citations"] = True

    return stats


def calculate_statistics(chat_history: dict) -> dict:
    """Calculate comprehensive statistics from chat history.

    Args:
        chat_history: Dict of conversations keyed by conversation_id.

    Returns:
        Dict with overview, message_counts, conversation_length,
        message_lengths, features, and temporal_distribution sections.
    """
    # Filter out invalid UUIDs
    valid_conversations = {k: v for k, v in chat_history.items() if is_valid_uuid(k)}
    skipped = [k for k in chat_history if not is_valid_uuid(k)]
    if skipped:
        print(f"Skipped {len(skipped)} conversations with invalid IDs: {skipped}")

    total_conversations = len(valid_conversations)

    # Initialize aggregators
    all_timestamps = []
    total_messages = 0
    total_user_messages = 0
    total_assistant_messages = 0

    messages_per_conversation = []
    user_messages_per_conversation = []

    all_user_message_lengths = []
    all_assistant_message_lengths = []

    conversations_with_tools = 0
    conversations_with_citations = 0
    tool_usage_counter: Counter[str] = Counter()

    # Date-based tracking
    conversations_by_date: defaultdict[date, int] = defaultdict(int)
    conversations_by_hour: defaultdict[int, int] = defaultdict(int)
    conversations_by_weekday: defaultdict[str, int] = defaultdict(int)

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    # Process each conversation
    for _conv_id, conv_data in valid_conversations.items():
        ts_str = conv_data.get("timestamp", "")
        if ts_str:
            try:
                ts = parse_timestamp(ts_str)
                all_timestamps.append(ts)
                conversations_by_date[ts.date()] += 1
                conversations_by_hour[ts.hour] += 1
                conversations_by_weekday[weekday_names[ts.weekday()]] += 1
            except (ValueError, TypeError):
                pass

        messages = conv_data.get("messages", [])
        msg_stats = analyze_messages(messages)

        total_messages += msg_stats["total"]
        total_user_messages += msg_stats["user"]
        total_assistant_messages += msg_stats["assistant"]

        messages_per_conversation.append(msg_stats["total"])
        user_messages_per_conversation.append(msg_stats["user"])

        all_user_message_lengths.extend(msg_stats["user_message_lengths"])
        all_assistant_message_lengths.extend(msg_stats["assistant_message_lengths"])

        if msg_stats["has_tool_use"]:
            conversations_with_tools += 1
            tool_usage_counter.update(msg_stats["tool_names"])

        if msg_stats["has_citations"]:
            conversations_with_citations += 1

    # Calculate time range
    if all_timestamps:
        all_timestamps.sort()
        first_conversation = all_timestamps[0]
        last_conversation = all_timestamps[-1]
        date_range_days = (last_conversation - first_conversation).days + 1
    else:
        first_conversation = None
        last_conversation = None
        date_range_days = 0

    # Build results dictionary
    results = {
        "overview": {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "first_conversation": (
                first_conversation.isoformat() if first_conversation else "N/A"
            ),
            "last_conversation": (
                last_conversation.isoformat() if last_conversation else "N/A"
            ),
            "date_range_days": date_range_days,
            "avg_conversations_per_day": (
                round(total_conversations / date_range_days, 2)
                if date_range_days > 0
                else 0
            ),
        },
        "message_counts": {
            "user_messages": total_user_messages,
            "assistant_messages": total_assistant_messages,
        },
        "conversation_length": {
            "avg_messages_per_conversation": round(
                safe_avg(messages_per_conversation), 2
            ),
            "median_messages_per_conversation": round(
                safe_median(messages_per_conversation), 2
            ),
            "min_messages": (
                min(messages_per_conversation) if messages_per_conversation else 0
            ),
            "max_messages": (
                max(messages_per_conversation) if messages_per_conversation else 0
            ),
            "avg_user_messages_per_conversation": round(
                safe_avg(user_messages_per_conversation), 2
            ),
        },
        "message_lengths": {
            "avg_user_message_chars": round(safe_avg(all_user_message_lengths), 2),
            "median_user_message_chars": round(
                safe_median(all_user_message_lengths), 2
            ),
            "avg_assistant_message_chars": round(
                safe_avg(all_assistant_message_lengths), 2
            ),
            "median_assistant_message_chars": round(
                safe_median(all_assistant_message_lengths), 2
            ),
        },
        "features": {
            "conversations_with_tool_use": conversations_with_tools,
            "tool_use_percentage": (
                round(100 * conversations_with_tools / total_conversations, 2)
                if total_conversations > 0
                else 0
            ),
            "conversations_with_citations": conversations_with_citations,
            "citations_percentage": (
                round(100 * conversations_with_citations / total_conversations, 2)
                if total_conversations > 0
                else 0
            ),
            "top_tools_used": dict(tool_usage_counter.most_common(10)),
        },
        "temporal_distribution": {
            "busiest_dates": {
                str(k): v
                for k, v in sorted(
                    conversations_by_date.items(), key=lambda x: x[1], reverse=True
                )[:10]
            },
            "by_hour": dict(sorted(conversations_by_hour.items())),
            "by_weekday": {
                day: conversations_by_weekday.get(day, 0) for day in weekday_names
            },
        },
    }

    return results


def format_report(stats: dict) -> str:
    """Format statistics into a human-readable report.

    Args:
        stats: Statistics dict from calculate_statistics().

    Returns:
        Formatted multi-line string report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CHAT HISTORY STATISTICS REPORT")
    lines.append("=" * 60)

    # Overview
    lines.append("\nOVERVIEW")
    lines.append("-" * 40)
    overview = stats["overview"]
    lines.append(f"  Total Conversations:      {overview['total_conversations']:,}")
    lines.append(f"  Total Messages:           {overview['total_messages']:,}")
    lines.append(f"  First Conversation:       {overview['first_conversation']}")
    lines.append(f"  Last Conversation:        {overview['last_conversation']}")
    lines.append(f"  Date Range:               {overview['date_range_days']} days")
    lines.append(f"  Avg Conversations/Day:    {overview['avg_conversations_per_day']}")

    # Message Counts
    lines.append("\nMESSAGE COUNTS")
    lines.append("-" * 40)
    msg_counts = stats["message_counts"]
    lines.append(f"  User Messages:            {msg_counts['user_messages']:,}")
    lines.append(f"  Assistant Messages:       {msg_counts['assistant_messages']:,}")

    # Conversation Length
    lines.append("\nCONVERSATION LENGTH")
    lines.append("-" * 40)
    conv_len = stats["conversation_length"]
    lines.append(
        f"  Avg Messages/Conversation:    {conv_len['avg_messages_per_conversation']}"
    )
    lines.append(
        f"  Median Messages/Conversation: {conv_len['median_messages_per_conversation']}"
    )
    lines.append(f"  Min Messages:                 {conv_len['min_messages']}")
    lines.append(f"  Max Messages:                 {conv_len['max_messages']}")
    lines.append(
        f"  Avg User Msgs/Conversation:   {conv_len['avg_user_messages_per_conversation']}"
    )

    # Message Lengths
    lines.append("\nMESSAGE LENGTHS (characters)")
    lines.append("-" * 40)
    msg_len = stats["message_lengths"]
    lines.append(f"  Avg User Message:         {msg_len['avg_user_message_chars']}")
    lines.append(f"  Median User Message:      {msg_len['median_user_message_chars']}")
    lines.append(
        f"  Avg Assistant Message:    {msg_len['avg_assistant_message_chars']}"
    )
    lines.append(
        f"  Median Assistant Message: {msg_len['median_assistant_message_chars']}"
    )

    # Features
    lines.append("\nFEATURE USAGE")
    lines.append("-" * 40)
    features = stats["features"]
    lines.append(
        f"  Conversations with Tool Use:  {features['conversations_with_tool_use']} ({features['tool_use_percentage']}%)"
    )
    lines.append(
        f"  Conversations with Citations: {features['conversations_with_citations']} ({features['citations_percentage']}%)"
    )
    if features["top_tools_used"]:
        lines.append("  Top Tools Used:")
        for tool, count in features["top_tools_used"].items():
            lines.append(f"    - {tool}: {count}")

    # Temporal Distribution
    lines.append("\nTEMPORAL DISTRIBUTION")
    lines.append("-" * 40)
    temporal = stats["temporal_distribution"]

    lines.append("  By Weekday:")
    for day, count in temporal["by_weekday"].items():
        bar = "#" * (count // max(1, max(temporal["by_weekday"].values()) // 20))
        lines.append(f"    {day:10} {count:4} {bar}")

    lines.append("\n  By Hour:")
    for hour, count in temporal["by_hour"].items():
        bar = "#" * (count // max(1, max(temporal["by_hour"].values()) // 20))
        lines.append(f"    {hour:02}:00  {count:4} {bar}")

    lines.append("\n  Top 10 Busiest Dates:")
    for date_str, count in list(temporal["busiest_dates"].items())[:10]:
        lines.append(f"    {date_str}: {count} conversations")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def save_statistics(stats: dict, output_path: str) -> None:
    """Save statistics to JSON file.

    Args:
        stats: Statistics dict to save.
        output_path: Path to output JSON file.
    """
    json_stats = json.loads(json.dumps(stats, default=str))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_stats, f, indent=2, ensure_ascii=False)


def run() -> dict:
    """Run usage statistics analysis.

    Loads raw chat history, calculates comprehensive statistics,
    prints a formatted report, and saves results to JSON.

    Returns:
        Statistics dict with overview, message_counts, conversation_length,
        message_lengths, features, and temporal_distribution sections.
    """
    raw_data_path = config["paths"]["raw_data"]
    output_path = "data/output/chat_history_stats.json"

    # Load chat history
    chat_history = read_json_file(raw_data_path)
    print(f"  Loaded {len(chat_history)} conversations from {raw_data_path}")

    # Calculate statistics
    stats = calculate_statistics(chat_history)

    # Print formatted report
    report = format_report(stats)
    print(report)

    # Save JSON stats
    save_statistics(stats, output_path)
    print(f"\n  Saved statistics to {output_path}")

    return stats


if __name__ == "__main__":
    # Get paths from config
    raw_data_path = config["paths"]["raw_data"]
    output_stats_path = "data/output/chat_history_stats.json"

    # Load chat history
    chat_history = read_json_file(raw_data_path)
    print(f"Loaded {len(chat_history)} conversations from {raw_data_path}")

    # Calculate statistics
    stats = calculate_statistics(chat_history)

    # Print formatted report
    report = format_report(stats)
    print(report)

    # Save JSON stats
    save_statistics(stats, output_stats_path)
    print(f"\nSaved statistics to {output_stats_path}")

    # Print sample
    print("\n### Sample statistics (overview) ###\n")
    print(json.dumps(stats["overview"], indent=2, ensure_ascii=False))
