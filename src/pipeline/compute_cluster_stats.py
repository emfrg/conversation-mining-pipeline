"""Step 8: Compute Cluster Stats - Pre-aggregate metrics for visualizations.

Computes resolution status, sentiment, and tool usage statistics
for each cluster. These pre-computed stats are used by both the
visualization step and the Streamlit dashboard.

Reads:
    config.experiment.dir/clusters_named.json - Named clusters from Step 6.
    config.paths.issue_reports - Issue reports with resolution/sentiment.
    config.paths.clean_conversations - Clean conversations for tool usage.

Writes:
    config.experiment.dir/cluster_stats.json - Pre-computed stats for visualizations.
"""

import json
import os

from config import config
from src.utils.helpers import filter_by_confidence, read_json_file


def compute_resolution_stats(clusters_data: dict, issue_reports: dict) -> list[dict]:
    """Compute resolution status breakdown per cluster.

    Args:
        clusters_data: Parsed clusters_named.json data.
        issue_reports: Parsed issue_reports.json data.

    Returns:
        List of dicts with title, resolved, partially_resolved, unresolved, total.
    """
    results = []

    for _cluster_key, cluster_info in clusters_data["clusters"].items():
        title = cluster_info["cluster_title"]
        items = cluster_info.get("items", [])

        counts = {"resolved": 0, "partially_resolved": 0, "unresolved": 0}

        for item in items:
            issue_id = item.get("id")
            if issue_id and issue_id in issue_reports:
                status = issue_reports[issue_id].get("resolution_status", "unknown")
                if status in counts:
                    counts[status] += 1

        results.append(
            {
                "title": title,
                "resolved": counts["resolved"],
                "partially_resolved": counts["partially_resolved"],
                "unresolved": counts["unresolved"],
                "total": sum(counts.values()),
            }
        )

    # Sort by total descending
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def compute_sentiment_stats(clusters_data: dict, issue_reports: dict) -> list[dict]:
    """Compute sentiment breakdown per cluster.

    Args:
        clusters_data: Parsed clusters_named.json data.
        issue_reports: Parsed issue_reports.json data.

    Returns:
        List of dicts with title, negative, other, total.
    """
    results = []

    for _cluster_key, cluster_info in clusters_data["clusters"].items():
        title = cluster_info["cluster_title"]
        items = cluster_info.get("items", [])

        counts = {"negative": 0, "other": 0}

        for item in items:
            issue_id = item.get("id")
            if issue_id and issue_id in issue_reports:
                sentiment = issue_reports[issue_id].get("user_sentiment", "unknown")
                if sentiment == "negative":
                    counts["negative"] += 1
                else:
                    counts["other"] += 1

        results.append(
            {
                "title": title,
                "negative": counts["negative"],
                "other": counts["other"],
                "total": sum(counts.values()),
            }
        )

    # Sort by total descending
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def _get_tools_used(conversation: dict) -> set:
    """Extract unique tools used in a conversation.

    Args:
        conversation: Conversation dict with turns.

    Returns:
        Set of tool names used.
    """
    tools = set()
    for turn in conversation.get("turns", []):
        if turn.get("type") == "tool":
            tool_name = turn.get("tool")
            if tool_name:
                tools.add(tool_name)
    return tools


# =============================================================================
# CUSTOMIZE FOR YOUR CHATBOT
# =============================================================================
# These tool names are specific to your chatbot implementation.
# Edit this list to match your chatbot's tool names from chat_history.json.
# Keep "no_tool" for conversations that don't use any tools.
# =============================================================================
TOOL_ORDER = [
    "no_tool",
    "rag_tool",
    "zone_checker",
    "unanswered_question_tool",
    "feedback_tool",
]


def compute_tool_use_stats(clusters_data: dict, conversations: dict) -> list[dict]:
    """Compute weighted tool usage per cluster.

    Each conversation contributes a total weight of 1, distributed equally
    across all tools used. This ensures bar length = conversation count.

    Args:
        clusters_data: Parsed clusters_named.json data.
        conversations: Parsed clean_conversations.json data.

    Returns:
        List of dicts with title, no_tool, rag_tool, zone_checker,
        unanswered_question_tool, feedback_tool, total.
    """
    results = []

    for _cluster_key, cluster_info in clusters_data["clusters"].items():
        title = cluster_info["cluster_title"]
        items = cluster_info.get("items", [])

        weighted_counts: dict[str, float] = {tool: 0.0 for tool in TOOL_ORDER}
        conversation_count = 0

        for item in items:
            conv_id = item.get("id")
            if conv_id and conv_id in conversations:
                conversation_count += 1
                tools_used = _get_tools_used(conversations[conv_id])

                if not tools_used:
                    weighted_counts["no_tool"] += 1.0
                else:
                    weight = 1.0 / len(tools_used)
                    for tool in tools_used:
                        if tool in weighted_counts:
                            weighted_counts[tool] += weight

        result = {"title": title, "total": conversation_count}
        for tool in TOOL_ORDER:
            result[tool] = weighted_counts[tool]
        results.append(result)

    # Sort by total descending
    results.sort(key=lambda x: x["total"], reverse=True)
    return results


def has_extra_agent_fields(issue_reports: dict) -> dict[str, bool]:
    """Check which extra agent fields exist in issue reports.

    Args:
        issue_reports: Parsed issue_reports.json data.

    Returns:
        Dict with boolean flags for each field type.
    """
    sample_issue: dict = next(iter(issue_reports.values()), {})
    return {
        "resolution_status": "resolution_status" in sample_issue,
        "user_sentiment": "user_sentiment" in sample_issue,
    }


def run() -> str:
    """Run the compute_cluster_stats pipeline step.

    Computes resolution, sentiment, and tool usage stats for each cluster.
    Saves results to cluster_stats.json.

    Returns:
        Path to the saved cluster_stats.json file.
    """
    experiment_dir = config["experiment"]["dir"]
    clusters_path = os.path.join(experiment_dir, "clusters_named.json")
    output_path = os.path.join(experiment_dir, "cluster_stats.json")
    recreate = config.get("recreate", False)

    # Skip if cluster_stats.json already exists (unless --recreate)
    if not recreate and os.path.exists(output_path):
        print(f"  Skipping: {output_path} already exists")
        return output_path

    # Load input data
    print(f"  Loading clusters from {clusters_path}...")
    clusters_data = read_json_file(clusters_path)

    print(f"  Loading issue reports from {config['paths']['issue_reports']}...")
    issue_reports = read_json_file(config["paths"]["issue_reports"])

    # Apply confidence filtering if configured
    min_confidence = config.get("filtering", {}).get("min_confidence")
    if min_confidence:
        original_count = len(issue_reports)
        issue_reports = filter_by_confidence(issue_reports, min_confidence)
        print(
            f"  Filtered to {min_confidence}+ confidence: {len(issue_reports)}/{original_count}"
        )

    print(f"  Loading conversations from {config['paths']['clean_conversations']}...")
    conversations = read_json_file(config["paths"]["clean_conversations"])

    # Check which fields are available
    extra_fields = has_extra_agent_fields(issue_reports)

    # Compute stats
    stats: dict = {}

    if extra_fields["resolution_status"]:
        print("  Computing resolution stats...")
        stats["resolution_by_cluster"] = compute_resolution_stats(
            clusters_data, issue_reports
        )
    else:
        print("  Skipping resolution stats (field not available)")
        stats["resolution_by_cluster"] = []

    if extra_fields["user_sentiment"]:
        print("  Computing sentiment stats...")
        stats["sentiment_by_cluster"] = compute_sentiment_stats(
            clusters_data, issue_reports
        )
    else:
        print("  Skipping sentiment stats (field not available)")
        stats["sentiment_by_cluster"] = []

    print("  Computing tool use stats...")
    stats["tool_use_by_cluster"] = compute_tool_use_stats(clusters_data, conversations)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved cluster stats to {output_path}")
    return output_path


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment_from_latest

    # Set up experiment directory for standalone execution
    setup_experiment_from_latest(config)
    run()
