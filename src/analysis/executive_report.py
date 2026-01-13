"""Executive Report Generator.

Prepares usage statistics and cluster analysis data for LLM-based executive
report generation. Uses the executive_report_agent to generate insights.

Input:
    chat_history_stats.json - Usage statistics.
    clusters_named.json - Named cluster data.

Output:
    executive_report.json - Generated executive report with insights.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.executive_report_agent import executive_report_generator
from src.schemas import ExecutiveReport

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_STATS_PATH = PROJECT_ROOT / "data" / "output" / "chat_history_stats.json"
DEFAULT_CLUSTERS_PATH = (
    PROJECT_ROOT / "data" / "models" / "kmeans_latest" / "clusters_named.json"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "output" / "executive_report.json"


def load_usage_stats(stats_path: Path | None = None) -> dict[str, Any]:
    """Load usage statistics from JSON file.

    Args:
        stats_path: Path to chat_history_stats.json. Uses default if None.

    Returns:
        Dictionary containing usage statistics.

    Raises:
        FileNotFoundError: If stats file doesn't exist.
    """
    path = stats_path or DEFAULT_STATS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Statistics file not found: {path}")

    with open(path, encoding="utf-8") as f:
        stats: dict[str, Any] = json.load(f)
        return stats


def load_cluster_data(clusters_path: Path | None = None) -> dict[str, Any]:
    """Load cluster data from JSON file.

    Args:
        clusters_path: Path to clusters_named.json. Uses default if None.

    Returns:
        Dictionary containing named cluster data.

    Raises:
        FileNotFoundError: If clusters file doesn't exist.
    """
    path = clusters_path or DEFAULT_CLUSTERS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Clusters file not found: {path}")

    with open(path, encoding="utf-8") as f:
        cluster_data: dict[str, Any] = json.load(f)
        return cluster_data


def prepare_usage_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """Extract usage statistics matching what's shown in Usage Statistics page.

    Args:
        stats: Raw statistics dictionary from chat_history_stats.json.

    Returns:
        Structured usage summary for LLM input.
    """
    overview = stats.get("overview", {})
    conversation_length = stats.get("conversation_length", {})
    message_lengths = stats.get("message_lengths", {})
    features = stats.get("features", {})
    temporal = stats.get("temporal_distribution", {})

    # Get top 10 busiest dates
    busiest_dates = temporal.get("busiest_dates", {})
    top_busiest_dates = dict(
        sorted(busiest_dates.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "total_conversations": overview.get("total_conversations", 0),
        "total_messages": overview.get("total_messages", 0),
        "date_range_days": overview.get("date_range_days", 0),
        "avg_conversations_per_day": overview.get("avg_conversations_per_day", 0),
        "median_messages_per_conversation": conversation_length.get(
            "median_messages_per_conversation", 0
        ),
        "first_conversation": overview.get("first_conversation", "")[:10],
        "last_conversation": overview.get("last_conversation", "")[:10],
        "tools": {
            "top_tools_used": features.get("top_tools_used", {}),
            "conversations_with_tool_use": features.get(
                "conversations_with_tool_use", 0
            ),
            "conversations_with_citations": features.get(
                "conversations_with_citations", 0
            ),
        },
        "message_lengths": {
            "median_user_chars": message_lengths.get("median_user_message_chars", 0),
            "median_assistant_chars": message_lengths.get(
                "median_assistant_message_chars", 0
            ),
        },
        "temporal": {
            "by_hour": temporal.get("by_hour", {}),
            "by_weekday": temporal.get("by_weekday", {}),
            "busiest_dates": top_busiest_dates,
        },
    }


def prepare_cluster_summary(cluster_data: dict[str, Any]) -> dict[str, Any]:
    """Extract cluster information matching what's shown in Cluster Analysis page.

    Args:
        cluster_data: Raw cluster data from clusters_named.json.

    Returns:
        Structured cluster summary for LLM input.
    """
    method = cluster_data.get("method", "kmeans")
    total_items = cluster_data.get("total_items", 0)
    n_clusters = cluster_data.get("n_clusters", 0)
    clusters = cluster_data.get("clusters", {})

    # Build cluster items list
    cluster_items = []
    for cluster_key, cluster_info in clusters.items():
        count = cluster_info.get("count", 0)
        percentage = round(count / total_items * 100, 1) if total_items > 0 else 0

        cluster_items.append(
            {
                "title": cluster_info.get("cluster_title", cluster_key),
                "count": count,
                "percentage": percentage,
                "description": cluster_info.get("cluster_description", ""),
                "detailed_description": cluster_info.get(
                    "cluster_detailed_description", ""
                ),
            }
        )

    # Sort by count descending
    cluster_items.sort(key=lambda x: x["count"], reverse=True)

    return {
        "method": method,
        "total_items": total_items,
        "n_clusters": n_clusters,
        "items": cluster_items,
    }


def prepare_llm_input(
    stats_path: Path | None = None,
    clusters_path: Path | None = None,
) -> dict[str, Any]:
    """Prepare combined data structure for LLM input.

    Loads and processes both usage statistics and cluster data,
    returning a single dictionary ready for JSON serialization.

    Args:
        stats_path: Path to chat_history_stats.json. Uses default if None.
        clusters_path: Path to clusters_named.json. Uses default if None.

    Returns:
        Combined dictionary with 'usage' and 'clusters' keys.
    """
    # Load raw data
    stats = load_usage_stats(stats_path)
    cluster_data = load_cluster_data(clusters_path)

    # Prepare summaries
    usage_summary = prepare_usage_summary(stats)
    cluster_summary = prepare_cluster_summary(cluster_data)

    return {
        "usage": usage_summary,
        "clusters": cluster_summary,
    }


def get_llm_input_json(
    stats_path: Path | None = None,
    clusters_path: Path | None = None,
    indent: int = 2,
) -> str:
    """Get LLM input as formatted JSON string.

    Convenience function that calls prepare_llm_input() and returns JSON.

    Args:
        stats_path: Path to chat_history_stats.json. Uses default if None.
        clusters_path: Path to clusters_named.json. Uses default if None.
        indent: JSON indentation level.

    Returns:
        JSON string ready to be inserted into prompt.
    """
    data = prepare_llm_input(stats_path, clusters_path)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def generate_report(
    stats_path: Path | None = None,
    clusters_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate executive report using LLM agent.

    Args:
        stats_path: Path to chat_history_stats.json. Uses default if None.
        clusters_path: Path to clusters_named.json. Uses default if None.
        output_path: Path to save report JSON. Uses default if None.

    Returns:
        Dictionary containing the generated report.
    """
    # Prepare data for LLM
    data_json = get_llm_input_json(stats_path, clusters_path)

    print("Calling LLM agent to generate executive report...")

    # Invoke the agent with structured output
    response: ExecutiveReport = executive_report_generator.invoke(
        {"data_json": data_json}
    )

    # Convert to dict
    report: dict[str, Any] = response.model_dump()

    # Add metadata
    report["_metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "stats_path": str(stats_path or DEFAULT_STATS_PATH),
        "clusters_path": str(clusters_path or DEFAULT_CLUSTERS_PATH),
    }

    # Save to file
    output = output_path or DEFAULT_OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Executive report saved to: {output}")

    return report


if __name__ == "__main__":
    try:
        report = generate_report()
        print("\n" + "=" * 60)
        print("EXECUTIVE SUMMARY")
        print("=" * 60)
        print(report.get("executive_summary", "No summary generated"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
