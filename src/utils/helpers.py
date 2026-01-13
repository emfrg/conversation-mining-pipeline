"""Helper Functions.

Common utility functions used across the pipeline.
"""

import json

import numpy as np
from qdrant_client import QdrantClient


def fetch_embeddings_from_qdrant(
    client: QdrantClient,
    collection_name: str,
) -> tuple[list, np.ndarray, list]:
    """Fetch all embeddings, IDs, and payloads from a Qdrant collection.

    Args:
        client: Qdrant client instance.
        collection_name: Name of the collection to fetch from.

    Returns:
        Tuple of (ids, embeddings, payloads) where embeddings is a numpy array.
    """
    points = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        points.extend(result[0])
        offset = result[1]
        if offset is None:
            break

    ids = [p.id for p in points]
    embeddings = np.array([p.vector for p in points])
    payloads = [p.payload for p in points]

    return ids, embeddings, payloads


def strip_json_fence(content: str) -> str:
    """Strip markdown JSON code fence if present.

    Args:
        content: String that may contain ```json fence markers.

    Returns:
        Clean string with fence markers removed.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def read_json_file(file_path: str) -> dict:
    """Read JSON file and return parsed data.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data as dict.
    """
    with open(file_path, encoding="utf-8") as f:
        data: dict = json.load(f)
        return data


def filter_by_confidence(issue_reports: dict, min_confidence: str | None) -> dict:
    """Filter issue reports by minimum confidence level.

    Args:
        issue_reports: Dict of issue reports keyed by conversation_id.
        min_confidence: "high", "medium", or None (no filtering).

    Returns:
        Filtered dict of issue reports.
    """
    if not min_confidence:
        return issue_reports

    confidence_levels = {"high": ["high"], "medium": ["high", "medium"]}
    allowed = confidence_levels.get(min_confidence, [])

    return {k: v for k, v in issue_reports.items() if v.get("confidence") in allowed}


def format_cluster_context(clusters: dict[str, dict]) -> str:
    """Format named clusters as context string for LLM prompt.

    Args:
        clusters: Dict of cluster data with cluster_title and cluster_description.

    Returns:
        Formatted string listing cluster titles and descriptions.
    """
    if not clusters:
        return "No clusters have been named yet."

    lines = []
    for key, data in clusters.items():
        if key == "noise" or "cluster_title" not in data:
            continue
        lines.append(f"- {data['cluster_title']}: {data['cluster_description']}")

    return "\n".join(lines) if lines else "No clusters have been named yet."
