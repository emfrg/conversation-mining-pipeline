"""Data loading functions with Streamlit caching."""

import json
from typing import Any

import streamlit as st

from config import config
from src.utils.helpers import filter_by_confidence
from streamlit_app.config import (
    CLEAN_CONVERSATIONS_PATH,
    CLUSTER_DATA_PATH,
    EXECUTIVE_REPORT_PATH,
    ISSUE_REPORTS_PATH,
    STATS_PATH,
)


@st.cache_data
def load_statistics() -> dict[str, Any] | None:
    """Load statistics JSON file.

    Returns:
        Dictionary containing statistics data, or None if file not found.
    """
    if not STATS_PATH.exists():
        return None

    with open(STATS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_cluster_data() -> dict[str, Any] | None:
    """Load cluster data from clusters_named.json.

    Returns:
        Dictionary containing named cluster data, or None if file not found.
    """
    clusters_path = CLUSTER_DATA_PATH / "clusters_named.json"
    if not clusters_path.exists():
        return None

    with open(clusters_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_cluster_metadata() -> dict[str, Any] | None:
    """Load cluster metadata including k-analysis.

    Returns:
        Dictionary containing clustering metadata, or None if file not found.
    """
    metadata_path = CLUSTER_DATA_PATH / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_executive_report() -> dict[str, Any] | None:
    """Load executive report JSON file.

    Returns:
        Dictionary containing executive report data, or None if file not found.
    """
    if not EXECUTIVE_REPORT_PATH.exists():
        return None

    with open(EXECUTIVE_REPORT_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_issue_reports() -> dict[str, Any] | None:
    """Load issue reports JSON file with optional confidence filtering.

    Applies the same confidence filtering as the pipeline if configured.

    Returns:
        Dictionary containing issue reports data, or None if file not found.
    """
    if not ISSUE_REPORTS_PATH.exists():
        return None

    with open(ISSUE_REPORTS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    # Apply confidence filtering if configured
    min_confidence = config.get("filtering", {}).get("min_confidence")
    if min_confidence:
        data = filter_by_confidence(data, min_confidence)

    return data


@st.cache_data
def load_clean_conversations() -> dict[str, Any] | None:
    """Load clean conversations JSON file.

    Returns:
        Dictionary containing clean conversations data, or None if file not found.
    """
    if not CLEAN_CONVERSATIONS_PATH.exists():
        return None

    with open(CLEAN_CONVERSATIONS_PATH, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_top_issues() -> dict[str, Any] | None:
    """Load precomputed top issues per cluster.

    Returns:
        Dictionary mapping cluster_id (str) -> list of top issues, or None if file not found.
    """
    top_issues_path = CLUSTER_DATA_PATH / "top_issues.json"
    if not top_issues_path.exists():
        return None

    with open(top_issues_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_cluster_stats() -> dict[str, Any] | None:
    """Load pre-computed cluster statistics for visualizations.

    Contains resolution_by_cluster, sentiment_by_cluster, and tool_use_by_cluster.
    Pre-computed by the compute_cluster_stats pipeline step.

    Returns:
        Dictionary containing cluster stats, or None if file not found.
    """
    stats_path = CLUSTER_DATA_PATH / "cluster_stats.json"
    if not stats_path.exists():
        return None

    with open(stats_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


@st.cache_data
def load_final_deduplicated() -> dict[str, Any] | None:
    """Load deduplicated FAQ questions with full traceability.

    Contains deduplicated questions per cluster with original_ids and variants
    for linking back to source conversations.

    Returns:
        Dictionary containing deduplicated FAQ data, or None if file not found.
    """
    dedup_path = CLUSTER_DATA_PATH / "final_deduplicated.json"
    if not dedup_path.exists():
        return None

    with open(dedup_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    """Get a single conversation by ID.

    Args:
        conversation_id: The conversation ID to fetch.

    Returns:
        The conversation dict, or None if not found.
    """
    conversations = load_clean_conversations()
    if conversations:
        result: dict[str, Any] | None = conversations.get(conversation_id)
        return result
    return None
