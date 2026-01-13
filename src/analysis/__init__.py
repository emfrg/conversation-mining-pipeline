"""
Analysis module for clustering evaluation and optimization.

Contains tools for determining optimal cluster parameters using
statistical methods like the elbow method, silhouette analysis, and DBCV.
"""

from src.analysis.hdbscan_plots import create_hdbscan_analysis_chart
from src.analysis.hdbscan_selection import (
    compute_hdbscan_analysis,
    compute_silhouette_excluding_noise,
    print_hdbscan_analysis_report,
    select_optimal_min_cluster_size,
)
from src.analysis.kmeans_plots import create_k_analysis_chart
from src.analysis.kmeans_selection import (
    compute_k_analysis,
    detect_elbow_point,
    print_k_analysis_report,
    select_optimal_k,
)

__all__ = [
    # KMeans
    "compute_k_analysis",
    "detect_elbow_point",
    "select_optimal_k",
    "print_k_analysis_report",
    "create_k_analysis_chart",
    # HDBSCAN
    "compute_hdbscan_analysis",
    "compute_silhouette_excluding_noise",
    "select_optimal_min_cluster_size",
    "print_hdbscan_analysis_report",
    "create_hdbscan_analysis_chart",
]
