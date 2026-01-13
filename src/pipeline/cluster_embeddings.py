"""Step 5: Cluster Embeddings - Group similar issues using clustering.

Applies clustering algorithm (K-Means or HDBSCAN) to reduced embeddings.
Supports parameter evaluation and auto-tuning for optimal cluster selection.

Reads:
    Qdrant collection (config.qdrant.reduced_collection) - PCA-reduced embeddings.

Writes:
    config.experiment.dir/model.joblib - Fitted clustering model.
    config.experiment.dir/labels.json - Cluster label assignments.
    config.experiment.dir/metadata.json - Clustering metadata and analysis.
    config.experiment.dir/clusters_readable.json - Human-readable cluster contents.
    config.experiment.dir/kmeans_k_analysis.png - K selection plot (if enabled).
    config.experiment.dir/hdbscan_analysis.png - HDBSCAN analysis plot (if enabled).
"""

import json
import os
import warnings
from typing import Any

# Silence SyntaxWarning from hdbscan's docstrings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")

import hdbscan  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from config import config  # noqa: E402
from src.analysis.hdbscan_plots import create_hdbscan_analysis_chart  # noqa: E402
from src.analysis.hdbscan_selection import (  # noqa: E402
    compute_hdbscan_analysis,
    print_hdbscan_analysis_report,
    select_optimal_min_cluster_size,
)
from src.analysis.kmeans_plots import create_k_analysis_chart  # noqa: E402
from src.analysis.kmeans_selection import (  # noqa: E402
    compute_k_analysis,
    print_k_analysis_report,
    select_optimal_k,
)
from src.utils.experiment_manager import save_config_snapshot  # noqa: E402
from src.utils.helpers import fetch_embeddings_from_qdrant  # noqa: E402

load_dotenv()

warnings.filterwarnings(
    "ignore", category=SyntaxWarning, module="hdbscan"
)  # prevent known issue with hdbscan


def cluster_kmeans(embeddings: np.ndarray) -> tuple[KMeans, np.ndarray, dict | None]:
    """Cluster embeddings using K-Means.

    If evaluation is enabled, computes elbow/silhouette for k_range.
    If auto_tune is enabled, uses optimal k; otherwise uses fixed n_clusters.

    Args:
        embeddings: Input embedding matrix (n_samples, n_features).

    Returns:
        Tuple of (fitted KMeans model, cluster labels array, k_analysis dict or None).
    """
    kmeans_config = config["clustering"]["kmeans"]
    random_state = kmeans_config["random_state"]
    eval_config = kmeans_config.get("evaluation", {})
    auto_tune_config = kmeans_config.get("auto_tune", {})

    k_analysis = None

    # Step 1: Run evaluation if enabled
    if eval_config.get("enabled", False):
        k_range = tuple(eval_config.get("k_range", [3, 30]))
        k_step = eval_config.get("k_step", 1)

        k_analysis = compute_k_analysis(
            embeddings=embeddings,
            k_range=k_range,
            k_step=k_step,
            random_state=random_state,
        )
        print_k_analysis_report(k_analysis)

    # Step 2: Determine n_clusters to use
    if auto_tune_config.get("enabled", False) and k_analysis is not None:
        selection_method = auto_tune_config.get("selection_method", "combined")
        silhouette_weight = auto_tune_config.get("silhouette_weight", 0.5)

        n_clusters = select_optimal_k(
            analysis=k_analysis,
            selection_method=selection_method,
            silhouette_weight=silhouette_weight,
        )
        print(f"  Auto-selected n_clusters={n_clusters} (method: {selection_method})")
        k_analysis["selected_k"] = n_clusters
        k_analysis["selection_method"] = selection_method
    else:
        n_clusters = kmeans_config["n_clusters"]
        print(f"  Using fixed n_clusters={n_clusters}")
        if k_analysis is not None:
            k_analysis["selected_k"] = n_clusters
            k_analysis["selection_method"] = "fixed"

    # Step 3: Run final clustering
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = model.fit_predict(embeddings)

    # Step 4: Save visualization if enabled (to experiment directory)
    if eval_config.get("save_plot", False) and k_analysis is not None:
        experiment_dir = config["experiment"]["dir"]
        plot_output = os.path.join(experiment_dir, "kmeans_k_analysis.png")
        create_k_analysis_chart(k_analysis, n_clusters, plot_output)

    return model, labels, k_analysis


def cluster_hdbscan(
    embeddings: np.ndarray,
) -> tuple[hdbscan.HDBSCAN, np.ndarray, dict | None]:
    """Cluster embeddings using HDBSCAN.

    If evaluation is enabled, computes DBCV/silhouette for min_cluster_size_range.
    If auto_tune is enabled, uses optimal min_cluster_size; otherwise uses fixed config.

    Args:
        embeddings: Input embedding matrix (n_samples, n_features).

    Returns:
        Tuple of (fitted HDBSCAN model, cluster labels array, analysis dict or None).
    """
    hdbscan_config = config["clustering"]["hdbscan"]
    min_samples = hdbscan_config["min_samples"]
    eval_config = hdbscan_config.get("evaluation", {})
    auto_tune_config = hdbscan_config.get("auto_tune", {})

    analysis = None

    # Step 1: Run evaluation if enabled
    if eval_config.get("enabled", False):
        size_range = tuple(eval_config.get("min_cluster_size_range", [5, 100]))
        step = eval_config.get("step", 5)

        analysis = compute_hdbscan_analysis(
            embeddings=embeddings,
            min_cluster_size_range=size_range,
            step=step,
            min_samples=min_samples,
        )
        print_hdbscan_analysis_report(analysis)

    # Step 2: Determine min_cluster_size to use
    if auto_tune_config.get("enabled", False) and analysis is not None:
        selection_method = auto_tune_config.get("selection_method", "combined")
        silhouette_weight = auto_tune_config.get("silhouette_weight", 0.5)

        min_cluster_size = select_optimal_min_cluster_size(
            analysis=analysis,
            selection_method=selection_method,
            silhouette_weight=silhouette_weight,
        )
        print(
            f"  Auto-selected min_cluster_size={min_cluster_size} (method: {selection_method})"
        )
        analysis["selected_size"] = min_cluster_size
        analysis["selection_method"] = selection_method
    else:
        min_cluster_size = hdbscan_config["min_cluster_size"]
        print(f"  Using fixed min_cluster_size={min_cluster_size}")
        if analysis is not None:
            analysis["selected_size"] = min_cluster_size
            analysis["selection_method"] = "fixed"

    # Step 3: Run final clustering
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = model.fit_predict(embeddings)

    # Step 4: Save visualization if enabled (to experiment directory)
    if eval_config.get("save_plot", False) and analysis is not None:
        experiment_dir = config["experiment"]["dir"]
        plot_output = os.path.join(experiment_dir, "hdbscan_analysis.png")
        create_hdbscan_analysis_chart(analysis, min_cluster_size, plot_output)

    return model, labels, analysis


def save_clustering_model(
    model: KMeans | hdbscan.HDBSCAN,
    labels: np.ndarray,
    ids: list,
    method: str,
    n_samples: int,
    analysis: dict | None = None,
) -> None:
    """Save clustering model and metadata to experiment directory.

    Args:
        model: Fitted clustering model (KMeans or HDBSCAN).
        labels: Cluster label array.
        ids: List of point IDs.
        method: Clustering method name ("kmeans" or "hdbscan").
        n_samples: Number of samples clustered.
        analysis: Optional analysis dict from parameter evaluation.
    """
    experiment_dir = config["experiment"]["dir"]
    source_collection = config["qdrant"]["reduced_collection"]

    # Save model
    model_path = os.path.join(experiment_dir, "model.joblib")
    joblib.dump(model, model_path)

    # Save labels mapping (id -> cluster)
    labels_map = {str(id_): int(label) for id_, label in zip(ids, labels, strict=False)}
    labels_path = os.path.join(experiment_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels_map, f, indent=2)

    # Save metadata
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
    n_noise = int(np.sum(labels == -1))

    metadata = {
        "method": method,
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "source_collection": source_collection,
    }

    # Add method-specific metadata
    if method == "kmeans":
        metadata["settings"] = {
            "n_clusters_used": n_clusters,
            "n_clusters_config": config["clustering"]["kmeans"]["n_clusters"],
            "random_state": config["clustering"]["kmeans"]["random_state"],
            "auto_selected": config["clustering"]["kmeans"]
            .get("auto_tune", {})
            .get("enabled", False),
        }
        metadata["inertia"] = float(model.inertia_)

        # Add k-selection analysis if available
        if analysis is not None:
            metadata["k_analysis"] = analysis

    elif method == "hdbscan":
        metadata["settings"] = {
            "min_cluster_size": config["clustering"]["hdbscan"]["min_cluster_size"],
            "min_samples": config["clustering"]["hdbscan"]["min_samples"],
            "auto_selected": config["clustering"]["hdbscan"]
            .get("auto_tune", {})
            .get("enabled", False),
        }

        # Add HDBSCAN analysis if available
        if analysis is not None:
            metadata["hdbscan_analysis"] = analysis

    metadata_path = os.path.join(experiment_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_clusters_for_inspection(
    labels: np.ndarray,
    ids: list,
    payloads: list,
    method: str,
) -> None:
    """Save clusters to JSON for human inspection.

    Args:
        labels: Cluster label array.
        ids: List of point IDs.
        payloads: List of payload dicts with issue metadata.
        method: Clustering method name.
    """
    experiment_dir = config["experiment"]["dir"]

    clusters: dict[str, dict[str, Any]] = {}
    for idx, label in enumerate(labels):
        label_key = "noise" if label == -1 else f"cluster_{label}"
        if label_key not in clusters:
            clusters[label_key] = {"count": 0, "items": []}

        payload = payloads[idx]
        metadata = payload.get("metadata", {})

        clusters[label_key]["items"].append(
            {
                "id": str(ids[idx]),
                "issue_title": metadata.get("issue_title", ""),
                "canonical_faq_question": metadata.get("canonical_faq_question", ""),
            }
        )
        clusters[label_key]["count"] += 1

    output = {
        "method": method,
        "total_items": len(ids),
        "n_clusters": len([k for k in clusters if k != "noise"]),
        "clusters": clusters,
    }

    output_path = os.path.join(experiment_dir, "clusters_readable.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def print_cluster_report(labels: np.ndarray, method: str) -> None:
    """Print clustering results summary.

    Args:
        labels: Cluster label array.
        method: Clustering method name.
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"\nClustering Results ({method}):")
    print(f"  Clusters found: {n_clusters}")
    if n_noise > 0:
        print(f"  Noise points: {n_noise}")

    print("\n  Cluster sizes:")
    for label in sorted(unique_labels):
        count = int(np.sum(labels == label))
        if label == -1:
            print(f"    Noise: {count} points")
        else:
            print(f"    Cluster {label}: {count} points")


def print_cluster_samples(
    labels: np.ndarray,
    ids: list,
    payloads: list,
    max_per_cluster: int = 2,
) -> None:
    """Print sample issues from each cluster.

    Args:
        labels: Cluster label array.
        ids: List of point IDs.
        payloads: List of payload dicts with issue metadata.
        max_per_cluster: Maximum samples to show per cluster.
    """
    print("\n### Sample issues per cluster ###")

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        label_name = "Noise" if label == -1 else f"Cluster {label}"

        print(f"\n{label_name}:")
        indices = np.where(labels == label)[0]

        for idx in indices[:max_per_cluster]:
            payload = payloads[idx]
            metadata = payload.get("metadata", {})
            title = metadata.get("issue_title", "No title")
            question = metadata.get("canonical_faq_question", "No question")
            print(f"  - {title}")
            print(f"    Q: {question[:80]}...")


def run() -> int:
    """Run the cluster_embeddings pipeline step.

    Applies clustering to reduced embeddings and saves results.

    Returns:
        Number of clusters found.
    """
    qdrant_host = config["qdrant"]["host"]
    qdrant_port = config["qdrant"]["port"]
    source_collection = config["qdrant"]["reduced_collection"]
    method = config["clustering"]["method"]
    experiment_dir = config["experiment"]["dir"]
    recreate = config.get("recreate", False)

    # Skip if labels already exist in experiment (unless --recreate)
    labels_path = os.path.join(experiment_dir, "labels.json")
    if not recreate and os.path.exists(labels_path):
        print(f"  Skipping: {labels_path} already exists")
        # Return cluster count from existing labels
        with open(labels_path) as f:
            labels = json.load(f)
        unique_labels = set(labels.values())
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        return n_clusters

    # Save config snapshot at start of clustering
    save_config_snapshot(config)

    # Initialize Qdrant client
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Fetch reduced embeddings
    print(f"  Fetching embeddings from {source_collection}...")
    ids, embeddings, payloads = fetch_embeddings_from_qdrant(client, source_collection)
    print(f"  Loaded {len(ids)} embeddings ({embeddings.shape[1]} dims)")

    # Run clustering based on selected method
    print(f"  Running {method} clustering...")
    analysis = None
    if method == "kmeans":
        model, labels, analysis = cluster_kmeans(embeddings)
    elif method == "hdbscan":
        model, labels, analysis = cluster_hdbscan(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Get cluster count
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"  Found {n_clusters} clusters")

    # Save model and labels
    save_clustering_model(model, labels, ids, method, len(ids), analysis)
    save_clusters_for_inspection(labels, ids, payloads, method)
    print(f"  Saved to {experiment_dir}/")

    return n_clusters


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment

    # Set up experiment directory for standalone execution
    setup_experiment(config)
    run()
