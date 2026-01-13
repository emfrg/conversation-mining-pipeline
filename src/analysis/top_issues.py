"""Step 7: Top Issues Per Cluster - Rank issues by proximity to cluster centroid.

Computes cluster centroids and identifies the most representative issues
per cluster based on proximity to centroid.

Reads:
    Qdrant collection (config.qdrant.reduced_collection) - PCA-reduced embeddings.
    config.experiment.dir/labels.json - Cluster assignments from Step 5.

Writes:
    config.experiment.dir/centroids.npy - Cluster centroids (n_clusters x n_dims).
    config.experiment.dir/top_issues.json - All issues per cluster ranked by distance.
"""

import json
import os

import numpy as np
from qdrant_client import QdrantClient

from config import config
from src.utils.helpers import fetch_embeddings_from_qdrant, read_json_file


def compute_centroids(
    embeddings: np.ndarray,
    ids: list[str],
    labels: dict[str, int],
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Compute centroids for each cluster.

    Args:
        embeddings: Array of shape (n_samples, n_dims).
        ids: List of IDs corresponding to embeddings.
        labels: Dict mapping ID -> cluster label.

    Returns:
        Tuple of (centroids array, cluster_indices dict).
        centroids: Array of shape (n_clusters, n_dims).
        cluster_indices: Dict mapping cluster_id -> list of indices in embeddings array.
    """
    # Build ID to index mapping
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}

    # Group indices by cluster
    cluster_indices: dict[int, list[int]] = {}
    for id_, cluster_id in labels.items():
        if id_ in id_to_idx:
            idx = id_to_idx[id_]
            if cluster_id not in cluster_indices:
                cluster_indices[cluster_id] = []
            cluster_indices[cluster_id].append(idx)

    # Compute centroid for each cluster
    n_clusters = max(cluster_indices.keys()) + 1
    n_dims = embeddings.shape[1]
    centroids = np.zeros((n_clusters, n_dims))

    for cluster_id, indices in cluster_indices.items():
        cluster_embeddings = embeddings[indices]
        centroids[cluster_id] = np.mean(cluster_embeddings, axis=0)

    return centroids, cluster_indices


def compute_distances_to_centroid(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    cluster_indices: dict[int, list[int]],
) -> dict[int, list[tuple[int, float]]]:
    """Compute distance from each item to its cluster centroid.

    Args:
        embeddings: Array of shape (n_samples, n_dims).
        centroids: Array of shape (n_clusters, n_dims).
        cluster_indices: Dict mapping cluster_id -> list of indices.

    Returns:
        Dict mapping cluster_id -> list of (index, distance) tuples, sorted by distance.
    """
    distances_by_cluster: dict[int, list[tuple[int, float]]] = {}

    for cluster_id, indices in cluster_indices.items():
        centroid = centroids[cluster_id]
        distances = []

        for idx in indices:
            dist = float(np.linalg.norm(embeddings[idx] - centroid))
            distances.append((idx, dist))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[1])
        distances_by_cluster[cluster_id] = distances

    return distances_by_cluster


def get_all_issues_ranked(
    distances_by_cluster: dict[int, list[tuple[int, float]]],
    ids: list[str],
    payloads: list[dict],
) -> dict[str, list[dict]]:
    """Get all issues per cluster ranked by proximity to centroid.

    Args:
        distances_by_cluster: Dict mapping cluster_id -> list of (index, distance).
        ids: List of IDs.
        payloads: List of payload dicts from Qdrant.

    Returns:
        Dict mapping cluster_id (as string) -> list of all issue dicts, sorted by distance.
    """
    all_issues: dict[str, list[dict]] = {}

    for cluster_id, distances in distances_by_cluster.items():
        cluster_issues = []
        for idx, dist in distances:  # Already sorted by distance
            metadata = payloads[idx].get("metadata", {})
            cluster_issues.append(
                {
                    "id": ids[idx],
                    "distance": round(dist, 6),
                    "issue_title": metadata.get("issue_title", ""),
                    "canonical_faq_question": metadata.get(
                        "canonical_faq_question", ""
                    ),
                }
            )
        all_issues[str(cluster_id)] = cluster_issues

    return all_issues


def run() -> str:
    """Run the top_issues pipeline step.

    Computes cluster centroids and ranks all issues by proximity to centroid.

    Returns:
        Path to the saved top_issues.json file.
    """
    experiment_dir = config["experiment"]["dir"]
    labels_path = os.path.join(experiment_dir, "labels.json")
    recreate = config.get("recreate", False)

    # Skip if top_issues.json already exists (unless --recreate)
    top_issues_path = os.path.join(experiment_dir, "top_issues.json")
    if not recreate and os.path.exists(top_issues_path):
        print(f"  Skipping: {top_issues_path} already exists")
        return top_issues_path

    # Connect to Qdrant
    print("  Connecting to Qdrant...")
    client = QdrantClient(
        host=config["qdrant"]["host"],
        port=config["qdrant"]["port"],
    )

    # Fetch embeddings
    collection_name = config["qdrant"]["reduced_collection"]
    print(f"  Fetching embeddings from {collection_name}...")
    ids, embeddings, payloads = fetch_embeddings_from_qdrant(client, collection_name)
    print(f"    Loaded {len(ids)} embeddings ({embeddings.shape[1]} dims)")

    # Load cluster labels
    print(f"  Loading cluster labels from {labels_path}...")
    labels: dict[str, int] = read_json_file(labels_path)
    print(f"    Loaded {len(labels)} labels")

    # Compute centroids
    print("  Computing centroids...")
    centroids, cluster_indices = compute_centroids(embeddings, ids, labels)
    print(f"    Computed {len(centroids)} centroids")

    # Compute distances
    print("  Computing distances to centroids...")
    distances_by_cluster = compute_distances_to_centroid(
        embeddings, centroids, cluster_indices
    )

    # Rank all issues
    print("  Ranking all issues by distance to centroid...")
    all_issues = get_all_issues_ranked(distances_by_cluster, ids, payloads)

    # Save centroids
    centroids_path = os.path.join(experiment_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    print(f"    Saved: {centroids_path}")

    # Save all issues ranked by distance
    top_issues_path = os.path.join(experiment_dir, "top_issues.json")
    with open(top_issues_path, "w") as f:
        json.dump(all_issues, f, indent=2)
    print(f"    Saved: {top_issues_path}")

    return top_issues_path


if __name__ == "__main__":
    from src.utils.experiment_manager import setup_experiment_from_latest

    # Set up experiment directory for standalone execution
    setup_experiment_from_latest(config)
    run()
