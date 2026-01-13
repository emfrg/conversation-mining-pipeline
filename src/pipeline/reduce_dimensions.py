"""Step 4: Reduce Dimensions - Apply PCA dimensionality reduction.

Applies PCA (Principal Component Analysis) to reduce embedding dimensionality.
Automatically determines optimal components to keep specified variance threshold.

Reads:
    Qdrant collection (config.qdrant.embeddings_collection) - High-dim embeddings.

Writes:
    Qdrant collection (config.qdrant.reduced_collection) - Reduced embeddings.
    config.paths.models_dir/pca_model.joblib - Fitted PCA model.
    config.paths.models_dir/pca_metadata.json - PCA model metadata.
"""

import json
import os

import joblib
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sklearn.decomposition import PCA

from config import config
from src.utils.helpers import fetch_embeddings_from_qdrant

load_dotenv()


def determine_n_components(
    embeddings: np.ndarray,
    variance_threshold: float | None = None,
) -> tuple[int, np.ndarray]:
    """Determine optimal number of PCA components based on explained variance.

    Args:
        embeddings: Input embedding matrix (n_samples, n_features).
        variance_threshold: Target cumulative variance to keep (default from config).

    Returns:
        Tuple of (n_components, cumulative_variance_array).
    """
    if variance_threshold is None:
        variance_threshold = config["pca"]["variance_threshold"]
    max_components = config["pca"]["max_components"]

    n_samples, n_features = embeddings.shape
    max_possible = min(n_samples - 1, n_features, max_components)

    # Fit PCA to analyze variance
    pca = PCA(n_components=max_possible)
    pca.fit(embeddings)

    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Find first component that exceeds threshold
    n_components = int(np.argmax(cumulative >= variance_threshold) + 1)

    # If threshold not achievable, use all available
    if cumulative[-1] < variance_threshold:
        n_components = max_possible

    return n_components, cumulative


def print_variance_report(cumulative: np.ndarray, n_components: int) -> None:
    """Print variance analysis for selected components.

    Args:
        cumulative: Cumulative explained variance ratio array.
        n_components: Number of components selected.
    """
    print("\nVariance Analysis:")
    print(f"  Components selected: {n_components}")
    print(f"  Variance captured: {cumulative[n_components - 1] * 100:.1f}%")

    # Show first 5 components
    print("\n  First 5 components:")
    for i in range(min(5, len(cumulative))):
        print(f"    PC{i + 1}: {cumulative[i] * 100:.1f}% cumulative")


def save_pca_model(pca: PCA, n_samples: int) -> None:
    """Save fitted PCA model and metadata to disk.

    Args:
        pca: Fitted PCA model instance.
        n_samples: Number of samples used for fitting.
    """
    models_dir = config["paths"]["models_dir"]
    source_collection = config["qdrant"]["embeddings_collection"]

    os.makedirs(models_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(models_dir, "pca_model.joblib")
    joblib.dump(pca, model_path)

    # Save metadata
    metadata = {
        "n_components": pca.n_components_,
        "explained_variance_total": float(sum(pca.explained_variance_ratio_)),
        "training_samples": n_samples,
        "source_collection": source_collection,
    }
    metadata_path = os.path.join(models_dir, "pca_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def ensure_reduced_collection(client: QdrantClient, n_components: int) -> None:
    """Ensure reduced embeddings collection exists. Recreates if exists.

    Args:
        client: Qdrant client instance.
        n_components: Vector dimension size for the collection.
    """
    reduced_collection = config["qdrant"]["reduced_collection"]

    if client.collection_exists(reduced_collection):
        client.delete_collection(reduced_collection)

    client.create_collection(
        collection_name=reduced_collection,
        vectors_config=VectorParams(size=n_components, distance=Distance.COSINE),
    )


def store_reduced_embeddings(
    client: QdrantClient,
    ids: list,
    reduced_embeddings: np.ndarray,
    payloads: list,
) -> None:
    """Store reduced embeddings in Qdrant collection.

    Args:
        client: Qdrant client instance.
        ids: List of point IDs.
        reduced_embeddings: Reduced embedding matrix.
        payloads: List of payload dicts for each point.
    """
    reduced_collection = config["qdrant"]["reduced_collection"]

    points = [
        PointStruct(id=id_, vector=emb.tolist(), payload=payload)
        for id_, emb, payload in zip(ids, reduced_embeddings, payloads, strict=False)
    ]
    client.upsert(collection_name=reduced_collection, points=points)


def run() -> int:
    """Run the reduce_dimensions pipeline step.

    Applies PCA to reduce embedding dimensionality and stores results.

    Returns:
        Number of reduced embeddings stored.
    """
    qdrant_host = config["qdrant"]["host"]
    qdrant_port = config["qdrant"]["port"]
    source_collection = config["qdrant"]["embeddings_collection"]
    reduced_collection = config["qdrant"]["reduced_collection"]
    variance_threshold = config["pca"]["variance_threshold"]
    models_dir = config["paths"]["models_dir"]
    recreate = config.get("recreate", False)

    # Initialize Qdrant client
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Skip if reduced collection already exists (unless --recreate)
    if not recreate and client.collection_exists(reduced_collection):
        info = client.get_collection(reduced_collection)
        print(
            f"  Skipping: Collection '{reduced_collection}' exists ({info.points_count} points)"
        )
        return int(info.points_count)

    # Fetch all embeddings from source collection
    print(f"  Fetching embeddings from {source_collection}...")
    ids, embeddings, payloads = fetch_embeddings_from_qdrant(client, source_collection)
    print(f"  Loaded {len(ids)} embeddings ({embeddings.shape[1]} dims)")

    # Determine optimal number of components
    n_components, cumulative = determine_n_components(embeddings, variance_threshold)
    print(
        f"  PCA components: {n_components} (keeping {cumulative[n_components - 1] * 100:.1f}% variance)"
    )

    # Fit PCA and transform
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    print(f"  Reduced: {embeddings.shape} -> {reduced.shape}")

    # Save PCA model
    save_pca_model(pca, len(ids))
    print(f"  Saved PCA model to {models_dir}/")

    # Store reduced embeddings in new collection
    ensure_reduced_collection(client, n_components)
    store_reduced_embeddings(client, ids, reduced, payloads)
    print(f"  Stored {len(ids)} reduced embeddings in {reduced_collection}")

    return len(ids)


if __name__ == "__main__":
    run()
