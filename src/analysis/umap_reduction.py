"""UMAP Dimensionality Reduction.

Reduces high-dimensional embeddings to 2D for scatter plot visualization.

Input:
    Embedding matrix (n_samples, n_features).

Output:
    2D coordinates array (n_samples, 2).
"""

from typing import cast

import numpy as np
import umap


def reduce_to_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP for visualization.

    Args:
        embeddings: Input embedding matrix (n_samples, n_features).
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        random_state: Random seed for reproducibility.

    Returns:
        2D coordinates array (n_samples, 2).
    """
    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_jobs=1,  # Required when using random_state for reproducibility
    )
    return cast(np.ndarray, reducer.fit_transform(embeddings))
