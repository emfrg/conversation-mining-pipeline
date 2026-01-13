"""HDBSCAN Parameter Optimization.

Determines the optimal min_cluster_size for HDBSCAN using DBCV and
silhouette analysis. Supports three selection methods:
- DBCV: Density-Based Clustering Validation via relative_validity_
- Silhouette: Measures cluster cohesion and separation (excluding noise)
- Combined: Weighted combination of both methods

Input:
    Embedding matrix (n_samples, n_features).

Output:
    Analysis dict with sizes, DBCV scores, silhouette scores, and selected size.
"""

import warnings

# Silence SyntaxWarning from hdbscan's docstrings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="hdbscan")

import hdbscan  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

# Suppress sklearn deprecation warning from hdbscan library
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
)


def compute_hdbscan_analysis(
    embeddings: np.ndarray,
    min_cluster_size_range: tuple[int, int],
    step: int,
    min_samples: int,
) -> dict:
    """
    Compute DBCV and silhouette scores for a range of min_cluster_size values.

    Args:
        embeddings: Input data matrix (n_samples, n_features).
        min_cluster_size_range: (min, max) range to evaluate.
        step: Step size for min_cluster_size values.
        min_samples: Fixed min_samples value to use.

    Returns:
        Dictionary containing:
        - min_cluster_size_range: The range evaluated
        - step: Step size used
        - all_results: Dict mapping min_cluster_size -> {dbcv, silhouette, n_clusters, n_noise}
        - best_dbcv_size: min_cluster_size with highest DBCV
        - best_dbcv_score: The highest DBCV score
        - best_silhouette_size: min_cluster_size with highest silhouette
        - best_silhouette_score: The highest silhouette score
    """
    min_size, max_size = min_cluster_size_range
    # Ensure max_size doesn't exceed n_samples
    max_size = min(max_size, len(embeddings) - 1)

    size_values = list(range(min_size, max_size + 1, step))

    all_results = {}
    dbcv_scores = []
    silhouette_scores = []

    print(
        f"  Evaluating min_cluster_size from {min_size} to {max_size} (step={step})..."
    )

    for size in size_values:
        model = hdbscan.HDBSCAN(
            min_cluster_size=size,
            min_samples=min_samples,
            gen_min_span_tree=True,  # Required for relative_validity_ (DBCV)
        )
        labels = model.fit_predict(embeddings)

        # Get DBCV score (relative_validity_)
        dbcv = float(model.relative_validity_)

        # Get silhouette score (excluding noise)
        sil_score = compute_silhouette_excluding_noise(embeddings, labels)

        # Count clusters and noise
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(labels == -1))

        all_results[size] = {
            "dbcv": dbcv,
            "silhouette": sil_score,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
        dbcv_scores.append(dbcv)
        silhouette_scores.append(sil_score)

    # Find best DBCV
    best_dbcv_idx = np.argmax(dbcv_scores)
    best_dbcv_size = size_values[best_dbcv_idx]
    best_dbcv_score = dbcv_scores[best_dbcv_idx]

    # Find best silhouette (filter out -1 values)
    valid_sil_scores = [s for s in silhouette_scores if s != -1]
    if valid_sil_scores:
        best_sil_idx = np.argmax(silhouette_scores)
        best_silhouette_size = size_values[best_sil_idx]
        best_silhouette_score = silhouette_scores[best_sil_idx]
    else:
        best_silhouette_size = size_values[0]
        best_silhouette_score = -1.0

    return {
        "min_cluster_size_range": list(min_cluster_size_range),
        "step": step,
        "all_results": all_results,
        "best_dbcv_size": best_dbcv_size,
        "best_dbcv_score": best_dbcv_score,
        "best_silhouette_size": best_silhouette_size,
        "best_silhouette_score": best_silhouette_score,
    }


def compute_silhouette_excluding_noise(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score excluding noise points (label=-1).

    Args:
        embeddings: Input data matrix.
        labels: Cluster labels from HDBSCAN.

    Returns:
        Silhouette score, or -1 if not enough valid clusters.
    """
    mask = labels != -1
    if mask.sum() < 2:
        return -1.0

    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return -1.0

    return float(silhouette_score(embeddings[mask], labels[mask]))


def select_optimal_min_cluster_size(
    analysis: dict,
    selection_method: str,
    silhouette_weight: float = 0.5,
) -> int:
    """
    Select optimal min_cluster_size based on the specified method.

    Args:
        analysis: Analysis results from compute_hdbscan_analysis().
        selection_method: "dbcv", "silhouette", or "combined".
        silhouette_weight: Weight for silhouette in combined method (0-1).
            0.0 = pure DBCV, 1.0 = pure silhouette.

    Returns:
        The selected optimal min_cluster_size value.

    Raises:
        ValueError: If selection_method is not "dbcv", "silhouette", or "combined".
    """
    if selection_method == "dbcv":
        return int(analysis["best_dbcv_size"])

    elif selection_method == "silhouette":
        return int(analysis["best_silhouette_size"])

    elif selection_method == "combined":
        return _select_combined(analysis, silhouette_weight)

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def _select_combined(analysis: dict, silhouette_weight: float) -> int:
    """
    Select optimal min_cluster_size using weighted combination of DBCV and silhouette.

    Scores each size based on:
    - Normalized DBCV score (higher = better)
    - Normalized silhouette score (higher = better)

    Args:
        analysis: Analysis results from compute_hdbscan_analysis().
        silhouette_weight: Weight for silhouette (0-1).

    Returns:
        The size with highest combined score.
    """
    all_results = analysis["all_results"]

    size_values = list(all_results.keys())
    dbcv_scores = np.array([all_results[s]["dbcv"] for s in size_values])
    silhouette_scores = np.array([all_results[s]["silhouette"] for s in size_values])

    # Normalize DBCV scores to [0, 1]
    dbcv_range = dbcv_scores.max() - dbcv_scores.min()
    if dbcv_range > 0:
        dbcv_norm = (dbcv_scores - dbcv_scores.min()) / dbcv_range
    else:
        dbcv_norm = np.ones_like(dbcv_scores)

    # Normalize silhouette scores to [0, 1], handling -1 values
    valid_mask = silhouette_scores != -1
    if valid_mask.any():
        valid_scores = silhouette_scores[valid_mask]
        sil_min, sil_max = valid_scores.min(), valid_scores.max()
        sil_range = sil_max - sil_min
        if sil_range > 0:
            sil_norm = np.where(
                valid_mask,
                (silhouette_scores - sil_min) / sil_range,
                0.0,  # Invalid scores get 0
            )
        else:
            sil_norm = np.where(valid_mask, 1.0, 0.0)
    else:
        sil_norm = np.zeros_like(silhouette_scores)

    # Combined score
    dbcv_weight = 1 - silhouette_weight
    combined_scores = dbcv_weight * dbcv_norm + silhouette_weight * sil_norm

    best_idx = np.argmax(combined_scores)
    return int(size_values[best_idx])


def print_hdbscan_analysis_report(analysis: dict) -> None:
    """
    Print HDBSCAN analysis results to console.

    Args:
        analysis: Analysis results from compute_hdbscan_analysis().
    """
    all_results = analysis["all_results"]
    best_dbcv_size = analysis["best_dbcv_size"]
    best_dbcv_score = analysis["best_dbcv_score"]
    best_sil_size = analysis["best_silhouette_size"]
    best_sil_score = analysis["best_silhouette_score"]

    print("\n  HDBSCAN Parameter Analysis:")
    print("  " + "-" * 66)
    print(
        f"  {'min_cluster_size':>16}  {'DBCV':>8}  {'Silhouette':>10}  {'Clusters':>8}  {'Noise':>6}  Notes"
    )
    print("  " + "-" * 66)

    for size, metrics in all_results.items():
        dbcv = metrics["dbcv"]
        sil = metrics["silhouette"]
        n_clusters = metrics["n_clusters"]
        n_noise = metrics["n_noise"]

        notes = []
        if size == best_dbcv_size:
            notes.append("best DBCV")
        if size == best_sil_size:
            notes.append("best silhouette")

        notes_str = f"  <-- {', '.join(notes)}" if notes else ""
        sil_str = f"{sil:>10.4f}" if sil != -1 else f"{'N/A':>10}"
        print(
            f"  {size:>16}  {dbcv:>8.4f}  {sil_str}  {n_clusters:>8}  {n_noise:>6}{notes_str}"
        )

    print("  " + "-" * 66)
    print("\n  Summary:")
    print(
        f"    Best DBCV at: min_cluster_size={best_dbcv_size} ({best_dbcv_score:.4f})"
    )
    sil_display = f"{best_sil_score:.4f}" if best_sil_score != -1 else "N/A"
    print(f"    Best silhouette at: min_cluster_size={best_sil_size} ({sil_display})")
