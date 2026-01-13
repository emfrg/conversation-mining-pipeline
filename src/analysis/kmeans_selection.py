"""K-Means Optimal K Selection.

Determines the optimal number of clusters for K-Means using elbow method
and silhouette analysis. Supports three selection methods:
- Elbow: Detects the "elbow" point in the inertia curve
- Silhouette: Maximizes cluster cohesion and separation
- Combined: Weighted combination of both methods

Input:
    Embedding matrix (n_samples, n_features).

Output:
    Analysis dict with k values, inertia, silhouette scores, and selected k.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_k_analysis(
    embeddings: np.ndarray,
    k_range: tuple[int, int],
    k_step: int,
    random_state: int,
) -> dict:
    """
    Compute inertia and silhouette scores for a range of k values.

    Args:
        embeddings: Input data matrix (n_samples, n_features).
        k_range: (min_k, max_k) range to evaluate.
        k_step: Step size for k values.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
        - k_range: The range evaluated
        - k_step: Step size used for k values
        - all_results: Dict mapping k -> {inertia, silhouette}
        - elbow_k: Detected elbow point
        - best_silhouette_k: K with highest silhouette score
        - best_silhouette_score: The highest silhouette score
    """
    min_k, max_k = k_range
    # Ensure max_k doesn't exceed n_samples - 1
    max_k = min(max_k, len(embeddings) - 1)

    k_values = list(range(min_k, max_k + 1, k_step))

    all_results = {}
    inertias = []
    silhouettes = []

    print(f"  Evaluating k from {min_k} to {max_k} (step={k_step})...")

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
        )
        labels = model.fit_predict(embeddings)

        inertia = float(model.inertia_)
        sil_score = float(silhouette_score(embeddings, labels))

        all_results[k] = {
            "inertia": inertia,
            "silhouette": sil_score,
        }
        inertias.append(inertia)
        silhouettes.append(sil_score)

    # Find elbow point
    elbow_k = detect_elbow_point(k_values, inertias)

    # Find best silhouette
    best_sil_idx = np.argmax(silhouettes)
    best_silhouette_k = k_values[best_sil_idx]
    best_silhouette_score = silhouettes[best_sil_idx]

    return {
        "k_range": list(k_range),
        "k_step": k_step,
        "all_results": all_results,
        "elbow_k": elbow_k,
        "best_silhouette_k": best_silhouette_k,
        "best_silhouette_score": best_silhouette_score,
    }


def detect_elbow_point(k_values: list[int], inertias: list[float]) -> int:
    """
    Detect elbow point using maximum perpendicular distance from line.

    Uses the "kneedle" approach: finds the point with maximum distance
    from the line connecting the first and last points of the curve.

    Args:
        k_values: List of k values evaluated.
        inertias: Corresponding inertia values.

    Returns:
        The k value at the detected elbow point.
    """
    if len(k_values) < 3:
        return k_values[0]

    # Normalize to [0, 1] for better distance calculation
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    # Handle edge case where all inertias are the same
    if y_range == 0:
        return k_values[0]

    x_norm = (x - x.min()) / x_range
    y_norm = (y - y.min()) / y_range

    # Line from first to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    # Calculate perpendicular distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return k_values[0]

    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        # Perpendicular distance from point to line
        d = np.abs(np.cross(line_vec, p1 - p)) / line_len
        distances.append(d)

    elbow_idx = int(np.argmax(distances))
    return k_values[elbow_idx]


def select_optimal_k(
    analysis: dict,
    selection_method: str,
    silhouette_weight: float = 0.5,
) -> int:
    """
    Select optimal k based on the specified method.

    Args:
        analysis: Analysis results from compute_k_analysis().
        selection_method: "elbow", "silhouette", or "combined".
        silhouette_weight: Weight for silhouette in combined method (0-1).
            0.0 = pure elbow, 1.0 = pure silhouette.

    Returns:
        The selected optimal k value.

    Raises:
        ValueError: If selection_method is not "elbow", "silhouette", or "combined".
    """
    if selection_method == "elbow":
        return int(analysis["elbow_k"])

    elif selection_method == "silhouette":
        return int(analysis["best_silhouette_k"])

    elif selection_method == "combined":
        return _select_combined(analysis, silhouette_weight)

    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def _select_combined(analysis: dict, silhouette_weight: float) -> int:
    """
    Select optimal k using weighted combination of elbow and silhouette.

    Scores each k based on:
    - Distance from elbow point (closer = better)
    - Silhouette score (higher = better)

    Args:
        analysis: Analysis results from compute_k_analysis().
        silhouette_weight: Weight for silhouette (0-1).

    Returns:
        The k with highest combined score.
    """
    all_results = analysis["all_results"]
    elbow_k = analysis["elbow_k"]

    k_values = list(all_results.keys())
    silhouettes = np.array([all_results[k]["silhouette"] for k in k_values])

    # Normalize silhouette scores to [0, 1]
    sil_range = silhouettes.max() - silhouettes.min()
    if sil_range > 0:
        sil_norm = (silhouettes - silhouettes.min()) / sil_range
    else:
        sil_norm = np.ones_like(silhouettes)

    # Calculate distance from elbow (closer = better)
    k_arr = np.array(k_values)
    elbow_dist = np.abs(k_arr - elbow_k)
    max_dist = elbow_dist.max()
    if max_dist > 0:
        elbow_score = 1 - (elbow_dist / max_dist)
    else:
        elbow_score = np.ones_like(elbow_dist, dtype=float)

    # Combined score
    elbow_weight = 1 - silhouette_weight
    combined_scores = elbow_weight * elbow_score + silhouette_weight * sil_norm

    # Round to eliminate floating-point noise (for deterministic results)
    combined_scores = np.round(combined_scores, decimals=6)

    # Find all k values with max score (handles ties deterministically)
    max_score = combined_scores.max()
    tied_indices = np.where(combined_scores == max_score)[0]

    # Tie-breaker: prefer smaller k (simpler, more interpretable model)
    best_idx = tied_indices[0]  # k_values is sorted ascending
    return int(k_values[best_idx])


def print_k_analysis_report(analysis: dict) -> None:
    """
    Print k-selection analysis results to console.

    Args:
        analysis: Analysis results from compute_k_analysis().
    """
    all_results = analysis["all_results"]
    elbow_k = analysis["elbow_k"]
    best_sil_k = analysis["best_silhouette_k"]
    best_sil_score = analysis["best_silhouette_score"]

    print("\n  K-Selection Analysis:")
    print("  " + "-" * 50)
    print(f"  {'k':>4}  {'Inertia':>12}  {'Silhouette':>10}  Notes")
    print("  " + "-" * 50)

    for k, metrics in all_results.items():
        inertia = metrics["inertia"]
        sil = metrics["silhouette"]

        notes = []
        if k == elbow_k:
            notes.append("elbow")
        if k == best_sil_k:
            notes.append("best silhouette")

        notes_str = f"  <-- {', '.join(notes)}" if notes else ""
        print(f"  {k:>4}  {inertia:>12.2f}  {sil:>10.4f}{notes_str}")

    print("  " + "-" * 50)
    print("\n  Summary:")
    print(f"    Elbow point detected at: k={elbow_k}")
    print(f"    Best silhouette score at: k={best_sil_k} ({best_sil_score:.4f})")
