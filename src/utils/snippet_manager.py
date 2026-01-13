"""Snippet Dataset Manager.

Handles creation, caching, and loading of snippet datasets.
When num_conversations is set, this module:
1. Creates a snippet dataset file with random sampling (saved to data/output/)
2. Rewrites config paths to include the snippet suffix
3. Scales clustering/PCA parameters proportionally to snippet size

Suffix format: _SNIPPET_{N}_conv (e.g., _SNIPPET_100_conv)
"""

import copy
import json
import random
from pathlib import Path

# Minimum values for scaled parameters to ensure meaningful results
MIN_N_CLUSTERS = 3
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1
MIN_PCA_COMPONENTS = 10


def get_snippet_suffix(num_conversations: int) -> str:
    """
    Generate standardized suffix based on num_conversations.

    Args:
        num_conversations: Number of conversations in the snippet.

    Returns:
        Suffix string like "_SNIPPET_100_conv"
    """
    return f"_SNIPPET_{num_conversations}_conv"


def get_snippet_dataset_path(base_raw_data_path: str, num_conversations: int) -> str:
    """
    Get the path where the snippet dataset should be saved.

    Snippets are saved to data/output/ to keep the dataset/ directory clean.

    Args:
        base_raw_data_path: Original raw_data path (e.g., "dataset/chat_history.json")
        num_conversations: Number of conversations in the snippet.

    Returns:
        Path like "data/output/chat_history_SNIPPET_100_conv.json"
    """
    p = Path(base_raw_data_path)
    suffix = get_snippet_suffix(num_conversations)
    # Save snippets to data/output/ instead of the original dataset directory
    return str(Path("data/output") / f"{p.stem}{suffix}{p.suffix}")


def apply_suffix_to_path(path: str, suffix: str) -> str:
    """
    Apply suffix to a file path before the extension.

    Args:
        path: Original path (e.g., "data/output/file.json")
        suffix: Suffix to add (e.g., "_SNIPPET_100_conv")

    Returns:
        Modified path (e.g., "data/output/file_SNIPPET_100_conv.json")
    """
    if not suffix:
        return path

    p = Path(path)
    return str(p.parent / f"{p.stem}{suffix}{p.suffix}")


def apply_suffix_to_name(name: str, suffix: str) -> str:
    """
    Apply suffix to a collection/resource name.

    Args:
        name: Original name (e.g., "faq_issues")
        suffix: Suffix to add (e.g., "_SNIPPET_100_conv")

    Returns:
        Modified name (e.g., "faq_issues_SNIPPET_100_conv")
    """
    if not suffix:
        return name
    return f"{name}{suffix}"


def apply_suffix_to_dir(dir_path: str, suffix: str) -> str:
    """
    Apply suffix to a directory path.

    Args:
        dir_path: Original directory path (e.g., "data/output/transcripts")
        suffix: Suffix to add (e.g., "_SNIPPET_100_conv")

    Returns:
        Modified path (e.g., "data/output/transcripts_SNIPPET_100_conv")
    """
    if not suffix:
        return dir_path
    return f"{dir_path}{suffix}"


def get_full_dataset_size(full_data_path: str) -> int:
    """
    Get the number of conversations in the full dataset.

    Args:
        full_data_path: Path to the full dataset.

    Returns:
        Number of conversations in the full dataset.
    """
    with open(full_data_path, encoding="utf-8") as f:
        full_data = json.load(f)
    return len(full_data)


def create_snippet_dataset(
    full_data_path: str,
    num_conversations: int,
    seed: int,
) -> tuple[str, int]:
    """
    Create a snippet dataset by randomly sampling N conversations.

    If the snippet file already exists, returns the existing path (caching).

    Args:
        full_data_path: Path to the full dataset (e.g., "data/dataset/chat_history.json")
        num_conversations: Number of conversations to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (path to snippet file, full dataset size).
    """
    snippet_path = get_snippet_dataset_path(full_data_path, num_conversations)

    # Load full dataset to get size (needed for parameter scaling)
    with open(full_data_path, encoding="utf-8") as f:
        full_data = json.load(f)
    full_size = len(full_data)

    # Check if already exists (caching)
    if Path(snippet_path).exists():
        print(f"  Using cached snippet: {snippet_path}")
        return snippet_path, full_size

    # Create new snippet
    print(f"  Creating snippet from {full_data_path}...")

    # Random sample with seed
    random.seed(seed)
    all_keys = list(full_data.keys())

    if num_conversations >= len(all_keys):
        print(
            f"  Warning: requested {num_conversations} but only {len(all_keys)} available"
        )
        sampled_keys = all_keys
    else:
        sampled_keys = random.sample(all_keys, num_conversations)

    # Create snippet dict
    snippet_data = {k: full_data[k] for k in sampled_keys}

    # Ensure parent directory exists
    Path(snippet_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to disk
    with open(snippet_path, "w", encoding="utf-8") as f:
        json.dump(snippet_data, f, indent=2, ensure_ascii=False)

    print(f"  Created snippet: {snippet_path} ({len(snippet_data)} conversations)")
    return snippet_path, full_size


def rewrite_config_paths(config: dict, suffix: str, snippet_path: str) -> dict:
    """
    Add suffix to all relevant path-based config values.

    Modifies:
    - paths.raw_data -> snippet dataset path (in data/output/)
    - paths.clean_conversations
    - paths.transcripts_dir
    - paths.issue_reports
    - paths.models_dir
    - paths.visualization_output
    - qdrant.embeddings_collection
    - qdrant.reduced_collection

    Args:
        config: Original config dictionary.
        suffix: Suffix to add (e.g., "_SNIPPET_100_conv")
        snippet_path: Path to the snippet dataset file.

    Returns:
        New config dictionary with modified paths.
    """
    new_config = copy.deepcopy(config)

    # Rewrite file paths
    paths = new_config["paths"]
    paths["raw_data"] = snippet_path  # Use the actual snippet path
    paths["clean_conversations"] = apply_suffix_to_path(
        paths["clean_conversations"], suffix
    )
    paths["transcripts_dir"] = apply_suffix_to_dir(paths["transcripts_dir"], suffix)
    paths["issue_reports"] = apply_suffix_to_path(paths["issue_reports"], suffix)
    paths["models_dir"] = apply_suffix_to_dir(paths["models_dir"], suffix)

    if "visualization_output" in paths:
        paths["visualization_output"] = apply_suffix_to_path(
            paths["visualization_output"], suffix
        )

    # Rewrite kmeans evaluation plot_output if present
    kmeans_config = new_config.get("clustering", {}).get("kmeans", {})
    kmeans_eval_config = kmeans_config.get("evaluation", {})
    if "plot_output" in kmeans_eval_config:
        kmeans_eval_config["plot_output"] = apply_suffix_to_path(
            kmeans_eval_config["plot_output"], suffix
        )

    # Rewrite hdbscan evaluation plot_output if present
    hdbscan_config = new_config.get("clustering", {}).get("hdbscan", {})
    hdbscan_eval_config = hdbscan_config.get("evaluation", {})
    if "plot_output" in hdbscan_eval_config:
        hdbscan_eval_config["plot_output"] = apply_suffix_to_path(
            hdbscan_eval_config["plot_output"], suffix
        )

    # Rewrite Qdrant collection names
    qdrant = new_config["qdrant"]
    qdrant["embeddings_collection"] = apply_suffix_to_name(
        qdrant["embeddings_collection"], suffix
    )
    qdrant["reduced_collection"] = apply_suffix_to_name(
        qdrant["reduced_collection"], suffix
    )

    return new_config


def scale_parameters(config: dict, snippet_size: int, full_size: int) -> dict:
    """
    Scale clustering and PCA parameters proportionally to snippet size.

    Applies ratio = snippet_size / full_size to:
    - clustering.kmeans.n_clusters
    - clustering.hdbscan.min_cluster_size
    - clustering.hdbscan.min_samples
    - pca.max_components

    Each parameter has a minimum threshold to ensure meaningful results.

    Args:
        config: Config dictionary (already modified with paths).
        snippet_size: Number of conversations in snippet.
        full_size: Number of conversations in full dataset.

    Returns:
        Config with scaled parameters.
    """
    if snippet_size >= full_size:
        # No scaling needed if using full dataset or more
        return config

    ratio = snippet_size / full_size
    print(
        f"  Scaling parameters by {ratio:.1%} (snippet/full = {snippet_size}/{full_size})"
    )

    # Scale kmeans n_clusters
    original_n_clusters = config["clustering"]["kmeans"]["n_clusters"]
    scaled_n_clusters = max(MIN_N_CLUSTERS, round(original_n_clusters * ratio))
    config["clustering"]["kmeans"]["n_clusters"] = scaled_n_clusters

    # Scale hdbscan min_cluster_size
    original_min_cluster_size = config["clustering"]["hdbscan"]["min_cluster_size"]
    scaled_min_cluster_size = max(
        MIN_CLUSTER_SIZE, round(original_min_cluster_size * ratio)
    )
    config["clustering"]["hdbscan"]["min_cluster_size"] = scaled_min_cluster_size

    # Scale hdbscan min_samples
    original_min_samples = config["clustering"]["hdbscan"]["min_samples"]
    scaled_min_samples = max(MIN_SAMPLES, round(original_min_samples * ratio))
    config["clustering"]["hdbscan"]["min_samples"] = scaled_min_samples

    # Scale pca max_components (also can't exceed sample count)
    original_max_components = config["pca"]["max_components"]
    scaled_max_components = max(
        MIN_PCA_COMPONENTS,
        min(snippet_size - 1, round(original_max_components * ratio)),
    )
    config["pca"]["max_components"] = scaled_max_components

    print(f"    n_clusters: {original_n_clusters} -> {scaled_n_clusters}")
    print(
        f"    min_cluster_size: {original_min_cluster_size} -> {scaled_min_cluster_size}"
    )
    print(f"    min_samples: {original_min_samples} -> {scaled_min_samples}")
    print(f"    max_components: {original_max_components} -> {scaled_max_components}")

    # Scale kmeans evaluation k_range if present
    kmeans_eval_config = config["clustering"]["kmeans"].get("evaluation", {})
    if "k_range" in kmeans_eval_config:
        original_k_range = kmeans_eval_config["k_range"]
        # Scale max_k, keep min_k reasonable
        scaled_max_k = max(5, round(original_k_range[1] * ratio))
        scaled_min_k = max(MIN_N_CLUSTERS, min(original_k_range[0], scaled_max_k - 2))
        kmeans_eval_config["k_range"] = [scaled_min_k, scaled_max_k]
        print(f"    k_range: {original_k_range} -> [{scaled_min_k}, {scaled_max_k}]")

    # Scale hdbscan evaluation min_cluster_size_range if present
    hdbscan_eval_config = config["clustering"]["hdbscan"].get("evaluation", {})
    if "min_cluster_size_range" in hdbscan_eval_config:
        original_range = hdbscan_eval_config["min_cluster_size_range"]
        # Scale max, keep min reasonable
        scaled_max = max(10, round(original_range[1] * ratio))
        scaled_min = max(MIN_CLUSTER_SIZE, min(original_range[0], scaled_max - 5))
        hdbscan_eval_config["min_cluster_size_range"] = [scaled_min, scaled_max]
        print(
            f"    min_cluster_size_range: {original_range} -> [{scaled_min}, {scaled_max}]"
        )

        # Also scale step if it would result in too few evaluations
        if "step" in hdbscan_eval_config:
            original_step = hdbscan_eval_config["step"]
            range_size = scaled_max - scaled_min
            # Ensure at least 5 evaluation points
            max_step = max(1, range_size // 5)
            scaled_step = min(original_step, max_step)
            if scaled_step != original_step:
                hdbscan_eval_config["step"] = scaled_step
                print(f"    step: {original_step} -> {scaled_step}")

    return config
