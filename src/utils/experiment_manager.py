"""
Experiment Manager for Clustering Pipeline

Handles timestamped experiment directories for clustering experiments.
Each experiment creates a unique folder: data/models/{method}_{YYYYMMDD_HHMMSS}/

Contents of each experiment folder:
- config_snapshot.json          (clustering config used)
- model.joblib                  (trained clustering model)
- labels.json                   (point ID -> cluster label mapping)
- metadata.json                 (experiment metadata and metrics)
- clusters_readable.json        (human-readable cluster contents)
- clusters_named.json           (LLM-generated cluster names)
- faq_clusters_{method}.png     (bar chart visualization)
- faq_clusters_scatter_{method}.png  (scatter plot visualization)
- {method}_analysis.png         (k-selection/parameter analysis chart, if enabled)
"""

import json
import os
from datetime import datetime


def generate_experiment_id() -> str:
    """Generate experiment ID with format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(base_models_dir: str, method: str, experiment_id: str) -> str:
    """
    Construct experiment directory path.

    Args:
        base_models_dir: Base models directory (e.g., "data/models")
        method: Clustering method (e.g., "hdbscan" or "kmeans")
        experiment_id: Timestamp string (e.g., "20251204_143022")

    Returns:
        Full path like "data/models/hdbscan_20251204_143022"
    """
    return os.path.join(base_models_dir, f"{method}_{experiment_id}")


def setup_experiment(config: dict) -> str:
    """
    Initialize experiment directory and update config with experiment paths.

    This function should be called ONCE before step 5 (cluster_embeddings) runs.
    It creates the experiment directory and stores paths in config["experiment"]
    for use by steps 5, 6, and 7.

    Args:
        config: The global config dictionary (will be modified in place)

    Returns:
        The experiment_id string
    """
    method = config["clustering"]["method"]
    base_models_dir = config["paths"]["models_dir"]

    experiment_id = generate_experiment_id()
    experiment_dir = get_experiment_dir(base_models_dir, method, experiment_id)

    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Store experiment info in config for steps 5-7 to access
    config["experiment"] = {
        "id": experiment_id,
        "dir": experiment_dir,
        "method": method,
    }

    print(f"  Experiment: {os.path.basename(experiment_dir)}")

    return experiment_id


def save_config_snapshot(config: dict) -> str:
    """
    Save configuration snapshot to experiment directory.

    Captures the clustering configuration used for this experiment
    to enable reproducibility and comparison.

    Args:
        config: The global config dictionary (must have config["experiment"] set)

    Returns:
        Path to saved config snapshot
    """
    experiment_dir = config["experiment"]["dir"]
    method = config["experiment"]["method"]

    snapshot = {
        "experiment_id": config["experiment"]["id"],
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "clustering": config["clustering"],
        "pca": config.get("pca", {}),
        "source_collections": {
            "embeddings": config["qdrant"]["embeddings_collection"],
            "reduced": config["qdrant"]["reduced_collection"],
        },
        "snippet_mode": config.get("snippet", {}).get("num_conversations") is not None,
    }

    snapshot_path = os.path.join(experiment_dir, "config_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot_path


def update_latest_symlink(config: dict) -> None:
    """
    Create/update 'latest' symlink pointing to current experiment.

    Creates a symlink at data/models/{method}_latest -> {method}_{timestamp}
    for convenient access to most recent results.

    Args:
        config: The global config dictionary (must have config["experiment"] set)
    """
    base_models_dir = config["paths"]["models_dir"]
    method = config["experiment"]["method"]
    experiment_dir = config["experiment"]["dir"]

    latest_link = os.path.join(base_models_dir, f"{method}_latest")

    # Remove existing symlink if present
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        # It's a real directory (legacy from before timestamping)
        # Don't remove it to preserve old data
        print(f"  Warning: {latest_link} exists as directory, not updating symlink")
        return

    # Create relative symlink (so it works if repo is moved)
    experiment_name = os.path.basename(experiment_dir)
    os.symlink(experiment_name, latest_link)
    print(f"  Symlink: {method}_latest -> {experiment_name}")


def get_experiment_output_path(config: dict, filename: str) -> str:
    """
    Get full path for a file within the current experiment directory.

    Helper function for steps 5-7 to construct output paths.

    Args:
        config: The global config dictionary (must have config["experiment"] set)
        filename: Name of the file (e.g., "model.joblib", "clusters_named.json")

    Returns:
        Full path like "data/models/hdbscan_20251204_143022/model.joblib"
    """
    return os.path.join(config["experiment"]["dir"], filename)


def get_experiment_config_key(config: dict) -> dict:
    """
    Extract config fields that define experiment identity.

    Two experiments with the same config key are considered equivalent
    and can be reused instead of creating a new one.
    """
    return {
        "method": config.get("clustering", {}).get("method"),
        "clustering": config.get("clustering", {}),
        "pca": config.get("pca", {}),
        "snippet_mode": config.get("snippet", {}).get("num_conversations"),
    }


def configs_match(current_config: dict, saved_snapshot: dict) -> bool:
    """
    Compare current config with a saved config snapshot for experiment equivalence.

    Args:
        current_config: The current runtime config
        saved_snapshot: A config_snapshot.json loaded from an experiment directory

    Returns:
        True if configs match and experiment can be reused
    """
    current_key = get_experiment_config_key(current_config)

    # The saved snapshot has a different structure - extract comparable fields
    saved_key = {
        "method": saved_snapshot.get("method"),
        "clustering": saved_snapshot.get("clustering", {}),
        "pca": saved_snapshot.get("pca", {}),
        "snippet_mode": saved_snapshot.get("snippet_mode"),
    }

    return current_key == saved_key


def find_matching_experiment(config: dict) -> str | None:
    """
    Search for existing experiment with matching config.

    Searches experiment directories (newest first) for one with
    a config_snapshot.json that matches the current config.

    Args:
        config: The current runtime config

    Returns:
        Experiment directory path if found, None otherwise
    """
    method = config["clustering"]["method"]
    base_dir = config["paths"]["models_dir"]

    if not os.path.exists(base_dir):
        return None

    # List all experiment dirs for this method, newest first
    experiment_dirs = []
    for name in os.listdir(base_dir):
        if name.startswith(f"{method}_") and not name.endswith("_latest"):
            exp_dir = os.path.join(base_dir, name)
            if os.path.isdir(exp_dir):
                experiment_dirs.append(name)

    # Sort by timestamp (newest first)
    experiment_dirs.sort(reverse=True)

    for name in experiment_dirs:
        exp_dir = os.path.join(base_dir, name)
        snapshot_path = os.path.join(exp_dir, "config_snapshot.json")
        if os.path.exists(snapshot_path):
            with open(snapshot_path) as f:
                saved_config = json.load(f)
            if configs_match(config, saved_config):
                return str(exp_dir)

    return None


def setup_experiment_from_existing(config: dict, experiment_dir: str) -> str:
    """
    Set up config to use an existing experiment directory.

    Used when a matching experiment is found and can be reused.

    Args:
        config: The global config dictionary (will be modified in place)
        experiment_dir: Path to the existing experiment directory

    Returns:
        The experiment_id string
    """
    method = config["clustering"]["method"]
    experiment_name = os.path.basename(experiment_dir)
    experiment_id = experiment_name.replace(f"{method}_", "")

    config["experiment"] = {
        "id": experiment_id,
        "dir": experiment_dir,
        "method": method,
    }

    print(f"  Reusing experiment: {experiment_name} (config matches)")

    return experiment_id


def setup_experiment_from_latest(config: dict) -> str | None:
    """
    Use existing 'latest' symlink instead of creating a new experiment.

    Used when running only visualize_clusters to reuse existing experiment.

    Args:
        config: The global config dictionary (will be modified in place)

    Returns:
        The experiment_id string if found, None if no latest symlink exists
    """
    method = config["clustering"]["method"]
    base_models_dir = config["paths"]["models_dir"]

    latest_link = os.path.join(base_models_dir, f"{method}_latest")

    # Check if latest symlink exists
    if not os.path.exists(latest_link):
        return None

    # Resolve the symlink to get actual experiment directory
    experiment_dir = os.path.realpath(latest_link)

    if not os.path.isdir(experiment_dir):
        return None

    # Extract experiment_id from directory name (e.g., "kmeans_20251208_123456")
    experiment_name = os.path.basename(experiment_dir)
    experiment_id = experiment_name.replace(f"{method}_", "")

    # Store experiment info in config
    config["experiment"] = {
        "id": experiment_id,
        "dir": experiment_dir,
        "method": method,
    }

    print(f"  Using existing experiment: {experiment_name}")

    return experiment_id
