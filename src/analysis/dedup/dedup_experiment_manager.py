"""Experiment Manager for Deduplication Pipeline.

Handles timestamped run directories for deduplication experiments.
Each run creates a unique folder: src/vis/outputs/dedup_runs/{YYYYMMDD_HHMMSS}/

Contents of each run folder:
- config_snapshot.json          (deduplication config used)
- decomposed_questions.json     (pass 1 decomposition output)
- deduplicated_questions.json   (pass 1 deduplication output)
- decomposed_synthesized.json   (pass 2 decomposition, if enabled)
- deduplicated_synthesized.json (pass 2 deduplication, if enabled)
- faq_summary.json              (final clean export)
"""

import json
import os
from datetime import datetime


def generate_run_id() -> str:
    """Generate run ID with format YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_run_dir(base_dir: str, run_id: str, mode: str = "within-cluster") -> str:
    """Construct run directory path.

    Args:
        base_dir: Base output directory (e.g., "src/vis/outputs/dedup_runs")
        run_id: Timestamp string (e.g., "20251209_143022")
        mode: "within-cluster" or "global"

    Returns:
        Full path like "src/vis/outputs/dedup_runs/20251209_143022"
        or "src/vis/outputs/dedup_runs/20251209_143022_global"
    """
    suffix = "_global" if mode == "global" else ""
    return os.path.join(base_dir, f"{run_id}{suffix}")


def setup_run(config: dict, mode: str = "within-cluster") -> str:
    """Initialize run directory and update config with run paths.

    Creates the run directory and stores paths in config["dedup_run"]
    for use by deduplication pipeline steps.

    Args:
        config: The global config dictionary (will be modified in place)
        mode: "within-cluster" or "global"

    Returns:
        The run_id string
    """
    base_dir = config.get("deduplication", {}).get(
        "output_base_dir", "src/analysis/outputs"
    )

    run_id = generate_run_id()
    run_dir = get_run_dir(base_dir, run_id, mode)

    # Create run directory
    os.makedirs(run_dir, exist_ok=True)

    # Store run info in config for pipeline steps to access
    config["dedup_run"] = {
        "id": run_id,
        "dir": run_dir,
        "mode": mode,
    }

    print(f"  Run: {os.path.basename(run_dir)}")

    return run_id


def save_config_snapshot(
    config: dict,
    run_dir: str,
    with_presuppositions: bool,
    second_pass: bool,
    threshold: float,
    second_pass_threshold: float,
    input_clusters_path: str,
) -> str:
    """Save configuration snapshot to run directory.

    Captures the deduplication configuration used for this run
    to enable reproducibility and comparison.

    Args:
        config: The global config dictionary
        run_dir: Path to the run directory
        with_presuppositions: Whether presuppositions were enabled
        second_pass: Whether second pass was enabled
        threshold: Similarity threshold used for pass 1
        second_pass_threshold: Similarity threshold for pass 2
        input_clusters_path: Path to input clusters_named.json

    Returns:
        Path to saved config snapshot
    """
    dedup_config = config.get("deduplication", {})
    mode = config.get("dedup_run", {}).get("mode", "within-cluster")

    snapshot = {
        "run_id": config.get("dedup_run", {}).get("id", ""),
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "with_presuppositions": with_presuppositions,
        "second_pass_enabled": second_pass,
        "similarity_threshold": threshold,
        "second_pass_threshold": second_pass_threshold,
        "embedding_model": dedup_config.get("embedding_model", "text-embedding-005"),
        "embedding_dimensions": dedup_config.get("dimensions", 768),
        "input_clusters_path": input_clusters_path,
    }

    snapshot_path = os.path.join(run_dir, "config_snapshot.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return snapshot_path


def update_latest_symlink(run_dir: str, mode: str = "within-cluster") -> None:
    """Create/update 'latest' symlink pointing to current run.

    Creates a symlink at dedup_runs/latest -> {timestamp}
    or dedup_runs/latest_global -> {timestamp}_global
    for convenient access to most recent results.

    Args:
        run_dir: Path to the current run directory
        mode: "within-cluster" or "global"
    """
    base_dir = os.path.dirname(run_dir)
    symlink_name = "latest_global" if mode == "global" else "latest"
    latest_link = os.path.join(base_dir, symlink_name)

    # Remove existing symlink if present
    if os.path.islink(latest_link):
        os.remove(latest_link)
    elif os.path.exists(latest_link):
        # It's a real directory - don't remove
        print(f"  Warning: {latest_link} exists as directory, not updating symlink")
        return

    # Create relative symlink (so it works if repo is moved)
    run_name = os.path.basename(run_dir)
    os.symlink(run_name, latest_link)
    print(f"  Symlink: {symlink_name} -> {run_name}")


def get_run_output_path(run_dir: str, filename: str) -> str:
    """Get full path for a file within the current run directory.

    Args:
        run_dir: Path to the run directory
        filename: Name of the file (e.g., "decomposed_questions.json")

    Returns:
        Full path like "src/vis/outputs/dedup_runs/20251209_143022/decomposed_questions.json"
    """
    return os.path.join(run_dir, filename)


def get_latest_run_dir(config: dict, mode: str = "within-cluster") -> str | None:
    """Get the latest run directory for a given mode.

    Args:
        config: The global config dictionary
        mode: "within-cluster" or "global"

    Returns:
        Path to the latest run directory, or None if not found
    """
    base_dir = config.get("deduplication", {}).get(
        "output_base_dir", "src/analysis/outputs"
    )
    symlink_name = "latest_global" if mode == "global" else "latest"
    latest_link = os.path.join(base_dir, symlink_name)

    if not os.path.exists(latest_link):
        return None

    # Resolve symlink to get actual directory
    return os.path.realpath(latest_link)


def list_runs(config: dict, mode: str | None = None) -> list[dict]:
    """List all deduplication runs.

    Args:
        config: The global config dictionary
        mode: Filter by mode ("within-cluster" or "global"), or None for all

    Returns:
        List of run info dicts with id, dir, mode, timestamp
    """
    base_dir = config.get("deduplication", {}).get(
        "output_base_dir", "src/analysis/outputs"
    )

    if not os.path.exists(base_dir):
        return []

    runs = []
    for name in os.listdir(base_dir):
        if name in ("latest", "latest_global"):
            continue

        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue

        # Determine mode from name
        run_mode = "global" if name.endswith("_global") else "within-cluster"
        if mode is not None and run_mode != mode:
            continue

        # Extract run_id
        run_id = name.replace("_global", "") if run_mode == "global" else name

        # Load config snapshot if available
        snapshot_path = os.path.join(run_dir, "config_snapshot.json")
        timestamp = None
        if os.path.exists(snapshot_path):
            with open(snapshot_path) as f:
                snapshot = json.load(f)
                timestamp = snapshot.get("timestamp")

        runs.append(
            {
                "id": run_id,
                "dir": run_dir,
                "mode": run_mode,
                "timestamp": timestamp,
            }
        )

    # Sort by id (timestamp) descending
    runs.sort(key=lambda x: x["id"], reverse=True)
    return runs
