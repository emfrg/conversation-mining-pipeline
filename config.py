"""Configuration Loader for FAQ Analytics Pipeline.

Loads settings from config.yaml and provides a simple dict interface.
Automatically applies snippet suffix to output paths when num_conversations is set.

Usage:
    from config import config

    # Access settings
    raw_data_path = config["paths"]["raw_data"]
    qdrant_host = config["qdrant"]["host"]
    clustering_method = config["clustering"]["method"]

    # Override snippet at runtime (for CLI)
    from config import get_config_with_snippet
    config = get_config_with_snippet(num_conversations=100)
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load raw configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.

    Returns:
        Raw configuration dictionary (no snippet processing applied).
    """
    if config_path is None:
        resolved_path = Path(__file__).parent / "config.yaml"
    else:
        resolved_path = Path(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    with open(resolved_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
        return config


def get_config_with_snippet(num_conversations: int | None = None) -> dict:
    """
    Get configuration with snippet processing applied.

    If num_conversations is provided (CLI override) or set in config,
    creates snippet dataset, rewrites all paths with suffix, and scales
    clustering/PCA parameters proportionally.

    Args:
        num_conversations: Optional override for snippet size.
                          If None, uses value from config.yaml.

    Returns:
        Configuration dictionary with snippet suffixes and scaled parameters.
    """
    raw_config = load_config()

    # Use CLI override if provided, otherwise use config value
    num_conv = num_conversations
    if num_conv is None:
        num_conv = raw_config.get("snippet", {}).get("num_conversations")

    # If no snippet requested, return raw config
    if num_conv is None:
        return raw_config

    # Import here to avoid circular imports
    from src.utils.snippet_manager import (
        create_snippet_dataset,
        get_snippet_suffix,
        rewrite_config_paths,
        scale_parameters,
    )

    seed = raw_config.get("snippet", {}).get("seed", 42)
    suffix = get_snippet_suffix(num_conv)

    # Create snippet dataset if needed (uses caching)
    # Returns tuple of (snippet_path, full_dataset_size)
    snippet_path, full_size = create_snippet_dataset(
        raw_config["paths"]["raw_data"],
        num_conv,
        seed,
    )

    # Rewrite all paths with snippet suffix
    config_with_snippet = rewrite_config_paths(raw_config, suffix, snippet_path)

    # Scale clustering/PCA parameters proportionally to snippet size
    config_with_snippet = scale_parameters(config_with_snippet, num_conv, full_size)

    # Store the effective num_conversations in config for display
    if "snippet" not in config_with_snippet:
        config_with_snippet["snippet"] = {}
    config_with_snippet["snippet"]["num_conversations"] = num_conv

    return config_with_snippet


# Load config on module import (uses config.yaml snippet setting)
config = get_config_with_snippet()
