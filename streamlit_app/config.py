"""Configuration for the Streamlit app."""

from pathlib import Path

import yaml

# Base paths - relative to project root (parent of streamlit_app)
PROJECT_ROOT = Path(__file__).parent.parent

# Load main config to get clustering method
_config_path = PROJECT_ROOT / "config.yaml"
with open(_config_path) as f:
    _main_config = yaml.safe_load(f)

# Get clustering method from main config
CLUSTERING_METHOD = _main_config.get("clustering", {}).get("method", "kmeans")

# Get project name from domain config (used in page titles and charts)
PROJECT_NAME = _main_config.get("domain", {}).get("project_name", "FAQ Analytics")

# Data paths
STATS_PATH = PROJECT_ROOT / "data" / "output" / "chat_history_stats.json"
# Use {method}_latest symlink to always point to most recent experiment
CLUSTER_DATA_PATH = PROJECT_ROOT / "data" / "models" / f"{CLUSTERING_METHOD}_latest"
EXECUTIVE_REPORT_PATH = PROJECT_ROOT / "data" / "output" / "executive_report.json"
ISSUE_REPORTS_PATH = PROJECT_ROOT / "data" / "output" / "issue_reports.json"
CLEAN_CONVERSATIONS_PATH = PROJECT_ROOT / "data" / "output" / "clean_conversations.json"
VIS_OUTPUTS_PATH = PROJECT_ROOT / "src" / "vis" / "outputs"
ASSETS_PATH = PROJECT_ROOT / "assets"

# App settings
APP_TITLE = "FAQ Analytics Dashboard"
APP_ICON = ":bar_chart:"
