"""Visualization modules for FAQ Analytics.

All visualization functions accept data as parameters and return chart objects
or save to provided paths.
"""

import matplotlib

matplotlib.use("Agg")

from src.vis.bar_chart import create_cluster_bar_chart
from src.vis.resolution_status import create_resolution_chart
from src.vis.scatter_plot import create_cluster_scatter_plot
from src.vis.sentiment_analysis import create_sentiment_chart
from src.vis.tag_wordcloud import create_tag_wordcloud
from src.vis.tool_use import create_tool_use_chart

__all__ = [
    "create_cluster_bar_chart",
    "create_cluster_scatter_plot",
    "create_resolution_chart",
    "create_sentiment_chart",
    "create_tag_wordcloud",
    "create_tool_use_chart",
]
