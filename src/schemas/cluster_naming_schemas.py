"""Pydantic schema for cluster naming agent structured output."""

from pydantic import BaseModel, Field


class ClusterName(BaseModel):
    """Structured output for cluster naming agent."""

    cluster_title: str = Field(description="Short descriptive title, 3-6 words")
    cluster_description: str = Field(
        description="Brief description of what this cluster is about"
    )
    cluster_detailed_description: str = Field(
        description="Detailed description with bullet points explaining the cluster"
    )
