"""Pydantic schema for executive report agent structured output."""

from pydantic import BaseModel, Field


class ClusterAnalysis(BaseModel):
    """Analysis of a single cluster for executive report."""

    cluster_title: str = Field(description="Title of the cluster")
    insight: str = Field(description="What this tells us about user needs")
    recommendation: str = Field(description="What to do about it")


class MetricsSummary(BaseModel):
    """Summary of metrics for executive report."""

    strengths: list[str] = Field(description="2-3 positive metrics/trends")
    concerns: list[str] = Field(description="2-3 areas needing attention")


class ExecutiveReport(BaseModel):
    """Structured output for executive report agent."""

    executive_summary: str = Field(
        description="2-3 paragraphs summarizing overall performance"
    )
    key_findings: list[str] = Field(description="5-7 main findings as bullet points")
    top_clusters_analysis: list[ClusterAnalysis] = Field(
        description="Analysis of top 3-5 clusters by size"
    )
    recommendations: list[str] = Field(description="3-5 prioritized action items")
    metrics_summary: MetricsSummary = Field(
        description="Summary of strengths and concerns"
    )
