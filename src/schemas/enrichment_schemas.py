"""Pydantic schema for enrichment agent structured output."""

from typing import Literal

from pydantic import BaseModel, Field


class EnrichmentReport(BaseModel):
    """Structured output for enrichment agent (sentiment, resolution, steps)."""

    steps_taken_by_agent: str = Field(description="What the agent tried or explained")
    resolution_status: Literal["resolved", "unresolved", "partially_resolved"] = Field(
        description="Whether the issue was resolved"
    )
    user_sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="User's sentiment based on the conversation"
    )
