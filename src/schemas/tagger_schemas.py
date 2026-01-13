"""Pydantic schema for tagger agent structured output."""

from pydantic import BaseModel, Field


class TaggerReport(BaseModel):
    """Structured output for tagger agent."""

    tags: list[str] = Field(description="3-8 short tags categorizing the conversation")
