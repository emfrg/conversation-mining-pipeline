"""Pydantic schemas for issue extraction agents."""

from typing import Literal

from pydantic import BaseModel, Field


class IssueReport(BaseModel):
    """Structured output for issue extraction from transcripts."""

    issue_title: str = Field(description="Short title, 5-12 words, like an FAQ title")
    user_problem: str = Field(
        description="What is going wrong from the user's perspective"
    )
    user_goal: str = Field(description="What the user ultimately wants to achieve")
    important_context: str = Field(
        description="Key constraints, environment, or background details"
    )
    canonical_faq_question: str = Field(
        description="Natural FAQ-style question capturing the main issue"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence that this is a single, well-understood issue"
    )
