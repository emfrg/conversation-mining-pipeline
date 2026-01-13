"""Pydantic schema for FAQ synthesizer agent structured output."""

from pydantic import BaseModel, Field


class FAQSynthesis(BaseModel):
    """Structured output for FAQ synthesizer agent."""

    faq_question: str = Field(description="Synthesized FAQ question, 15-30 words max")
