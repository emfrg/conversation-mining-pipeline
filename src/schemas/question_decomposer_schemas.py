"""Pydantic schema for question decomposer agent structured output."""

from typing import Literal

from pydantic import BaseModel, Field


class AtomicQuestion(BaseModel):
    """A single atomic question from decomposition."""

    text: str = Field(description="The atomic question text")
    type: Literal["explicit", "presupposition"] = Field(
        description="Whether this is an explicit question or a presupposition"
    )


class QuestionDecomposition(BaseModel):
    """Structured output for question decomposer agent."""

    question_type: Literal["compound", "simple"] = Field(
        description="Whether the original question was compound or simple"
    )
    atomic_questions: list[AtomicQuestion] = Field(
        description="List of atomic sub-questions"
    )
