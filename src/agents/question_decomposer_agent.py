"""Question Decomposer Agent - LLM agent for decomposing FAQ questions into atomic units.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import QuestionDecomposition
from src.utils.prompts import (
    QUESTION_DECOMPOSER_HUMAN_TEMPLATE,
    QUESTION_DECOMPOSER_SYSTEM_PROMPT,
    QUESTION_DECOMPOSER_WITH_PRESUPPOSITIONS_SYSTEM_PROMPT,
    get_general_context,
)

load_dotenv()

# Create the chat model (model name from config)
llm = ChatAnthropicVertex(
    model_name=config["llm"]["model_name"],
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("VERTEX_AI_LOCATION", "europe-west1"),
)


def get_question_decomposer(
    with_presuppositions: bool = False,
) -> Runnable[dict, QuestionDecomposition]:
    """Get the question decomposer chain with structured output.

    Args:
        with_presuppositions: If True, use the prompt that also extracts
            context-relevant presuppositions. If False (default), only
            decompose compound questions into atomic sub-questions.

    Returns:
        The configured question decomposer chain.
    """
    if with_presuppositions:
        system_prompt = QUESTION_DECOMPOSER_WITH_PRESUPPOSITIONS_SYSTEM_PROMPT.format(
            general_context=get_general_context()
        )
    else:
        system_prompt = QUESTION_DECOMPOSER_SYSTEM_PROMPT.format(
            general_context=get_general_context()
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", QUESTION_DECOMPOSER_HUMAN_TEMPLATE),
        ]
    )

    return prompt | llm.with_structured_output(QuestionDecomposition)
