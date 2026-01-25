"""Question Decomposer Agent - LLM agent for decomposing FAQ questions into atomic units.

Uses structured output for guaranteed schema compliance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import QuestionDecomposition
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    QUESTION_DECOMPOSER_HUMAN_TEMPLATE,
    QUESTION_DECOMPOSER_SYSTEM_PROMPT,
    QUESTION_DECOMPOSER_WITH_PRESUPPOSITIONS_SYSTEM_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()


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
