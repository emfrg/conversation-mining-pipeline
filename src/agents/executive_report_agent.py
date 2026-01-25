"""Executive Report Agent - LLM agent for generating executive reports.

Uses structured output for guaranteed schema compliance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import ExecutiveReport
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    EXECUTIVE_REPORT_HUMAN_TEMPLATE,
    EXECUTIVE_REPORT_SYSTEM_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            EXECUTIVE_REPORT_SYSTEM_PROMPT.format(
                general_context=get_general_context()
            ),
        ),
        ("human", EXECUTIVE_REPORT_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
executive_report_generator: Runnable[dict, ExecutiveReport] = (
    prompt | llm.with_structured_output(ExecutiveReport)
)
