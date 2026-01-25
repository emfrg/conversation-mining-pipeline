"""Issue Agent - LLM agent for extracting structured issue reports from transcripts.

Uses structured output for guaranteed schema compliance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas.issue_schemas import IssueReport
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    ISSUE_AGENT_HUMAN_TEMPLATE,
    ISSUE_SUMMARY_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

# Create the prompt template with system message and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ISSUE_SUMMARY_PROMPT.format(general_context=get_general_context())),
        ("human", ISSUE_AGENT_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
# This uses tool-calling mode to guarantee valid IssueReport schema
issue_extractor: Runnable[dict, IssueReport] = prompt | llm.with_structured_output(
    IssueReport
)
