"""Tagger Agent - LLM agent for extracting tags with pooling.

Uses structured output for guaranteed schema compliance.
This agent runs SEQUENTIALLY with tag pooling for consistent tagging.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import TaggerReport
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    TAGGER_AGENT_HUMAN_TEMPLATE,
    TAGGER_AGENT_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", TAGGER_AGENT_PROMPT.format(general_context=get_general_context())),
        ("human", TAGGER_AGENT_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
tagger_extractor: Runnable[dict, TaggerReport] = prompt | llm.with_structured_output(
    TaggerReport
)
