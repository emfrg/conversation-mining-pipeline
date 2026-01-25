"""Enrichment Agent - LLM agent for extracting sentiment, resolution, and steps.

Uses structured output for guaranteed schema compliance.
This agent runs in PARALLEL with the primary agent for fast processing.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import EnrichmentReport
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    ENRICHMENT_AGENT_HUMAN_TEMPLATE,
    ENRICHMENT_AGENT_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ENRICHMENT_AGENT_PROMPT.format(general_context=get_general_context()),
        ),
        ("human", ENRICHMENT_AGENT_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
enrichment_extractor: Runnable[dict, EnrichmentReport] = (
    prompt | llm.with_structured_output(EnrichmentReport)
)
