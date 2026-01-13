"""Enrichment Agent - LLM agent for extracting sentiment, resolution, and steps.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
This agent runs in PARALLEL with the primary agent for fast processing.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import EnrichmentReport
from src.utils.prompts import (
    ENRICHMENT_AGENT_HUMAN_TEMPLATE,
    ENRICHMENT_AGENT_PROMPT,
    get_general_context,
)

load_dotenv()

# Create the chat model (model name from config)
llm = ChatAnthropicVertex(
    model_name=config["llm"]["model_name"],
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("VERTEX_AI_LOCATION", "europe-west1"),
)

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
