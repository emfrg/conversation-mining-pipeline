"""Tagger Agent - LLM agent for extracting tags with pooling.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
This agent runs SEQUENTIALLY with tag pooling for consistent tagging.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import TaggerReport
from src.utils.prompts import (
    TAGGER_AGENT_HUMAN_TEMPLATE,
    TAGGER_AGENT_PROMPT,
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
        ("system", TAGGER_AGENT_PROMPT.format(general_context=get_general_context())),
        ("human", TAGGER_AGENT_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
tagger_extractor: Runnable[dict, TaggerReport] = prompt | llm.with_structured_output(
    TaggerReport
)
