"""Cluster Naming Agent - LLM agent for generating human-readable cluster names.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import ClusterName
from src.utils.prompts import (
    CLUSTER_NAMER_HUMAN_TEMPLATE,
    CLUSTER_NAMER_SYSTEM_PROMPT,
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
            CLUSTER_NAMER_SYSTEM_PROMPT.format(general_context=get_general_context()),
        ),
        ("human", CLUSTER_NAMER_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
cluster_namer: Runnable[dict, ClusterName] = prompt | llm.with_structured_output(
    ClusterName
)
