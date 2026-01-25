"""Cluster Naming Agent - LLM agent for generating human-readable cluster names.

Uses structured output for guaranteed schema compliance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import ClusterName
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    CLUSTER_NAMER_HUMAN_TEMPLATE,
    CLUSTER_NAMER_SYSTEM_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

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
