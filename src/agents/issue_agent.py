"""Issue Agent - LLM agent for extracting structured issue reports from transcripts.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas.issue_schemas import IssueReport
from src.utils.prompts import (
    ISSUE_AGENT_HUMAN_TEMPLATE,
    ISSUE_SUMMARY_PROMPT,
    get_general_context,
)

load_dotenv()

# Create the chat model (model name from config)
llm = ChatAnthropicVertex(
    model_name=config["llm"]["model_name"],
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("VERTEX_AI_LOCATION", "europe-west1"),
)

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
