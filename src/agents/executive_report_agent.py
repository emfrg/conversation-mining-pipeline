"""Executive Report Agent - LLM agent for generating executive reports.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import ExecutiveReport
from src.utils.prompts import (
    EXECUTIVE_REPORT_HUMAN_TEMPLATE,
    EXECUTIVE_REPORT_SYSTEM_PROMPT,
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
