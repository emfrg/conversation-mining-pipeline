"""FAQ Synthesizer Agent - LLM agent for generating clean FAQ questions.

Uses ChatAnthropicVertex with structured output for guaranteed schema compliance.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

from config import config
from src.schemas import FAQSynthesis
from src.utils.prompts import (
    FAQ_SYNTHESIZER_HUMAN_TEMPLATE,
    FAQ_SYNTHESIZER_SYSTEM_PROMPT,
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
            FAQ_SYNTHESIZER_SYSTEM_PROMPT.format(general_context=get_general_context()),
        ),
        ("human", FAQ_SYNTHESIZER_HUMAN_TEMPLATE),
    ]
)

# Create the chain with structured output
faq_synthesizer: Runnable[dict, FAQSynthesis] = prompt | llm.with_structured_output(
    FAQSynthesis
)
