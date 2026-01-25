"""FAQ Synthesizer Agent - LLM agent for generating clean FAQ questions.

Uses structured output for guaranteed schema compliance.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.schemas import FAQSynthesis
from src.utils.llm_factory import get_llm
from src.utils.prompts import (
    FAQ_SYNTHESIZER_HUMAN_TEMPLATE,
    FAQ_SYNTHESIZER_SYSTEM_PROMPT,
    get_general_context,
)

# Create the chat model (provider from config)
llm = get_llm()

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
