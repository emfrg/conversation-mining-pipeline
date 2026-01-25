"""LLM Factory - Centralized LLM model creation.

Supports switching between Claude (via Vertex AI Model Garden) and Gemini.
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from config import config

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"


def get_llm() -> BaseChatModel:
    """Get the configured LLM model.

    Returns Claude (via ChatAnthropicVertex) or Gemini (via ChatGoogleGenerativeAI)
    based on config["llm"]["provider"].

    Returns:
        Configured chat model instance.
    """
    provider = config["llm"].get("provider", "claude")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("VERTEX_AI_LOCATION", "europe-west1")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            project=project,
            location=location,
            thinking_budget=0,  # Minimal thinking for faster responses
        )
    else:
        # Default to Claude via Vertex AI Model Garden
        from langchain_google_vertexai.model_garden import ChatAnthropicVertex

        return ChatAnthropicVertex(
            model_name=config["llm"]["model_name"],
            project=project,
            location=location,
        )
