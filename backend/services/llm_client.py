"""
Thin wrapper around the Gemini API. Every agent that needs an LLM call goes
through this module instead of instantiating its own client — this is the
single place we'd swap providers, add retry logic, or add cost logging.
"""

import json
import logging

from google import genai
from google.genai import types

from configs.settings import settings

logger = logging.getLogger("raguard.llm")

_client: genai.Client | None = None
MODEL_NAME = "gemini-2.0-flash"


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def generate_text(prompt: str, system_instruction: str | None = None, temperature: float = 0.2) -> str:
    """Basic text generation call, used for reasoning/analysis steps."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def generate_json(prompt: str, system_instruction: str | None = None, temperature: float = 0.1) -> dict:
    """
    Generation call that forces JSON output, used whenever an agent needs
    structured data back (intent classification, entity lists, etc.)
    rather than free-form prose.
    """
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Failed to parse LLM JSON response: {e}. Raw: {response.text}")
        return {}