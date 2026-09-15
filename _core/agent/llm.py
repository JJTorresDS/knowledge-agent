"""Ollama, OpenRouter, OpenAI, or Mistral client for the Agents SDK."""

from agents import Model, OpenAIChatCompletionsModel, OpenAIResponsesModel
from openai import AsyncOpenAI

from _core.config import (
    MISTRAL_BASE_URL,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    OPENROUTER_BASE_URL,
    _api_key,
    settings,
)


def build_model(*, provider: str | None = None, model: str | None = None) -> Model:
    provider = (provider or settings.llm_provider).strip().lower()
    model_name = model or settings.model
    api_key = (
        settings.api_key
        if provider == settings.llm_provider
        else _api_key(provider)
    )
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
    elif provider == "ollama":
        client = AsyncOpenAI(
            base_url=OLLAMA_BASE_URL, api_key=api_key or "ollama"
        )
    elif provider == "openrouter":
        client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
    elif provider == "mistral":
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=MISTRAL_BASE_URL,
        )
        return OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=client,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Options: ollama, openrouter, openai, mistral"
        )

    return OpenAIResponsesModel(
        model=model_name,
        openai_client=client,
    )
