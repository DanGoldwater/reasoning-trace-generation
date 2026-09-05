"""Factories for provider-neutral pydantic-ai agents."""

from typing import Any, overload

from openai import AsyncOpenAI
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from src.llm.config import (
    DEFAULT_GENERATION_MAX_TOKENS,
    AnthropicSettings,
    LLMSettings,
    OllamaSettings,
)

# Ollama ignores the key, but the OpenAI-compatible client insists on one.
PLACEHOLDER_API_KEY = "ollama"

# The OpenAI client retries a timed-out request twice by default, so a request
# the model was never going to finish costs three times the timeout before it
# is reported. A local server does not drop requests; it just runs long.
OLLAMA_REQUEST_RETRIES = 0

# Reasoning traces are only comparable across runs if sampling is deterministic.
DETERMINISTIC_TEMPERATURE = 0.0

# pydantic-ai defaults to sending the generation budget as
# `max_completion_tokens`, which Ollama accepts and silently ignores; Ollama
# reads `max_tokens`. Without this the budget never leaves the process, and a
# small model can think until the request times out. pydantic-ai's own
# OpenRouter and Azure providers set this flag; its Ollama provider does not.
OLLAMA_PROFILE_OVERRIDES = OpenAIModelProfile(
    openai_chat_supports_max_completion_tokens=False
)

# Kept as a public alias while generation limits move into provider settings.
DEFAULT_MAX_TOKENS = DEFAULT_GENERATION_MAX_TOKENS


@overload
def build_model(settings: OllamaSettings) -> OpenAIChatModel: ...


@overload
def build_model(settings: AnthropicSettings) -> AnthropicModel: ...


def build_model(settings: LLMSettings) -> OpenAIChatModel | AnthropicModel:
    """Build the Pydantic AI model for the selected provider."""
    if isinstance(settings, OllamaSettings):
        provider = OllamaProvider(
            openai_client=AsyncOpenAI(
                base_url=settings.openai_base_url,
                api_key=PLACEHOLDER_API_KEY,
                max_retries=OLLAMA_REQUEST_RETRIES,
            )
        )
        return OpenAIChatModel(
            settings.model_name,
            provider=provider,
            profile=OLLAMA_PROFILE_OVERRIDES,
        )

    provider = AnthropicProvider(api_key=settings.api_key)
    return AnthropicModel(settings.model_name, provider=provider)


@overload
def build_agent(
    settings: LLMSettings,
    *,
    instructions: str | None = None,
    temperature: float = DETERMINISTIC_TEMPERATURE,
    max_tokens: int | None = None,
) -> Agent[None, str]: ...


@overload
def build_agent[OutputT](
    settings: LLMSettings,
    *,
    output_type: type[OutputT],
    instructions: str | None = None,
    temperature: float = DETERMINISTIC_TEMPERATURE,
    max_tokens: int | None = None,
) -> Agent[None, OutputT]: ...


def build_agent(
    settings: LLMSettings,
    *,
    output_type: type[Any] = str,
    instructions: str | None = None,
    temperature: float = DETERMINISTIC_TEMPERATURE,
    max_tokens: int | None = None,
) -> Agent[None, Any]:
    """Build an agent that produces ``output_type`` from the configured model.

    Anything other than plain text is requested with schema-constrained decoding:
    a small local model will happily call an output tool with a prose string,
    but it cannot violate a schema the sampler itself enforces.

    ``max_tokens`` overrides the setting for exceptional calls. By default the
    generation budget, including the model's thinking, comes from ``settings``.
    """
    constrained: Any = str if output_type is str else NativeOutput(output_type)
    return Agent(
        build_model(settings),
        output_type=constrained,
        instructions=instructions,
        model_settings=ModelSettings(
            timeout=settings.timeout_seconds,
            temperature=temperature,
            max_tokens=(
                settings.generation_max_tokens if max_tokens is None else max_tokens
            ),
        ),
    )
