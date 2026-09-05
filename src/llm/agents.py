"""Factories for pydantic-ai agents backed by a local Ollama server."""

from typing import Any, overload

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from src.llm.config import OllamaSettings

# Ollama ignores the key, but the OpenAI-compatible client insists on one.
PLACEHOLDER_API_KEY = "ollama"

# Reasoning traces are only comparable across runs if sampling is deterministic.
DETERMINISTIC_TEMPERATURE = 0.0


def build_model(settings: OllamaSettings) -> OpenAIChatModel:
    """Build a pydantic-ai model that talks to the configured Ollama server."""
    provider = OllamaProvider(
        base_url=settings.openai_base_url,
        api_key=PLACEHOLDER_API_KEY,
    )
    return OpenAIChatModel(settings.model_name, provider=provider)


@overload
def build_agent(
    settings: OllamaSettings,
    *,
    instructions: str | None = None,
    temperature: float = DETERMINISTIC_TEMPERATURE,
    max_tokens: int | None = None,
) -> Agent[None, str]: ...


@overload
def build_agent[OutputT](
    settings: OllamaSettings,
    *,
    output_type: type[OutputT],
    instructions: str | None = None,
    temperature: float = DETERMINISTIC_TEMPERATURE,
    max_tokens: int | None = None,
) -> Agent[None, OutputT]: ...


def build_agent(
    settings: OllamaSettings,
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
