"""Integration tests that run against a real local Ollama server, unmocked."""

import pytest
from pydantic import BaseModel

from src.llm.agents import build_agent
from src.llm.config import OllamaSettings
from src.llm.health import (
    OllamaUnavailableError,
    list_installed_models,
    require_ready,
)

pytestmark = pytest.mark.integration


class City(BaseModel):
    """Structured output small enough for a 0.6B model to fill in reliably."""

    name: str
    country: str


def test_the_tiny_model_is_installed_on_the_live_server(
    settings: OllamaSettings,
) -> None:
    assert settings.model_name in list_installed_models(settings)


async def test_agent_generates_text_from_the_live_model(
    settings: OllamaSettings,
) -> None:
    agent = build_agent(
        settings, instructions="Answer with a single word and nothing else."
    )

    result = await agent.run("What is the capital of France?")

    assert "paris" in result.output.lower()


async def test_agent_returns_validated_structured_output_from_the_live_model(
    settings: OllamaSettings,
) -> None:
    agent = build_agent(
        settings,
        instructions="Extract the city and the country it is in.",
        output_type=City,
    )

    result = await agent.run("The conference was held in Paris, France.")

    assert isinstance(result.output, City)
    assert result.output.name == "Paris"
    assert result.output.country == "France"


def test_require_ready_rejects_a_model_that_is_not_installed(
    settings: OllamaSettings,
) -> None:
    missing = OllamaSettings(
        base_url=settings.base_url, model_name="definitely-not-pulled:0b"
    )

    with pytest.raises(
        OllamaUnavailableError, match="ollama pull definitely-not-pulled:0b"
    ):
        require_ready(missing)
