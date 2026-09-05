"""Unit tests for the pydantic-ai model and agent factories."""

from collections.abc import Mapping

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel

from src.llm.agents import build_agent, build_model
from src.settings import (
    DEFAULT_GENERATION_MAX_TOKENS,
    AnthropicSettings,
    OllamaSettings,
)


def model_settings_of[OutputT](agent: Agent[None, OutputT]) -> Mapping[str, object]:
    """Read back the settings the factory baked into an agent."""
    settings = agent.model_settings
    assert settings is not None
    assert not callable(settings)
    return settings


def test_model_is_pointed_at_the_configured_ollama_server() -> None:
    settings = OllamaSettings(
        base_url="http://ollama.internal:9999", model_name="gemma3:1b"
    )

    model = build_model(settings)

    assert model.model_name == "gemma3:1b"
    assert model.base_url.rstrip("/") == "http://ollama.internal:9999/v1"
    assert model.system == "ollama"


def test_model_is_pointed_at_anthropic_without_making_a_request() -> None:
    model = build_model(
        AnthropicSettings(api_key="test-key", model_name="claude-sonnet-4-5")
    )

    assert isinstance(model, AnthropicModel)
    assert model.model_name == "claude-sonnet-4-5"
    assert model.system == "anthropic"


async def test_anthropic_agent_uses_the_same_pydantic_ai_override_seam() -> None:
    """The provider is never contacted when a test supplies a FunctionModel."""
    agent = build_agent(AnthropicSettings(api_key="test-key"))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    with agent.override(model=FunctionModel(respond)):
        result = await agent.run("Hello?")

    assert result.output == "ok"


def test_agent_uses_the_configured_model_and_timeout() -> None:
    settings = OllamaSettings(model_name="qwen3:0.6b", timeout_seconds=30.0)

    agent = build_agent(settings)

    assert isinstance(agent.model, OpenAIChatModel)
    assert agent.model.model_name == "qwen3:0.6b"
    assert model_settings_of(agent).get("timeout") == 30.0


async def test_instructions_reach_the_model_on_every_run() -> None:
    """Observed through the request the agent actually sends, not a private field."""
    seen: list[str | None] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.append(request.instructions)
        return ModelResponse(parts=[TextPart("ok")])

    agent = build_agent(OllamaSettings(), instructions="Answer in one word.")

    with agent.override(model=FunctionModel(record)):
        await agent.run("Hello?")

    assert seen == ["Answer in one word."]


class Verdict(BaseModel):
    """A tiny structured output used to exercise typed agent results."""

    answer: str
    confidence: float


async def test_typed_output_is_validated_into_the_requested_model() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"answer": "B", "confidence": 0.5}')])

    agent = build_agent(OllamaSettings(), output_type=Verdict)

    with agent.override(model=FunctionModel(respond)):
        result = await agent.run("Which option?")

    assert result.output == Verdict(answer="B", confidence=0.5)


def test_agent_defaults_to_deterministic_sampling() -> None:
    """Reasoning traces must be reproducible, so temperature is pinned at zero."""
    agent = build_agent(OllamaSettings())

    assert model_settings_of(agent).get("temperature") == 0.0


def test_caller_can_raise_the_temperature() -> None:
    agent = build_agent(OllamaSettings(), temperature=0.7)

    assert model_settings_of(agent).get("temperature") == 0.7


async def test_malformed_structured_output_is_retried_rather_than_raised() -> None:
    """Small models sometimes answer in prose; the agent gets another attempt."""
    attempts = 0

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse(
                parts=[TextPart("The answer is B, quite confidently.")]
            )
        return ModelResponse(parts=[TextPart('{"answer": "B", "confidence": 0.5}')])

    agent = build_agent(OllamaSettings(), output_type=Verdict)

    with agent.override(model=FunctionModel(flaky)):
        result = await agent.run("Which option?")

    assert attempts == 2
    assert result.output == Verdict(answer="B", confidence=0.5)


async def test_structured_output_uses_schema_constrained_decoding() -> None:
    """Tiny models fill a constrained schema far more reliably than a tool call."""
    modes: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        modes.append(info.model_request_parameters.output_mode)
        return ModelResponse(parts=[TextPart('{"answer": "B", "confidence": 0.5}')])

    agent = build_agent(OllamaSettings(), output_type=Verdict)

    with agent.override(model=FunctionModel(record)):
        await agent.run("Which option?")

    assert modes == ["native"]


async def test_plain_text_output_is_left_unconstrained() -> None:
    modes: list[str] = []

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        modes.append(info.model_request_parameters.output_mode)
        return ModelResponse(parts=[TextPart("plain answer")])

    agent = build_agent(OllamaSettings())

    with agent.override(model=FunctionModel(record)):
        await agent.run("Which option?")

    assert modes == ["text"]


def test_agent_bounds_generation_by_default() -> None:
    """Nothing constrains the thinking channel, so only a token budget bounds it.

    Without this, a question the model finds contradictory spirals in ``<think>``
    until it exhausts the context, costing the whole request timeout.
    """
    agent = build_agent(OllamaSettings())

    assert model_settings_of(agent).get("max_tokens") == DEFAULT_GENERATION_MAX_TOKENS


def test_agent_uses_the_generation_budget_from_its_settings() -> None:
    agent = build_agent(OllamaSettings(generation_max_tokens=123))

    assert model_settings_of(agent).get("max_tokens") == 123


def test_caller_can_widen_the_reasoning_budget() -> None:
    agent = build_agent(OllamaSettings(), max_tokens=4096)

    assert model_settings_of(agent).get("max_tokens") == 4096


def test_the_ollama_budget_is_sent_in_the_field_ollama_actually_reads() -> None:
    """Ollama honours ``max_tokens`` and silently ignores ``max_completion_tokens``.

    pydantic-ai picks between those two fields from this profile flag, and its
    default sends the one Ollama ignores — which leaves the budget above
    decorative, and lets a small model think until the request times out.
    """
    profile = build_model(OllamaSettings()).profile

    assert profile is not None
    assert profile.get("openai_chat_supports_max_completion_tokens") is False
