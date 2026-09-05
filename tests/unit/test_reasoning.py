"""Unit tests for pairing a reasoning trace with a structured answer.

Stubbed at pydantic-ai's ``FunctionModel`` seam, so these run without a server.
"""

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.llm.agents import build_agent
from src.llm.config import OllamaSettings
from src.llm.reasoning import (
    MissingReasoningError,
    ReasoningOverranError,
    run_reasoned,
)


class Choice(BaseModel):
    """The shape the generation pipeline asks a model to fill in."""

    answer: str


def responding(*parts: ModelResponsePart) -> FunctionModel:
    """A model that always replies with exactly ``parts``."""

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(parts))

    return FunctionModel(stub)


async def test_thinking_part_becomes_the_reasoning_beside_the_typed_answer() -> None:
    agent = build_agent(OllamaSettings(), output_type=Choice)
    reply = responding(
        ThinkingPart("Dolphins are mammals."), TextPart('{"answer": "B"}')
    )

    with agent.override(model=reply):
        result = await run_reasoned(agent, "Which of these is a mammal?")

    assert result.reasoning == "Dolphins are mammals."
    assert result.output == Choice(answer="B")


async def test_an_answer_with_no_reasoning_trace_is_rejected() -> None:
    """A row with an empty trace is useless downstream, so it fails loudly.

    The message names the model that actually answered, which is the thing the
    reader has to go and reconfigure.
    """
    agent = build_agent(OllamaSettings(), output_type=Choice)

    with agent.override(model=responding(TextPart('{"answer": "B"}'))):
        with pytest.raises(MissingReasoningError, match="function:stub"):
            await run_reasoned(agent, "Which of these is a mammal?")


async def test_whitespace_only_reasoning_counts_as_missing() -> None:
    agent = build_agent(OllamaSettings(), output_type=Choice)
    reply = responding(ThinkingPart("   \n  "), TextPart('{"answer": "B"}'))

    with agent.override(model=reply):
        with pytest.raises(MissingReasoningError):
            await run_reasoned(agent, "Which of these is a mammal?")


async def test_only_the_reasoning_behind_the_final_answer_is_kept() -> None:
    """A rejected attempt's thinking did not produce the answer we return."""
    attempts = 0

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return ModelResponse(
                parts=[
                    ThinkingPart("Salmon swim, so probably A."),
                    TextPart("I think it is A."),
                ]
            )
        return ModelResponse(
            parts=[
                ThinkingPart("Dolphins are mammals."),
                TextPart('{"answer": "B"}'),
            ]
        )

    agent = build_agent(OllamaSettings(), output_type=Choice)

    with agent.override(model=FunctionModel(flaky)):
        result = await run_reasoned(agent, "Which of these is a mammal?")

    assert attempts == 2
    assert result.reasoning == "Dolphins are mammals."


async def test_reasoning_streamed_as_several_parts_is_joined() -> None:
    agent = build_agent(OllamaSettings(), output_type=Choice)
    reply = responding(
        ThinkingPart("Salmon is a fish."),
        ThinkingPart("Dolphins are mammals."),
        TextPart('{"answer": "B"}'),
    )

    with agent.override(model=reply):
        result = await run_reasoned(agent, "Which of these is a mammal?")

    assert result.reasoning == "Salmon is a fish.\nDolphins are mammals."


async def test_the_prompt_actually_sent_to_the_model_is_captured() -> None:
    """Read back from the request, so it cannot drift from what was really sent."""
    agent = build_agent(
        OllamaSettings(),
        output_type=Choice,
        instructions="Answer with A or B.",
    )
    reply = responding(
        ThinkingPart("Dolphins are mammals."), TextPart('{"answer":"B"}')
    )

    with agent.override(model=reply):
        result = await run_reasoned(agent, "Which of these is a mammal?")

    assert result.prompt == "Answer with A or B.\n\nWhich of these is a mammal?"


async def test_prompt_is_just_the_question_when_there_are_no_instructions() -> None:
    agent = build_agent(OllamaSettings(), output_type=Choice)
    reply = responding(
        ThinkingPart("Dolphins are mammals."), TextPart('{"answer":"B"}')
    )

    with agent.override(model=reply):
        result = await run_reasoned(agent, "Which of these is a mammal?")

    assert result.prompt == "Which of these is a mammal?"


async def test_a_trace_that_ran_out_of_budget_is_rejected() -> None:
    """``finish_reason='length'`` means the trace was cut mid-thought."""
    agent = build_agent(OllamaSettings(), output_type=Choice)

    def truncated(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ThinkingPart("Neither option fits, so let me"),
                TextPart('{"answer":"B"}'),
            ],
            finish_reason="length",
        )

    with agent.override(model=FunctionModel(truncated)):
        with pytest.raises(ReasoningOverranError, match="budget"):
            await run_reasoned(agent, "What is the capital of France?")


async def test_overrunning_before_any_answer_is_reported_as_an_overrun() -> None:
    """The real failure: thinking never ends, so no answer is ever emitted.

    Without this the caller sees a bare validation failure and has no idea the
    model simply never stopped reasoning.
    """
    agent = build_agent(OllamaSettings(), output_type=Choice)

    def never_answers(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart("Wait, could this be a trick? Let me reconsider")],
            finish_reason="length",
        )

    with agent.override(model=FunctionModel(never_answers)):
        with pytest.raises(ReasoningOverranError, match="budget"):
            await run_reasoned(agent, "What is the capital of France?")
