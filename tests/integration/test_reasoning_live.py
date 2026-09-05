"""Reasoning traces and structured answers from a real local Ollama server.

Nothing here is mocked: the point is to prove that a small local model really
does hand back a trace *and* a schema-valid answer in the same call, and to pin
down the one case where it cannot.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from src.llm.agents import DEFAULT_MAX_TOKENS, build_agent
from src.llm.config import OllamaSettings
from src.llm.reasoning import Reasoned, ReasoningOverranError, run_reasoned

pytestmark = pytest.mark.integration

FORCED_CHOICE = "Decide which option is correct. Answer with its key, A or B."


class Choice(BaseModel):
    """The A/B choice the generation pipeline asks a model to commit to."""

    answer: Literal["A", "B"]


async def ask(
    settings: OllamaSettings,
    question: str,
    option_a: str,
    option_b: str,
    *,
    instructions: str = FORCED_CHOICE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Reasoned[Choice]:
    """Put a two-option question to the live model, as the pipeline will."""
    agent = build_agent(
        settings,
        output_type=Choice,
        instructions=instructions,
        max_tokens=max_tokens,
    )
    return await run_reasoned(agent, f"{question}\nA: {option_a}\nB: {option_b}")


async def a_mammal(
    settings: OllamaSettings,
    *,
    instructions: str = FORCED_CHOICE,
) -> Reasoned[Choice]:
    """The reference question: B is unambiguously correct."""
    return await ask(
        settings,
        "Which of these is a mammal?",
        "Salmon",
        "Dolphin",
        instructions=instructions,
    )


async def test_a_trace_and_a_schema_valid_answer_arrive_from_one_call(
    settings: OllamaSettings,
) -> None:
    result = await a_mammal(settings)

    assert result.output == Choice(answer="B")
    assert result.reasoning.strip()


async def test_the_trace_is_stored_without_its_think_tags(
    settings: OllamaSettings,
) -> None:
    result = await a_mammal(settings)

    assert "<think>" not in result.reasoning
    assert "</think>" not in result.reasoning


async def test_the_reasoning_discusses_the_option_it_settled_on(
    settings: OllamaSettings,
) -> None:
    """A trace that never mentions the winning option did not produce the answer."""
    result = await a_mammal(settings)

    assert "dolphin" in result.reasoning.lower()


async def test_the_schema_beats_an_instruction_that_contradicts_it(
    settings: OllamaSettings,
) -> None:
    """Negative case: told to answer in prose, the model must still emit a key.

    Constrained decoding governs the answer channel, so the instruction loses.
    """
    result = await a_mammal(
        settings,
        instructions=(
            f"{FORCED_CHOICE}\nAnswer with the full name of the winning option, "
            "spelled out in words. Never reply with a bare letter."
        ),
    )

    assert result.output.answer in {"A", "B"}


async def test_a_question_with_no_correct_option_overruns_rather_than_guessing(
    settings: OllamaSettings,
) -> None:
    """The limitation, pinned deliberately.

    Only the answer channel is constrained; thinking runs free. Asked something
    no option answers, this model reasons in circles instead of committing, and
    never reaches the constrained answer at all. The budget is what stops it, so
    the pipeline gets a named error instead of a request timeout.
    """
    with pytest.raises(ReasoningOverranError, match="token budget"):
        await ask(
            settings,
            "What is the capital of France?",
            "Salmon",
            "Dolphin",
            max_tokens=256,
        )


async def test_the_recorded_prompt_is_the_one_the_model_was_given(
    settings: OllamaSettings,
) -> None:
    result = await a_mammal(settings)

    assert FORCED_CHOICE in result.prompt
    assert "A: Salmon" in result.prompt
