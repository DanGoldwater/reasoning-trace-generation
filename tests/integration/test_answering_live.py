"""One real question, put to the real model, asserted from every angle.

Every test here reads the *same* generation, produced once by the ``live_record``
fixture. Splitting the assertions across named tests keeps a failure legible
without paying for the model again.
"""

import asyncio
from pathlib import Path

import pytest

from src.dataset.models import Question, TraceRecord
from src.dataset.runs import append_record, read_records
from src.generation.answering import answer_question, build_answering_agent
from src.generation.prompts import ANSWER_INSTRUCTIONS
from src.llm.config import INTEGRATION_TEST_TIMEOUT_SECONDS, OllamaSettings
from src.llm.reasoning import ReasoningOverranError
from tests.integration.conftest import DUMMY_COMPLETION_ID, DUMMY_QUESTION_ID

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(INTEGRATION_TEST_TIMEOUT_SECONDS),
]


def test_the_live_record_is_complete_and_correctly_identified(
    live_record: TraceRecord,
    dummy_question: Question,
) -> None:
    """Everything the written row is made of, checked on one real generation."""
    assert live_record.question_id == DUMMY_QUESTION_ID
    assert live_record.completion_id == DUMMY_COMPLETION_ID
    assert live_record.hf_sample == dummy_question.sample
    assert live_record.completion.answer in live_record.hf_sample.options
    assert live_record.completion.reasoning.strip()


def test_the_live_trace_is_prose_rather_than_tagged_markup(
    live_record: TraceRecord,
) -> None:
    """The trace is stored as the model's thinking, with its tags stripped."""
    reasoning = live_record.completion.reasoning

    assert "<think>" not in reasoning
    assert "</think>" not in reasoning
    assert len(reasoning) > len(live_record.completion.answer)


def test_the_live_prompt_records_what_the_model_was_actually_given(
    live_record: TraceRecord,
    dummy_question: Question,
) -> None:
    full_prompt = live_record.prompting.full_prompt

    assert ANSWER_INSTRUCTIONS in full_prompt
    assert dummy_question.sample.question in full_prompt
    assert "A: No" in full_prompt
    assert "B: Yes" in full_prompt


def test_the_live_model_reaches_the_answer_the_prompt_spells_out(
    live_record: TraceRecord,
) -> None:
    """The dummy question states its own answer, so a wrong key means trouble."""
    assert live_record.completion.answer == live_record.hf_sample.answer


def test_a_live_record_survives_the_round_trip_to_a_run_file(
    live_record: TraceRecord,
    tmp_path: Path,
) -> None:
    """The real record, not a hand-built one, is what has to serialise."""
    path = tmp_path / "runs" / "live.jsonl"

    append_record(path, live_record)

    assert read_records(path) == [live_record]


def test_the_generation_budget_really_reaches_the_server(
    settings: OllamaSettings,
    dummy_question: Question,
) -> None:
    """A budget too small to think in must cut the model off, not be ignored.

    Ollama reads ``max_tokens`` and ignores ``max_completion_tokens``, which is
    what pydantic-ai sends by default. When that regresses, the budget silently
    stops applying and a looping model runs until the request times out; here
    that would turn a one-second test into a timeout.
    """
    agent = build_answering_agent(settings, dummy_question.sample, max_tokens=16)

    with pytest.raises(ReasoningOverranError):
        asyncio.run(answer_question(agent, dummy_question))
