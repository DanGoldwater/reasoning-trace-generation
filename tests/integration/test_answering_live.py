"""One live smoke test of the question-to-record contract, using production code."""

import asyncio

import httpx
import pytest

from src.dataset.models import HFSample, Question, TraceRecord
from src.generation.answering import answer_question, build_answering_agent
from src.llm.config import INTEGRATION_TEST_TIMEOUT_SECONDS, OllamaSettings
from src.llm.health import require_ready

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(INTEGRATION_TEST_TIMEOUT_SECONDS),
]


async def test_local_question_returns_a_structured_answer_and_reasoning() -> None:
    # Keep the dataset's JSON-encoded options, but supply the fact needed to
    # answer. This tests our output contract, not biomedical knowledge or recall.
    question = Question(
        question_id=42,
        sample=HFSample.model_validate(
            {
                "question": (
                    "The screen classified cancer cell line CTRL-001 as sensitive "
                    "to drug X. According to this screen, is CTRL-001 sensitive "
                    "to drug X?"
                ),
                "options": '{"A": "No", "B": "Yes"}',
                "answer": "B",
            }
        ),
    )
    settings = OllamaSettings.integration_from_env()
    # Readiness is cheap and should fail promptly with setup instructions.
    with httpx.Client(timeout=5.0) as client:
        require_ready(settings, client=client)

    agent = build_answering_agent(settings, question.sample)
    # Bound the entire run, including any validation retries, not just each
    # HTTP request. The pytest deadline also covers setup and synchronous code.
    async with asyncio.timeout(settings.timeout_seconds):
        record = await answer_question(agent, question, completion_id=3)

    assert isinstance(record, TraceRecord)
    assert record.question_id == question.question_id
    assert record.completion_id == 3
    assert record.hf_sample == question.sample
    assert record.completion.answer in question.sample.options
    assert record.completion.answer == "B"
    assert record.completion.reasoning.strip()
    assert "<think>" not in record.completion.reasoning
    assert "</think>" not in record.completion.reasoning
    assert question.sample.question in record.prompting.full_prompt
    assert TraceRecord.model_validate_json(record.model_dump_json()) == record
