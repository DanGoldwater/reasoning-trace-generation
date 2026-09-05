"""Fixtures binding the integration suite to a real, running Ollama server.

These tests are deliberately not skipped when Ollama is missing: a broken local
setup should fail loudly rather than pass silently.

The live model is slow, so it is called as few times as possible: the fixtures
below each make **one** request per session, and the tests assert against the
result they share. Adding an assertion is free; adding a fixture is not.
"""

import asyncio

import pytest

from src.dataset.models import HFSample, Question, TraceRecord
from src.generation.answering import answer_question, build_answering_agent
from src.llm.agents import build_agent
from src.llm.config import OllamaSettings
from src.llm.health import require_ready

# A question in exactly the shape the real dataset rows arrive in — options as a
# JSON string, an A/B answer key, a metadata block — but answerable from the
# prompt alone. Real rows turn on recall the model does not have, and it spends
# its whole token budget failing to remember; that is a generation problem to
# solve in the pipeline, not something every test run should pay for.
DUMMY_ROW = {
    "question": (
        "In a screen, the cancer cell line CTRL-001 showed 99% growth inhibition "
        "at 1 nM of drug X, far below the 1 uM threshold for calling a line "
        "sensitive. Would CTRL-001 be sensitive to treatment with drug X?"
    ),
    "options": '{"A": "No", "B": "Yes"}',
    "answer": "B",
    "metadata": {"cell_line": "CTRL-001", "drug": "drug X"},
}
DUMMY_QUESTION_ID = 42
DUMMY_COMPLETION_ID = 3


@pytest.fixture(scope="session")
def settings() -> OllamaSettings:
    """Real settings from the centrally-defined integration profile."""
    return OllamaSettings.integration_from_env()


@pytest.fixture(scope="session", autouse=True)
def _ollama_is_ready(settings: OllamaSettings) -> None:
    """Fail the whole integration suite up front if Ollama is not usable."""
    require_ready(settings)


@pytest.fixture(scope="session")
def dummy_question() -> Question:
    """The dataset-shaped question the live suite puts to the model."""
    return Question(
        question_id=DUMMY_QUESTION_ID,
        sample=HFSample.model_validate(DUMMY_ROW),
    )


@pytest.fixture(scope="session")
def live_record(settings: OllamaSettings, dummy_question: Question) -> TraceRecord:
    """One real generation, shared by every test that asserts about its result.

    Synchronous on purpose: ``asyncio.run`` keeps this a plain session fixture,
    rather than an async one whose event loop has to be widened to match.
    """
    agent = build_answering_agent(settings, dummy_question.sample)
    return asyncio.run(
        answer_question(agent, dummy_question, completion_id=DUMMY_COMPLETION_ID)
    )


@pytest.fixture(scope="session")
def live_text(settings: OllamaSettings) -> str:
    """One real unstructured generation, proving the plain-text path works."""
    agent = build_agent(
        settings, instructions="Answer with a single word and nothing else."
    )
    return asyncio.run(agent.run("What is the capital of France?")).output
