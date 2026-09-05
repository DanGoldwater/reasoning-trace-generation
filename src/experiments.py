"""Sequential generation, quality evaluation, and durable experiment artifacts."""

import asyncio
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Literal

import httpx
import petname
from pydantic import BaseModel
from pydantic_ai.models import Model

from src.data_fetching import load_private_dataset
from src.dataset.loading import load_questions
from src.dataset.models import Question, TraceRecord
from src.dataset.runs import append_record
from src.generation.answering import build_answering_agent
from src.generation.attempts import GenerationAttempt, generate_attempt
from src.judging import ReasoningHallucination
from src.llm.health import require_ready
from src.quality import (
    CorrectAnswer,
    FailedRecord,
    NonEmptyAnswer,
    NonEmptyReasoning,
    QualityGate,
    evaluate_gates,
)
from src.settings import RUN_NAME_WORDS, OllamaSettings, RunSettings


class RunMetadata(BaseModel):
    name: str
    settings: RunSettings
    gates: list[str]
    input_sha256: str
    started_at: datetime
    finished_at: datetime | None = None
    status: Literal["running", "completed", "interrupted", "failed"] = "running"
    passed: int = 0
    failed: int = 0


def save_metadata(directory: Path, metadata: RunMetadata) -> None:
    (directory / "run.json").write_text(
        metadata.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def select_gates(
    gates: Sequence[QualityGate] | None, settings: RunSettings
) -> tuple[QualityGate, ...]:
    selected = (
        tuple(gates)
        if gates is not None
        else (NonEmptyAnswer(), NonEmptyReasoning(), CorrectAnswer())
    )
    if settings.llm_judge == "on":
        selected += (ReasoningHallucination(settings.judge),)
    return selected


def prepare_questions(settings: RunSettings) -> list[Question]:
    if not settings.input_path.exists():
        try:
            load_private_dataset(output_path=settings.input_path)
        except Exception as error:
            raise RuntimeError(
                "Could not fetch the missing dataset. Set HF_TOKEN in your .env "
                "file to a token with access to owkin/technical_test, and retry. "
                "Also check your network connection."
            ) from error
    questions = load_questions(settings.input_path)
    if settings.question_limit is not None:
        questions = questions[: settings.question_limit]
    return questions


def initialize_run(
    settings: RunSettings, gates: Sequence[QualityGate]
) -> tuple[Path, RunMetadata]:
    directory = settings.runs_dir / petname.Generate(RUN_NAME_WORDS, "-")
    directory.mkdir(parents=True)
    for filename in ("passed.jsonl", "failed.jsonl"):
        (directory / filename).touch()
    metadata = RunMetadata(
        name=directory.name,
        settings=settings,
        gates=[gate.name for gate in gates],
        input_sha256=hashlib.sha256(settings.input_path.read_bytes()).hexdigest(),
        started_at=datetime.now(UTC),
    )
    save_metadata(directory, metadata)
    return directory, metadata


async def persist_attempt(
    attempt: GenerationAttempt,
    gates: Sequence[QualityGate],
    directory: Path,
    metadata: RunMetadata,
) -> bool:
    """Write the attempt to the passed or failed file; True when it passed."""
    candidate = attempt.record
    failures = [attempt.failure] if attempt.failure is not None else []
    failures.extend(
        await evaluate_gates(
            candidate, gates, generation_complete=attempt.failure is None
        )
    )
    if failures:
        append_record(
            directory / "failed.jsonl",
            FailedRecord(
                record=candidate,
                failures=failures,
                raw_response=attempt.raw_response,
            ),
        )
        metadata.failed += 1
    else:
        append_record(
            directory / "passed.jsonl",
            TraceRecord.model_validate(candidate.model_dump()),
        )
        metadata.passed += 1
    save_metadata(directory, metadata)
    return not failures


async def process_question(
    question: Question,
    settings: RunSettings,
    gates: Sequence[QualityGate],
    model: Model | None,
    directory: Path,
    metadata: RunMetadata,
) -> tuple[int, int]:
    """Generate and persist every completion for one question, counting outcomes."""
    agent = build_answering_agent(
        settings.llm,
        question.sample,
        instructions=settings.instructions,
        temperature=settings.temperature,
    )
    if model is not None:
        agent.model = model
    outcomes: list[bool] = []
    for completion_id in range(settings.completions_per_question):
        attempt = await generate_attempt(
            agent,
            question,
            completion_id=completion_id,
            timeout_seconds=settings.llm.timeout_seconds,
            instructions=settings.instructions,
            verbose=settings.verbose_ollama
            and isinstance(settings.llm, OllamaSettings),
        )
        outcomes.append(await persist_attempt(attempt, gates, directory, metadata))
    passed = sum(outcomes)
    return passed, len(outcomes) - passed


async def run_experiment(
    settings: RunSettings,
    *,
    gates: Sequence[QualityGate] | None = None,
    model: Model | None = None,
) -> Path:
    """Write each evaluated completion before requesting the next one.

    An optional model override is the external inference boundary used by tests.
    """
    selected = select_gates(gates, settings)
    questions = prepare_questions(settings)
    if model is None and isinstance(settings.llm, OllamaSettings):
        with httpx.Client(timeout=settings.llm.health_timeout_seconds) as client:
            require_ready(settings.llm, client=client)
    directory, metadata = initialize_run(settings, selected)
    print(f"Experiment {metadata.name}: {directory}", flush=True)
    try:
        for index, question in enumerate(questions, start=1):
            started = monotonic()
            progress = (
                f"[{metadata.name}] Question {index}/{len(questions)} "
                f"(id={question.question_id}):"
            )
            print(
                f"{progress} generating {settings.completions_per_question}...",
                flush=True,
            )
            passed, failed = await process_question(
                question, settings, selected, model, directory, metadata
            )
            print(
                f"{progress} {passed} passed, {failed} failed "
                f"in {monotonic() - started:.1f}s; "
                f"total: {metadata.passed} passed, {metadata.failed} failed",
                flush=True,
            )
        metadata.status = "completed"
    except (KeyboardInterrupt, asyncio.CancelledError):
        metadata.status = "interrupted"
        raise
    except Exception:
        metadata.status = "failed"
        raise
    finally:
        metadata.finished_at = datetime.now(UTC)
        save_metadata(directory, metadata)
    return directory
