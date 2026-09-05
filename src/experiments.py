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
from pydantic_ai import Agent
from pydantic_ai.models import Model

from src.data_fetching import load_private_dataset
from src.dataset.loading import load_questions
from src.dataset.models import Question, TraceRecord
from src.dataset.runs import append_record
from src.generation.answering import Choice, choice_type
from src.generation.attempts import GenerationAttempt, generate_attempt
from src.judging import ReasoningHallucination
from src.llm.agents import build_agent
from src.llm.health import require_ready
from src.quality import (
    CorrectAnswer,
    FailedRecord,
    NonEmptyAnswer,
    NonEmptyReasoning,
    QualityGate,
    evaluate_gates,
)
from src.settings import (
    RUN_DIRECTORY_ATTEMPTS,
    RUN_NAME_WORDS,
    OllamaSettings,
    RunSettings,
)


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
    temporary = directory / "run.json.tmp"
    temporary.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "run.json")


def create_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(RUN_DIRECTORY_ATTEMPTS):
        directory = root / petname.Generate(RUN_NAME_WORDS, "-")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise RuntimeError("Could not allocate a unique petname for this run.")


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
    names = [gate.name for gate in selected]
    if len(set(names)) != len(names):
        raise ValueError("Quality gate names must be unique.")
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
    directory = create_run_directory(settings.runs_dir)
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
    selected: Sequence[QualityGate],
    directory: Path,
    metadata: RunMetadata,
) -> bool:
    """Write the attempt to the passed or failed file; True when it passed."""
    candidate = attempt.record
    failures = [attempt.failure] if attempt.failure is not None else []
    failures.extend(
        await evaluate_gates(
            candidate, tuple(selected), generation_complete=attempt.failure is None
        )
    )
    if failures:
        rejected = FailedRecord(
            record=candidate,
            failures=failures,
            raw_response=attempt.raw_response,
        )
        with (directory / "failed.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(rejected.model_dump_json() + "\n")
        metadata.failed += 1
    else:
        record = TraceRecord.model_validate(candidate.model_dump())
        append_record(directory / "passed.jsonl", record)
        metadata.passed += 1
    save_metadata(directory, metadata)
    return not failures


def build_question_agent(
    question: Question,
    settings: RunSettings,
    model: Model | None,
) -> Agent[None, Choice]:
    """An agent constrained to this question's option keys."""
    agent = build_agent(
        settings.llm,
        output_type=choice_type(question.sample.options),
        instructions=settings.instructions,
        temperature=settings.temperature,
    )
    if model is not None:
        agent.model = model
    return agent


async def process_question(
    question: Question,
    settings: RunSettings,
    gates: Sequence[QualityGate],
    model: Model | None,
    directory: Path,
    metadata: RunMetadata,
) -> tuple[int, int]:
    """Generate and persist every completion for one question, counting outcomes."""
    agent = build_question_agent(question, settings, model)
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
            print(
                f"[{metadata.name}] Question {index}/{len(questions)} "
                f"(id={question.question_id}): "
                f"generating {settings.completions_per_question}...",
                flush=True,
            )
            passed, failed = await process_question(
                question, settings, selected, model, directory, metadata
            )
            print(
                f"[{metadata.name}] Question {index}/{len(questions)} "
                f"(id={question.question_id}): "
                f"{passed} passed, {failed} failed in {monotonic() - started:.1f}s; "
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
