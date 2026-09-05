"""Sequential generation, quality evaluation, and durable experiment artifacts."""

import asyncio
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import petname
from pydantic import BaseModel
from pydantic_ai.models import Model

from src.data_fetching import load_private_dataset
from src.dataset.loading import load_questions
from src.dataset.models import TraceRecord
from src.dataset.runs import append_record
from src.generation.answering import choice_type
from src.generation.attempts import generate_attempt
from src.llm.agents import build_agent
from src.llm.health import require_ready
from src.quality import (
    CorrectAnswer,
    FailedRecord,
    NonEmptyReasoning,
    QualityGate,
    evaluate_gates,
)
from src.settings import OllamaSettings, RunSettings


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
    for _ in range(100):
        directory = root / petname.Generate(3, "-")
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        return directory
    raise RuntimeError("Could not allocate a unique petname for this run.")


async def run_experiment(
    settings: RunSettings,
    *,
    gates: Sequence[QualityGate] | None = None,
    model: Model | None = None,
) -> Path:
    """Write each evaluated completion before requesting the next one.

    An optional model override is the external inference boundary used by tests.
    """
    selected = (
        tuple(gates) if gates is not None else (NonEmptyReasoning(), CorrectAnswer())
    )
    names = [gate.name for gate in selected]
    if len(set(names)) != len(names):
        raise ValueError("Quality gate names must be unique.")
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
    if model is None and isinstance(settings.llm, OllamaSettings):
        with httpx.Client(timeout=5.0) as client:
            require_ready(settings.llm, client=client)
    directory = create_run_directory(settings.runs_dir)
    for filename in ("passed.jsonl", "failed.jsonl"):
        (directory / filename).touch()
    metadata = RunMetadata(
        name=directory.name,
        settings=settings,
        gates=names,
        input_sha256=hashlib.sha256(settings.input_path.read_bytes()).hexdigest(),
        started_at=datetime.now(UTC),
    )
    save_metadata(directory, metadata)
    try:
        for question in questions:
            agent = build_agent(
                settings.llm,
                output_type=choice_type(question.sample.options),
                instructions=settings.instructions,
                temperature=settings.temperature,
            )
            if model is not None:
                agent.model = model
            for completion_id in range(settings.completions_per_question):
                attempt = await generate_attempt(
                    agent,
                    question,
                    completion_id=completion_id,
                    timeout_seconds=settings.llm.timeout_seconds,
                    instructions=settings.instructions,
                )
                candidate = attempt.record
                failures = [attempt.failure] if attempt.failure is not None else []
                failures.extend(evaluate_gates(candidate, selected))
                if failures:
                    rejected = FailedRecord(
                        record=candidate,
                        failures=failures,
                        raw_response=attempt.raw_response,
                    )
                    with (directory / "failed.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(rejected.model_dump_json() + "\n")
                    metadata.failed += 1
                else:
                    record = TraceRecord.model_validate(candidate.model_dump())
                    append_record(directory / "passed.jsonl", record)
                    metadata.passed += 1
                save_metadata(directory, metadata)
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
