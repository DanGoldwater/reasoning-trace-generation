"""The experiment runner's persisted output contract."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.dataset.runs import read_records
from src.experiments import run_experiment
from src.settings import OllamaSettings, RunSettings


async def test_run_persists_a_passing_record_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart("The evidence says yes."), TextPart('{"answer":"B"}')]
        )

    settings = RunSettings(
        input_path=source, runs_dir=tmp_path / "runs", llm=OllamaSettings()
    )
    directory = await run_experiment(settings, model=FunctionModel(respond))
    records = read_records(directory / "passed.jsonl")
    assert len(records) == 1
    assert records[0].completion.answer == "B"
    assert records[0].completion.reasoning == "The evidence says yes."
    assert (directory / "failed.jsonl").read_text() == ""
    assert (directory / "run.json").exists()


async def test_failures_preserve_outputs_and_all_gate_names_before_next_question(
    tmp_path: Path,
) -> None:
    import json

    from src.quality import FailedRecord

    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question": "First?",
                    "options": {"A": "No", "B": "Yes"},
                    "answer": "B",
                },
                {
                    "question": "Second?",
                    "options": {"A": "No", "B": "Yes"},
                    "answer": "B",
                },
            ]
        )
    )
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart('{"answer":"A"}')])
        directories = list((tmp_path / "runs").iterdir())
        rejected = FailedRecord.model_validate_json(
            (directories[0] / "failed.jsonl").read_text()
        )
        assert rejected.record.question_id == 0
        assert rejected.record.completion.answer == "A"
        assert rejected.record.completion.reasoning == ""
        assert [failure.gate for failure in rejected.failures] == [
            "non_empty_reasoning",
            "correct_answer",
        ]
        assert all(failure.reason for failure in rejected.failures)
        return ModelResponse(
            parts=[
                ThinkingPart("Yes follows from the question."),
                TextPart('{"answer":"B"}'),
            ]
        )

    directory = await run_experiment(
        RunSettings(input_path=source, runs_dir=tmp_path / "runs"),
        model=FunctionModel(respond),
    )
    assert calls == 2
    assert read_records(directory / "passed.jsonl")[0].question_id == 1
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["passed"] == 1
    assert metadata["failed"] == 1
    assert metadata["status"] == "completed"


async def test_provider_transport_failure_is_recorded_and_run_continues(
    tmp_path: Path,
) -> None:
    import json

    from pydantic_ai.exceptions import ModelAPIError

    from src.quality import FailedRecord

    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelAPIError(model_name="stub", message="Connection lost")
        return ModelResponse(parts=[ThinkingPart("Yes."), TextPart('{"answer":"B"}')])

    directory = await run_experiment(
        RunSettings(
            input_path=source,
            runs_dir=tmp_path / "runs",
            completions_per_question=2,
        ),
        model=FunctionModel(respond),
    )
    rejected = FailedRecord.model_validate_json(
        (directory / "failed.jsonl").read_text()
    )
    assert rejected.record.completion.answer is None
    assert rejected.failures[0].gate == "generation"
    assert rejected.failures[0].reason == "ModelAPIError"
    assert read_records(directory / "passed.jsonl")[0].completion_id == 1
    assert json.loads((directory / "run.json").read_text())["status"] == "completed"


async def test_truncated_output_retains_reasoning_and_answer(tmp_path: Path) -> None:
    from src.quality import FailedRecord

    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart("I was still reasoning"), TextPart('{"answer":"B"}')],
            finish_reason="length",
        )

    directory = await run_experiment(
        RunSettings(input_path=source, runs_dir=tmp_path / "runs"),
        model=FunctionModel(respond),
    )
    rejected = FailedRecord.model_validate_json(
        (directory / "failed.jsonl").read_text()
    )
    assert read_records(directory / "passed.jsonl") == []
    assert rejected.record.completion.reasoning == "I was still reasoning"
    assert rejected.record.completion.answer == "B"
    assert rejected.raw_response == '{"answer":"B"}'
    assert [(f.gate, f.reason) for f in rejected.failures] == [
        ("generation", "ReasoningOverranError")
    ]


async def test_malformed_output_is_saved_after_bounded_validation_retries(
    tmp_path: Path,
) -> None:
    from src.quality import FailedRecord

    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[ThinkingPart("I think yes"), TextPart("My answer is probably yes")]
        )

    directory = await run_experiment(
        RunSettings(input_path=source, runs_dir=tmp_path / "runs"),
        model=FunctionModel(respond),
    )
    rejected = FailedRecord.model_validate_json(
        (directory / "failed.jsonl").read_text()
    )
    assert calls == 2
    assert rejected.record.completion.reasoning == "I think yes"
    assert rejected.record.completion.answer is None
    assert rejected.raw_response == "My answer is probably yes"
    assert [failure.gate for failure in rejected.failures] == [
        "generation",
        "correct_answer",
    ]


async def test_interruption_keeps_previous_completion_and_marks_run(
    tmp_path: Path,
) -> None:
    import asyncio
    import json

    import pytest

    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError
        return ModelResponse(parts=[ThinkingPart("Yes"), TextPart('{"answer":"B"}')])

    with pytest.raises(asyncio.CancelledError):
        await run_experiment(
            RunSettings(
                input_path=source,
                runs_dir=tmp_path / "runs",
                completions_per_question=2,
            ),
            model=FunctionModel(respond),
        )
    (directory,) = (tmp_path / "runs").iterdir()
    assert len(read_records(directory / "passed.jsonl")) == 1
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["passed"] == 1
    assert metadata["finished_at"] is not None


async def test_missing_data_without_token_explains_env_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pytest

    monkeypatch.setenv("HF_TOKEN", "")
    with pytest.raises(RuntimeError, match=r"HF_TOKEN in your .env"):
        await run_experiment(RunSettings(input_path=tmp_path / "missing.json"))


def test_run_metadata_never_contains_anthropic_credentials(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from src.experiments import RunMetadata
    from src.settings import AnthropicSettings

    settings = RunSettings(
        input_path=tmp_path / "input.json",
        llm=AnthropicSettings(api_key="test-credential"),
    )
    metadata = RunMetadata(
        name="example",
        settings=settings,
        gates=[],
        input_sha256="abc",
        started_at=datetime.now(UTC),
    )
    assert "test-credential" not in metadata.model_dump_json()
    assert "api_key" not in metadata.model_dump_json()
    assert "test-credential" not in repr(metadata)


async def test_custom_gate_and_question_limit_apply_to_each_completion(
    tmp_path: Path,
) -> None:
    import json

    from src.quality import CandidateRecord, FailedRecord, QualityGate

    class MinimumTraceLength(QualityGate):
        name = "minimum_trace_length"

        def check(self, record: CandidateRecord) -> str | None:
            return (
                "Trace is too short." if len(record.completion.reasoning) < 20 else None
            )

    source = tmp_path / "questions.json"
    source.write_text(
        json.dumps(
            [
                {
                    "question": "First?",
                    "options": {"A": "No", "B": "Yes"},
                    "answer": "B",
                },
                {
                    "question": "Second?",
                    "options": {"A": "No", "B": "Yes"},
                    "answer": "B",
                },
            ]
        )
    )

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ThinkingPart("Yes."), TextPart('{"answer":"B"}')])

    directory = await run_experiment(
        RunSettings(
            input_path=source,
            runs_dir=tmp_path / "runs",
            question_limit=1,
            completions_per_question=2,
        ),
        gates=[MinimumTraceLength()],
        model=FunctionModel(respond),
    )
    rejected = [
        FailedRecord.model_validate_json(line)
        for line in (directory / "failed.jsonl").read_text().splitlines()
    ]
    assert [(r.record.question_id, r.record.completion_id) for r in rejected] == [
        (0, 0),
        (0, 1),
    ]
    assert all(r.failures[0].gate == "minimum_trace_length" for r in rejected)
    assert read_records(directory / "passed.jsonl") == []


async def test_missing_dataset_is_fetched_and_then_processed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets import Dataset

    monkeypatch.setenv("HF_TOKEN", "test-token")

    def fetch(repo: str, *, token: str, split: str) -> Dataset:
        assert repo == "owkin/technical_test"
        assert token == "test-token"
        assert split == "train"
        return Dataset.from_list(
            [{"question": "Which?", "options": '{"A":"No","B":"Yes"}', "answer": "B"}]
        )

    monkeypatch.setattr("src.data_fetching.load_dataset", fetch)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ThinkingPart("Yes."), TextPart('{"answer":"B"}')])

    source = tmp_path / "missing.json"
    directory = await run_experiment(
        RunSettings(input_path=source, runs_dir=tmp_path / "runs"),
        model=FunctionModel(respond),
    )
    assert source.exists()
    assert read_records(directory / "passed.jsonl")[0].hf_sample.answer == "B"


async def test_timeout_is_recorded_without_waiting_for_remaining_generation(
    tmp_path: Path,
) -> None:
    import asyncio

    from src.quality import FailedRecord

    source = tmp_path / "questions.json"
    source.write_text(
        '[{"question":"Which?","options":{"A":"No","B":"Yes"},"answer":"B"}]'
    )

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        raise AssertionError("The deadline should cancel this request.")

    directory = await run_experiment(
        RunSettings(
            input_path=source,
            runs_dir=tmp_path / "runs",
            llm=OllamaSettings(timeout_seconds=0.05),
        ),
        model=FunctionModel(respond),
    )
    rejected = FailedRecord.model_validate_json(
        (directory / "failed.jsonl").read_text()
    )
    assert rejected.failures[0].reason == "TimeoutError"
    assert rejected.record.completion.answer is None
    assert (directory / "passed.jsonl").read_text() == ""
