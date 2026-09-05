"""One live smoke test of the question-to-record contract, using production code."""

import hashlib
import json
from pathlib import Path

import pytest

from src.dataset.models import HFSample, Question, TraceRecord
from src.dataset.runs import read_records
from src.experiments import run_experiment
from src.settings import INTEGRATION_TEST_TIMEOUT_SECONDS, OllamaSettings, RunSettings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(INTEGRATION_TEST_TIMEOUT_SECONDS),
]


async def test_local_question_runs_through_gates_to_disk(tmp_path: Path) -> None:
    # Keep the dataset's JSON-encoded options, but supply the fact needed to
    # answer. This tests our output contract, not biomedical knowledge or recall.
    question = Question(
        question_id=0,
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
    source = tmp_path / "questions.json"
    source.write_text(json.dumps([question.sample.model_dump()]), encoding="utf-8")
    run_settings = RunSettings(
        llm_judge="off",
        input_path=source,
        runs_dir=tmp_path / "runs",
        llm=settings,
        question_limit=1,
        completions_per_question=1,
    )
    directory = await run_experiment(run_settings)
    records = read_records(directory / "passed.jsonl")
    assert len(records) == 1, (directory / "failed.jsonl").read_text()
    record = records[0]
    assert directory.parent == tmp_path / "runs"
    assert len(directory.name.split("-")) == 3
    assert {path.name for path in directory.iterdir()} == {
        "passed.jsonl",
        "failed.jsonl",
        "run.json",
    }
    assert (directory / "failed.jsonl").read_text() == ""
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["name"] == directory.name
    assert metadata["settings"] == run_settings.model_dump(mode="json")
    assert metadata["gates"] == [
        "non_empty_answer",
        "non_empty_reasoning",
        "correct_answer",
    ]
    assert metadata["status"] == "completed"
    assert metadata["passed"] == 1
    assert metadata["failed"] == 0
    assert metadata["finished_at"] >= metadata["started_at"]
    assert metadata["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "api_key" not in metadata["settings"]["llm"]

    assert isinstance(record, TraceRecord)
    assert record.question_id == question.question_id
    assert record.completion_id == 0
    assert record.hf_sample == question.sample
    assert record.completion.answer in question.sample.options
    assert record.completion.answer == "B"
    assert record.completion.reasoning.strip()
    assert "<think>" not in record.completion.reasoning
    assert "</think>" not in record.completion.reasoning
    assert question.sample.question in record.prompting.full_prompt
    assert TraceRecord.model_validate_json(record.model_dump_json()) == record
