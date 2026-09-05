"""Unit tests for writing generated traces to a run's JSON Lines file."""

from pathlib import Path

from src.dataset.models import Completion, HFSample, Prompting, TraceRecord
from src.dataset.runs import append_record, read_records

SAMPLE = HFSample.model_validate(
    {
        "question": "Would HT-1197 be sensitive to Panobinostat?",
        "options": '{"A": "No", "B": "Yes"}',
        "answer": "B",
    }
)


def record(
    completion_id: int,
    answer: str = "B",
    reasoning: str = "Panobinostat is an HDAC...",
) -> TraceRecord:
    """A record differing only in the fields a test cares about."""
    return TraceRecord(
        question_id=7,
        completion_id=completion_id,
        hf_sample=SAMPLE,
        completion=Completion(reasoning=reasoning, answer=answer),
        prompting=Prompting(full_prompt="Answer with A or B."),
    )


def test_appended_records_read_back_unchanged_and_in_order(tmp_path: Path) -> None:
    """A run is only useful if what comes back out is what went in."""
    path = tmp_path / "runs" / "run.jsonl"

    append_record(path, record(0))
    append_record(path, record(1, answer="A"))

    assert read_records(path) == [record(0), record(1, answer="A")]


def test_a_multi_line_reasoning_trace_still_occupies_one_line(tmp_path: Path) -> None:
    """Traces are full of newlines; one record per line is what JSON Lines means."""
    path = tmp_path / "run.jsonl"
    trace = "First, compare mechanism.\nThen, the IC50."

    append_record(path, record(0, reasoning=trace))

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
    assert read_records(path)[0].completion.reasoning == trace
