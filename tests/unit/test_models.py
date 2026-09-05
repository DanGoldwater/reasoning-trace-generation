"""Unit tests for the record schema the pipeline writes out.

The input rows and the output record are different shapes: these tests pin the
translation between them, and the invariants a row must satisfy to be worth
keeping.
"""

import pytest
from pydantic import ValidationError

from src.dataset.models import Completion, HFSample, Prompting, TraceRecord

RAW_ROW = {
    "question": (
        "Would the Bladder/Urinary Tract cancer cell line HT-1197 be sensitive "
        "to treatment with Panobinostat?"
    ),
    "options": '{"A": "No", "B": "Yes"}',
    "answer": "B",
    "metadata": {"cell_line": "HT-1197", "drug": "Panobinostat"},
}


def test_options_arrive_as_a_json_string_and_become_a_mapping() -> None:
    """The HF rows encode options as JSON text; the record wants real keys."""
    sample = HFSample.model_validate(RAW_ROW)

    assert sample.options == {"A": "No", "B": "Yes"}


def test_a_gold_answer_that_is_not_one_of_the_options_is_rejected() -> None:
    """A row we cannot score against is worse than no row at all."""
    with pytest.raises(ValidationError, match="C"):
        HFSample.model_validate(RAW_ROW | {"answer": "C"})


def test_a_record_serialises_to_the_shape_the_spec_asks_for() -> None:
    """The written JSON is the deliverable, so its exact shape is pinned here."""
    record = TraceRecord(
        question_id=0,
        completion_id=0,
        hf_sample=HFSample.model_validate(
            {
                "question": "At 95.0 nM concentration, which of these two drugs...?",
                "options": '{"A": "Hydrocortisone", "B": "CCT 018159"}',
                "answer": "B",
                "metadata": {"study": "CTRPv2_2015"},
            }
        ),
        completion=Completion(
            reasoning="First, compare mechanism and expected hallmark shifts...",
            answer="B",
        ),
        prompting=Prompting(full_prompt="[exact prompt sent to the local model]"),
    )

    assert record.model_dump() == {
        "question_id": 0,
        "completion_id": 0,
        "hf_sample": {
            "question": "At 95.0 nM concentration, which of these two drugs...?",
            "answer": "B",
            "options": {"A": "Hydrocortisone", "B": "CCT 018159"},
        },
        "completion": {
            "reasoning": "First, compare mechanism and expected hallmark shifts...",
            "answer": "B",
        },
        "prompting": {"full_prompt": "[exact prompt sent to the local model]"},
    }


def test_a_completion_with_a_blank_reasoning_trace_is_rejected() -> None:
    """An empty trace is the whole point of the dataset missing, not a detail."""
    with pytest.raises(ValidationError, match="reasoning"):
        Completion(reasoning="   \n ", answer="B")


def test_a_completion_answering_outside_the_offered_options_is_rejected() -> None:
    """'C' on a two-option question means the row cannot be scored."""
    with pytest.raises(ValidationError, match="C"):
        TraceRecord(
            question_id=0,
            completion_id=0,
            hf_sample=HFSample.model_validate(RAW_ROW),
            completion=Completion(reasoning="Panobinostat is an HDAC...", answer="C"),
            prompting=Prompting(full_prompt="..."),
        )
