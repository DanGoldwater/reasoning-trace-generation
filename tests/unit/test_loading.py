"""Unit tests for reading the source Q&A file off disk."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from src.dataset.loading import InvalidQuestionError, load_questions

ROWS = [
    {
        "question": "Would HT-1197 be sensitive to Panobinostat?",
        "options": '{"A": "No", "B": "Yes"}',
        "answer": "B",
        "metadata": {"drug": "Panobinostat"},
    },
    {
        "question": "Would SW1710 be sensitive to Vorinostat?",
        "options": '{"A": "No", "B": "Yes"}',
        "answer": "A",
        "metadata": {"drug": "Vorinostat"},
    },
]


def write_rows(directory: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    """Write ``rows`` in the shape ``src/dataset/fetching.py`` saves them."""
    path = directory / "private_qa.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_questions_are_numbered_by_their_position_in_the_source_file(
    tmp_path: Path,
) -> None:
    """``question_id`` is the row's index, so a record points back at its source."""
    questions = load_questions(write_rows(tmp_path, ROWS))

    assert [question.question_id for question in questions] == [0, 1]
    assert questions[1].sample.question == "Would SW1710 be sensitive to Vorinostat?"
    assert questions[1].sample.options == {"A": "No", "B": "Yes"}


def test_an_unusable_row_fails_loudly_and_says_which_row(tmp_path: Path) -> None:
    """Finding the offending row by hand in a 600-row file is not a reasonable ask."""
    broken = [ROWS[0], ROWS[1] | {"answer": "C"}]

    with pytest.raises(InvalidQuestionError, match="row 1"):
        load_questions(write_rows(tmp_path, broken))
