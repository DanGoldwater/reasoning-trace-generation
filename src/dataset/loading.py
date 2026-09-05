"""Reading the source Q&A file that ``fetching.py`` saved."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.dataset.models import HFSample, Question

DEFAULT_QUESTIONS_PATH = Path("data/private_qa.json")


class InvalidQuestionError(ValueError):
    """Raised when a source row cannot be read as a question."""


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[Question]:
    """Load every question from ``path``, numbered by its position in the file.

    Position is the identifier because it is the only stable handle the source
    rows offer: a record's ``question_id`` indexes straight back into this file.
    """
    rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [_question_from(row, index, path) for index, row in enumerate(rows)]


def _question_from(row: dict[str, Any], index: int, path: Path) -> Question:
    """Read one row, reporting where a rejected row lives rather than what it is."""
    try:
        return Question(question_id=index, sample=HFSample.model_validate(row))
    except ValidationError as error:
        message = f"{path}, row {index} is not a usable question: {error}"
        raise InvalidQuestionError(message) from error
