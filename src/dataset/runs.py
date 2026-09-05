"""Writing generated traces to, and reading them back from, a run file."""

from pathlib import Path

from pydantic import BaseModel

from src.dataset.models import TraceRecord


def append_record(path: Path, record: BaseModel) -> None:
    """Append one record to the JSON Lines file at ``path``.

    Appending a line at a time means a run that dies halfway leaves every trace
    it had already generated on disk, intact and readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as run_file:
        run_file.write(record.model_dump_json() + "\n")


def read_records(path: Path) -> list[TraceRecord]:
    """Read every record back out of the JSON Lines file at ``path``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [TraceRecord.model_validate_json(line) for line in lines if line.strip()]
