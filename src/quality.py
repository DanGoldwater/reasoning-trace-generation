"""Composable quality gates over generated candidates."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.dataset.models import HFSample, Prompting


class CandidateCompletion(BaseModel):
    reasoning: str = ""
    answer: str | None = None


class CandidateRecord(BaseModel):
    """Preserves output even when it cannot become a valid training record."""

    question_id: int
    completion_id: int
    hf_sample: HFSample
    completion: CandidateCompletion
    prompting: Prompting


class GateFailure(BaseModel):
    gate: str
    reason: str


class QualityGate(ABC):
    """A named, side-effect-free check; return a reason to reject a candidate."""

    name: str

    @abstractmethod
    def check(self, record: CandidateRecord) -> str | None: ...


class NonEmptyReasoning(QualityGate):
    name = "non_empty_reasoning"

    def check(self, record: CandidateRecord) -> str | None:
        return None if record.completion.reasoning.strip() else "Reasoning is empty."


class CorrectAnswer(QualityGate):
    name = "correct_answer"

    def check(self, record: CandidateRecord) -> str | None:
        if record.completion.answer == record.hf_sample.answer:
            return None
        return "Predicted answer does not match the ground-truth option key."


def evaluate_gates(
    record: CandidateRecord, gates: tuple[QualityGate, ...]
) -> list[GateFailure]:
    return [
        GateFailure(gate=gate.name, reason=reason)
        for gate in gates
        if (reason := gate.check(record)) is not None
    ]


class FailedRecord(BaseModel):
    record: CandidateRecord
    failures: list[GateFailure]
    raw_response: str | None = None
