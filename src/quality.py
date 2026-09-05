"""Composable quality gates over generated candidates."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

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


class FailureType(StrEnum):
    NO_ANSWER = "no_answer"
    WRONG_ANSWER = "wrong_answer"
    MISSING_REASONING = "missing_reasoning"
    UNSUPPORTED_REASONING = "unsupported_reasoning"
    GENERATION_ERROR = "generation_error"
    JUDGE_ERROR = "judge_error"
    CUSTOM = "custom"


class JudgeVerdict(BaseModel):
    has_significant_hallucination: bool = Field(strict=True)
    explanation: str = Field(min_length=1)

    @field_validator("explanation")
    @classmethod
    def nonblank_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Judge explanation must not be blank.")
        return value.strip()


class GateFailure(BaseModel):
    gate: str
    reason: str
    failure_type: FailureType = FailureType.CUSTOM
    judge_verdict: JudgeVerdict | None = None


class QualityGate(ABC):
    """A named asynchronous check; return a structured failure or None."""

    name: str
    failure_type: FailureType = FailureType.CUSTOM
    requires_complete_generation: bool = False

    def reject(self, reason: str) -> GateFailure:
        return GateFailure(
            gate=self.name, reason=reason, failure_type=self.failure_type
        )

    @abstractmethod
    async def check(self, record: CandidateRecord) -> GateFailure | None: ...


class NonEmptyAnswer(QualityGate):
    name = "non_empty_answer"
    failure_type = FailureType.NO_ANSWER

    async def check(self, record: CandidateRecord) -> GateFailure | None:
        answer = record.completion.answer
        return (
            None
            if answer and answer.strip()
            else self.reject("No answer was produced.")
        )


class NonEmptyReasoning(QualityGate):
    name = "non_empty_reasoning"
    failure_type = FailureType.MISSING_REASONING

    async def check(self, record: CandidateRecord) -> GateFailure | None:
        return (
            None
            if record.completion.reasoning.strip()
            else self.reject("Reasoning is empty.")
        )


class CorrectAnswer(QualityGate):
    name = "correct_answer"
    failure_type = FailureType.WRONG_ANSWER

    async def check(self, record: CandidateRecord) -> GateFailure | None:
        if not record.completion.answer or not record.completion.answer.strip():
            return None
        if record.completion.answer == record.hf_sample.answer:
            return None
        return self.reject(
            "Predicted answer does not match the ground-truth option key."
        )


async def evaluate_gates(
    record: CandidateRecord,
    gates: Sequence[QualityGate],
    *,
    generation_complete: bool = True,
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    for gate in gates:
        if gate.requires_complete_generation and not generation_complete:
            continue
        failure = await gate.check(record)
        if failure is not None:
            failures.append(failure)
    return failures


class FailedRecord(BaseModel):
    record: CandidateRecord
    failures: list[GateFailure]
    raw_response: str | None = None
