"""The record schema this pipeline reads in and writes out."""

import json
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


class HFSample(BaseModel):
    """One question from the source dataset, as it is stored in the record."""

    question: str
    answer: str
    options: dict[str, str]

    @field_validator("options", mode="before")
    @classmethod
    def _parse_json_options(cls, value: Any) -> Any:
        """Accept the JSON-encoded options string the HF rows actually carry."""
        if isinstance(value, str):
            return json.loads(value)
        return value

    @model_validator(mode="after")
    def _answer_is_one_of_the_options(self) -> "HFSample":
        """A gold answer outside the options cannot be scored against."""
        if self.answer not in self.options:
            keys = ", ".join(sorted(self.options)) or "none"
            message = f"answer {self.answer!r} is not one of the options ({keys})."
            raise ValueError(message)
        return self


class Completion(BaseModel):
    """What the model produced: its reasoning trace and the choice it made."""

    reasoning: str
    answer: str

    @field_validator("reasoning")
    @classmethod
    def _reasoning_is_not_blank(cls, value: str) -> str:
        """A blank trace is the dataset's entire value missing from the row."""
        if not value.strip():
            message = "reasoning must not be blank."
            raise ValueError(message)
        return value


class Prompting(BaseModel):
    """Exactly what was put on the wire, kept so a run can be reproduced."""

    full_prompt: str


class TraceRecord(BaseModel):
    """One generated reasoning trace, as written to a run's JSON Lines file."""

    question_id: int
    completion_id: int
    hf_sample: HFSample
    completion: Completion
    prompting: Prompting

    @model_validator(mode="after")
    def _completion_chose_an_offered_option(self) -> "TraceRecord":
        """The model's key has to exist in this question's options to mean anything."""
        if self.completion.answer not in self.hf_sample.options:
            keys = ", ".join(sorted(self.hf_sample.options)) or "none"
            message = (
                f"completion answered {self.completion.answer!r}, which question "
                f"{self.question_id} does not offer ({keys})."
            )
            raise ValueError(message)
        return self


class Question(BaseModel):
    """A source question with the identifier its generated records carry."""

    question_id: int
    sample: HFSample
