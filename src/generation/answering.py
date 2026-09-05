"""Putting one question to a model and recording what came back."""

from collections.abc import Iterable

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from src.dataset.models import Completion, HFSample, Prompting, Question, TraceRecord
from src.generation.prompts import ANSWER_INSTRUCTIONS, render_question
from src.llm.agents import build_agent
from src.llm.reasoning import run_reasoned
from src.settings import LLMSettings


class Choice(BaseModel):
    """The shape a model must fill in: the key of the option it chose."""

    answer: str


def choice_type(option_keys: Iterable[str]) -> type[Choice]:
    """Build the answer schema for one question's option keys.

    The keys go into the schema itself rather than into the prompt alone, so
    schema-constrained decoding makes an off-menu answer unrepresentable instead
    of merely discouraged.
    """
    keys = tuple(sorted(option_keys))

    class AllowedChoice(Choice):
        answer: str = Field(json_schema_extra={"enum": list(keys)})

        @field_validator("answer")
        @classmethod
        def offered_option(cls, value: str) -> str:
            if value not in keys:
                raise ValueError("answer must be an offered option")
            return value

    return AllowedChoice


def build_answering_agent(
    settings: LLMSettings,
    sample: HFSample,
    *,
    max_tokens: int | None = None,
) -> Agent[None, Choice]:
    """Build the agent that answers ``sample``, constrained to its option keys."""
    return build_agent(
        settings,
        output_type=choice_type(sample.options),
        instructions=ANSWER_INSTRUCTIONS,
        max_tokens=max_tokens,
    )


async def answer_question(
    agent: Agent[None, Choice],
    question: Question,
    *,
    completion_id: int = 0,
) -> TraceRecord:
    """Ask ``agent`` one question and assemble the record for that completion.

    ``agent`` is passed in rather than built here because a run asks the same
    question repeatedly for different ``completion_id`` values.
    """
    reasoned = await run_reasoned(agent, render_question(question.sample))
    return TraceRecord(
        question_id=question.question_id,
        completion_id=completion_id,
        hf_sample=question.sample,
        completion=Completion(
            reasoning=reasoned.reasoning,
            answer=reasoned.output.answer,
        ),
        prompting=Prompting(full_prompt=reasoned.prompt),
    )
