"""The answer schema a question constrains its model to, and the agent for it."""

from collections.abc import Iterable

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from src.dataset.models import HFSample
from src.generation.prompts import ANSWER_INSTRUCTIONS
from src.llm.agents import build_agent
from src.settings import DETERMINISTIC_TEMPERATURE, LLMSettings


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
    instructions: str = ANSWER_INSTRUCTIONS,
    temperature: float = DETERMINISTIC_TEMPERATURE,
) -> Agent[None, Choice]:
    """Build the agent that answers ``sample``, constrained to its option keys."""
    return build_agent(
        settings,
        output_type=choice_type(sample.options),
        instructions=instructions,
        temperature=temperature,
    )
