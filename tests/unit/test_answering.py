"""Unit tests for turning one question into one finished record.

Stubbed at pydantic-ai's ``FunctionModel`` seam, so these run without a server.
"""

import pytest
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.dataset.models import Completion, HFSample, Question
from src.generation.answering import (
    answer_question,
    build_answering_agent,
    choice_type,
)
from src.generation.prompts import ANSWER_INSTRUCTIONS
from src.llm.config import OllamaSettings

SAMPLE = HFSample.model_validate(
    {
        "question": "Would the Bladder/Urinary Tract cancer cell line HT-1197 be "
        "sensitive to treatment with Panobinostat?",
        "options": '{"A": "No", "B": "Yes"}',
        "answer": "B",
    }
)
QUESTION = Question(question_id=7, sample=SAMPLE)
TRACE = "Panobinostat is an HDAC inhibitor with sub-micromolar IC50s."


def responding(*parts: ModelResponsePart) -> FunctionModel:
    """A model that always replies with exactly ``parts``."""

    def stub(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(parts))

    return FunctionModel(stub)


async def test_the_record_carries_the_models_trace_and_choice_under_both_ids() -> None:
    agent = build_answering_agent(OllamaSettings(), SAMPLE)

    with agent.override(
        model=responding(ThinkingPart(TRACE), TextPart('{"answer":"B"}'))
    ):
        record = await answer_question(agent, QUESTION, completion_id=3)

    assert record.question_id == 7
    assert record.completion_id == 3
    assert record.hf_sample == SAMPLE
    assert record.completion == Completion(reasoning=TRACE, answer="B")


async def test_the_recorded_prompt_holds_the_question_and_every_option() -> None:
    """``full_prompt`` is the reproducibility handle, so it must be complete."""
    agent = build_answering_agent(OllamaSettings(), SAMPLE)

    with agent.override(
        model=responding(ThinkingPart(TRACE), TextPart('{"answer":"B"}'))
    ):
        record = await answer_question(agent, QUESTION)

    assert SAMPLE.question in record.prompting.full_prompt
    assert "A: No" in record.prompting.full_prompt
    assert "B: Yes" in record.prompting.full_prompt
    assert ANSWER_INSTRUCTIONS in record.prompting.full_prompt


async def test_a_key_the_question_never_offered_cannot_reach_a_record() -> None:
    """The option keys live in the output schema, so 'C' is not a valid answer.

    Against a real server the sampler refuses it outright; here the validator
    that backs that constraint is what rejects it.
    """
    agent = build_answering_agent(OllamaSettings(), SAMPLE)

    with agent.override(
        model=responding(ThinkingPart(TRACE), TextPart('{"answer":"C"}'))
    ):
        with pytest.raises(UnexpectedModelBehavior):
            await answer_question(agent, QUESTION)


def test_the_output_schema_offers_exactly_the_questions_option_keys() -> None:
    """What the sampler is constrained to, read off the schema it is given."""
    schema = choice_type(SAMPLE.options).model_json_schema()

    assert schema["properties"]["answer"]["enum"] == ["A", "B"]
