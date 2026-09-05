"""Unit tests for turning one question into one recorded attempt.

Stubbed at pydantic-ai's ``FunctionModel`` seam, so these run without a server.
"""

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from src.dataset.models import HFSample, Question
from src.generation.answering import build_answering_agent, choice_type
from src.generation.attempts import GenerationAttempt, generate_attempt
from src.generation.prompts import ANSWER_INSTRUCTIONS
from src.quality import CandidateCompletion, FailureType
from src.settings import OllamaSettings

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


async def attempt_answering(
    *parts: ModelResponsePart, completion_id: int = 0
) -> GenerationAttempt:
    """Put ``QUESTION`` to a model that replies with ``parts``."""
    agent = build_answering_agent(OllamaSettings(), SAMPLE)
    with agent.override(model=responding(*parts)):
        return await generate_attempt(
            agent,
            QUESTION,
            completion_id=completion_id,
            timeout_seconds=5.0,
            instructions=ANSWER_INSTRUCTIONS,
        )


async def test_the_record_carries_the_models_trace_and_choice_under_both_ids() -> None:
    attempt = await attempt_answering(
        ThinkingPart(TRACE), TextPart('{"answer":"B"}'), completion_id=3
    )

    assert attempt.failure is None
    assert attempt.record.question_id == 7
    assert attempt.record.completion_id == 3
    assert attempt.record.hf_sample == SAMPLE
    assert attempt.record.completion == CandidateCompletion(reasoning=TRACE, answer="B")


async def test_the_recorded_prompt_holds_the_question_and_every_option() -> None:
    """``full_prompt`` is the reproducibility handle, so it must be complete."""
    attempt = await attempt_answering(ThinkingPart(TRACE), TextPart('{"answer":"B"}'))

    full_prompt = attempt.record.prompting.full_prompt
    assert SAMPLE.question in full_prompt
    assert "A: No" in full_prompt
    assert "B: Yes" in full_prompt
    assert ANSWER_INSTRUCTIONS in full_prompt


async def test_a_key_the_question_never_offered_cannot_reach_a_record() -> None:
    """The option keys live in the output schema, so 'C' is not a valid answer.

    Against a real server the sampler refuses it outright; here the validator
    that backs that constraint is what rejects it. The attempt is kept as a
    generation failure, off-menu key and all, rather than becoming a record.
    """
    attempt = await attempt_answering(ThinkingPart(TRACE), TextPart('{"answer":"C"}'))

    assert attempt.failure is not None
    assert attempt.failure.failure_type is FailureType.GENERATION_ERROR
    assert attempt.raw_response == '{"answer":"C"}'


def test_the_output_schema_offers_exactly_the_questions_option_keys() -> None:
    """What the sampler is constrained to, read off the schema it is given."""
    schema = choice_type(SAMPLE.options).model_json_schema()

    assert schema["properties"]["answer"]["enum"] == ["A", "B"]
