"""Capture a generation attempt without losing rejected or partial outputs."""

import asyncio

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from src.dataset.models import Prompting, Question
from src.generation.answering import Choice
from src.generation.prompts import render_question
from src.llm.reasoning import (
    MissingReasoningError,
    ReasoningOverranError,
    prompt_sent,
    run_reasoned,
)
from src.quality import CandidateCompletion, CandidateRecord, GateFailure


class GenerationAttempt(BaseModel):
    record: CandidateRecord
    failure: GateFailure | None = None
    raw_response: str | None = None


async def generate_attempt(
    agent: Agent[None, Choice],
    question: Question,
    *,
    completion_id: int,
    timeout_seconds: float,
    instructions: str,
) -> GenerationAttempt:
    """Keep parseable output, or recover the final response after a failed request."""
    prompt = render_question(question.sample)
    attempt = GenerationAttempt(
        record=CandidateRecord(
            question_id=question.question_id,
            completion_id=completion_id,
            hf_sample=question.sample,
            completion=CandidateCompletion(),
            prompting=Prompting(
                full_prompt="\n\n".join(part for part in (instructions, prompt) if part)
            ),
        )
    )
    messages: list[ModelMessage] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            result = await run_reasoned(
                agent, prompt, allow_empty=True, audit_messages=messages
            )
        attempt.record.completion = CandidateCompletion(
            reasoning=result.reasoning,
            answer=result.output.answer,
        )
        attempt.record.prompting = Prompting(full_prompt=result.prompt)
    except (
        UnexpectedModelBehavior,
        ReasoningOverranError,
        MissingReasoningError,
        TimeoutError,
        httpx.HTTPError,
        ModelAPIError,
    ) as error:
        if isinstance(error, ModelHTTPError) and error.status_code in (
            401,
            403,
            404,
        ):
            raise
        attempt.failure = GateFailure(gate="generation", reason=type(error).__name__)
        final = next(
            (m for m in reversed(messages) if isinstance(m, ModelResponse)), None
        )
        if final is not None:
            attempt.record.completion.reasoning = "\n".join(
                part.content for part in final.parts if isinstance(part, ThinkingPart)
            ).strip()
            attempt.raw_response = "\n".join(
                part.content if isinstance(part, TextPart) else part.args_as_json_str()
                for part in final.parts
                if isinstance(part, (TextPart, ToolCallPart))
            )
            try:
                partial = CandidateCompletion.model_validate_json(attempt.raw_response)
            except ValueError:
                pass
            else:
                attempt.record.completion.answer = partial.answer
    if messages:
        attempt.record.prompting = Prompting(full_prompt=prompt_sent(messages))
    return attempt
