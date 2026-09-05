"""Capture a generation attempt without losing rejected or partial outputs."""

import asyncio

import httpx
from pydantic import BaseModel
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from src.dataset.models import Prompting, Question
from src.generation.answering import Choice
from src.generation.prompts import render_question
from src.llm.reasoning import (
    MissingReasoningError,
    ReasoningOverranError,
    prompt_sent,
    reasoning_from,
    run_reasoned,
)
from src.quality import CandidateCompletion, CandidateRecord, FailureType, GateFailure

# Errors that say something about this one request, so the run records them and
# moves on to the next question.
RECOVERABLE_ERRORS = (
    UnexpectedModelBehavior,
    ReasoningOverranError,
    MissingReasoningError,
    TimeoutError,
    httpx.HTTPError,
    ModelAPIError,
)

# Bad credentials or a missing model would reject every remaining question the
# same way, so they end the run rather than filling a file with failures.
FATAL_STATUS_CODES = (401, 403, 404)


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
    verbose: bool = False,
) -> GenerationAttempt:
    """Keep parseable output, or recover the final response after a failed request."""
    prompt = render_question(question.sample)
    attempt = _unanswered(question, completion_id, instructions, prompt)
    messages: list[ModelMessage] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            result = await run_reasoned(
                agent, prompt, allow_empty=True, audit_messages=messages
            )
    except RECOVERABLE_ERRORS as error:
        _reraise_if_fatal(error)
        attempt.failure = GateFailure(
            gate="generation",
            reason=type(error).__name__,
            failure_type=FailureType.GENERATION_ERROR,
        )
        _salvage(attempt, messages)
    else:
        attempt.record.completion = CandidateCompletion(
            reasoning=result.reasoning,
            answer=result.output.answer,
        )
    finally:
        if verbose:
            _print_exchange(question.question_id, completion_id, messages)
    if messages:
        attempt.record.prompting = Prompting(full_prompt=prompt_sent(messages))
    return attempt


def _unanswered(
    question: Question,
    completion_id: int,
    instructions: str,
    prompt: str,
) -> GenerationAttempt:
    """The record as it stands before the model has said anything.

    Building it up front means a request that dies without ever reaching the
    provider still leaves a record of the question and the prompt it was asked
    with; the prompt actually put on the wire replaces this one if it exists.
    """
    return GenerationAttempt(
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


def _reraise_if_fatal(error: Exception) -> None:
    """Let through the errors that every later question would hit too."""
    if isinstance(error, ModelHTTPError) and error.status_code in FATAL_STATUS_CODES:
        raise error


def _salvage(attempt: GenerationAttempt, messages: list[ModelMessage]) -> None:
    """Recover onto ``attempt`` whatever the model did manage to say.

    A rejected request usually still carries a full reasoning trace, and an
    answer the parser choked on; both are worth keeping for inspection.
    """
    final = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, ModelResponse)
        ),
        None,
    )
    if final is None:
        return
    attempt.record.completion.reasoning = reasoning_from(final)
    attempt.raw_response = _answer_channel(final)
    attempt.record.completion.answer = _answer_key(attempt.raw_response)


def _answer_channel(response: ModelResponse) -> str:
    """The answer as it arrived, before the parse that rejected it."""
    return "\n".join(
        part.content if isinstance(part, TextPart) else part.args_as_json_str()
        for part in response.parts
        if isinstance(part, (TextPart, ToolCallPart))
    )


def _answer_key(raw_response: str) -> str | None:
    """The chosen option key, for output that parses despite the failed request."""
    try:
        return CandidateCompletion.model_validate_json(raw_response).answer
    except ValueError:
        return None


def _print_exchange(
    question_id: int,
    completion_id: int,
    messages: list[ModelMessage],
) -> None:
    """Dump every response in the exchange, for eyeballing a local model."""
    responses: list[ModelMessage] = [
        message for message in messages if isinstance(message, ModelResponse)
    ]
    print(
        f"Ollama question {question_id}, completion {completion_id}:",
        ModelMessagesTypeAdapter.dump_json(responses, indent=2).decode(),
        sep="\n",
        flush=True,
    )
