"""Running an agent for a reasoning trace and a structured answer together."""

from dataclasses import dataclass

from pydantic_ai import Agent, UnexpectedModelBehavior, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    UserPromptPart,
)


class MissingReasoningError(RuntimeError):
    """Raised when a model answered without emitting any reasoning trace."""


class ReasoningOverranError(RuntimeError):
    """Raised when a model used its whole token budget and never finished thinking.

    Schema-constrained decoding governs the answer channel only. Asked something
    it cannot answer within the schema, a model can reason in circles until the
    budget runs out, never reaching the answer at all.
    """


@dataclass(frozen=True)
class Reasoned[OutputT]:
    """A model's reasoning trace paired with the answer it argued its way to."""

    reasoning: str
    output: OutputT
    prompt: str


async def run_reasoned[OutputT](
    agent: Agent[None, OutputT],
    user_prompt: str,
) -> Reasoned[OutputT]:
    """Run ``agent`` and return its reasoning trace alongside its typed output.

    Only the final response's thinking is kept. A malformed attempt that
    pydantic-ai retried was argued towards an answer that was thrown away, so
    attributing its reasoning to the answer we return would be a lie.
    """
    with capture_run_messages() as captured:
        try:
            result = await agent.run(user_prompt)
        except UnexpectedModelBehavior as error:
            # A run that never produced a parseable answer usually ran out of
            # budget mid-thought; say so rather than leaving a bare parse error.
            _reject_if_overran(_final_response(captured))
            raise ReasoningOverranError(str(error)) from error

    messages = result.all_messages()
    final = _final_response(messages)
    _reject_if_overran(final)

    reasoning = "\n".join(
        part.content for part in final.parts if isinstance(part, ThinkingPart)
    ).strip()
    if not reasoning:
        message = (
            f"{final.model_name or 'The model'} answered without a reasoning "
            "trace. Check that the model supports thinking and that thinking "
            "is enabled."
        )
        raise MissingReasoningError(message)

    return Reasoned(
        reasoning=reasoning,
        output=result.output,
        prompt=_prompt_sent(messages),
    )


def _reject_if_overran(response: ModelResponse) -> None:
    """Refuse a trace that was cut off mid-thought by the token budget."""
    if response.finish_reason != "length":
        return
    model = response.model_name or "The model"
    message = (
        f"{model} used its whole token budget without finishing its reasoning, "
        "so the trace is truncated mid-thought. This usually means the question "
        "cannot be answered within the output schema. Raise max_tokens if the "
        "question is genuinely hard, or drop the sample."
    )
    raise ReasoningOverranError(message)


def _final_response(messages: list[ModelMessage]) -> ModelResponse:
    """The last response the model sent, which is the one that answered."""
    for message in reversed(messages):
        if isinstance(message, ModelResponse):
            return message
    error = "The agent run produced no model response at all."
    raise MissingReasoningError(error)


def _prompt_sent(messages: list[ModelMessage]) -> str:
    """The instructions and question as they went out, read back off the request.

    Deriving this from the request rather than from the caller's arguments keeps
    it honest: whatever pydantic-ai put on the wire is what gets recorded.
    """
    for message in messages:
        if isinstance(message, ModelRequest):
            blocks = [message.instructions or ""]
            blocks += [
                part.content
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            ]
            return "\n\n".join(block for block in blocks if block)
    return ""
