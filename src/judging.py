"""An Anthropic quality gate over completed reasoning, never a generator."""

import asyncio
from typing import Literal

import httpx
from anthropic import AsyncAnthropic
from anthropic.types.beta import (
    BetaThinkingConfigAdaptiveParam,
    BetaThinkingConfigDisabledParam,
    BetaThinkingConfigParam,
)
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from src.quality import (
    CandidateRecord,
    FailureType,
    GateFailure,
    JudgeVerdict,
    QualityGate,
)
from src.settings import JudgeSettings


def thinking_config(mode: Literal["adaptive", "disabled"]) -> BetaThinkingConfigParam:
    """Widen the configured mode into the request shape the API expects."""
    if mode == "adaptive":
        return BetaThinkingConfigAdaptiveParam(type="adaptive")
    return BetaThinkingConfigDisabledParam(type="disabled")


class JudgeInput(BaseModel):
    """Withhold the gold answer so label agreement cannot bias the judge."""

    question: str
    options: dict[str, str]
    answer: str
    reasoning: str


class ReasoningHallucination(QualityGate):
    name = "reasoning_hallucination"
    failure_type = FailureType.UNSUPPORTED_REASONING
    requires_complete_generation = True

    def __init__(self, settings: JudgeSettings, *, model: Model | None = None) -> None:
        self.settings = settings
        if model is None:
            if settings.api_key is None:
                raise ValueError("The judge requires ANTHROPIC_API_KEY.")
            model = AnthropicModel(
                settings.model_name,
                provider=AnthropicProvider(
                    anthropic_client=AsyncAnthropic(
                        api_key=settings.api_key, max_retries=settings.request_retries
                    )
                ),
            )
        self.agent = Agent(
            model,
            output_type=NativeOutput(JudgeVerdict),
            instructions=settings.instructions,
            retries=settings.output_retries,
            model_settings=AnthropicModelSettings(
                max_tokens=settings.max_tokens,
                timeout=settings.timeout_seconds,
                anthropic_thinking=thinking_config(settings.thinking),
            ),
        )

    async def check(self, record: CandidateRecord) -> GateFailure | None:
        answer = record.completion.answer
        if (
            answer is None
            or answer not in record.hf_sample.options
            or not record.completion.reasoning.strip()
        ):
            return None
        candidate = JudgeInput(
            question=record.hf_sample.question,
            options=record.hf_sample.options,
            answer=answer,
            reasoning=record.completion.reasoning,
        )
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                result = await self.agent.run(candidate.model_dump_json())
        except (
            TimeoutError,
            httpx.HTTPError,
            ModelAPIError,
            UnexpectedModelBehavior,
        ) as error:
            return GateFailure(
                gate=self.name,
                failure_type=FailureType.JUDGE_ERROR,
                reason=f"Judge did not return a valid verdict: {type(error).__name__}.",
            )
        verdict = result.output
        if not verdict.has_significant_hallucination:
            return None
        return GateFailure(
            gate=self.name,
            failure_type=self.failure_type,
            reason=verdict.explanation,
            judge_verdict=verdict,
        )
