"""Validated provider and experiment settings, independent of LLM plumbing."""

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.generation.prompts import ANSWER_INSTRUCTIONS

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "qwen3.5:4b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_GENERATION_MAX_TOKENS = 1536
DEFAULT_ANTHROPIC_MODEL_NAME = "claude-sonnet-4-5"
INTEGRATION_MODEL_NAME = DEFAULT_MODEL_NAME
INTEGRATION_REQUEST_TIMEOUT_SECONDS = 60.0
INTEGRATION_GENERATION_MAX_TOKENS = DEFAULT_GENERATION_MAX_TOKENS
INTEGRATION_TEST_TIMEOUT_SECONDS = 90.0


class ProviderSettings(BaseSettings):
    """Load environment and .env values; explicit constructor values win."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, populate_by_name=True
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        if env is None:
            return cls()
        # Supplying every field also isolates explicit mappings from the real env.
        values = {
            (
                field.validation_alias
                if isinstance(field.validation_alias, str)
                else name
            ): env.get(
                field.validation_alias
                if isinstance(field.validation_alias, str)
                else name,
                field.default if not field.is_required() else None,
            )
            for name, field in cls.model_fields.items()
        }
        return cls.model_validate(values)


class OllamaSettings(ProviderSettings):
    """Local model connection and generation budget."""

    provider: Literal["ollama"] = "ollama"
    base_url: str = Field(default=DEFAULT_BASE_URL, validation_alias="OLLAMA_BASE_URL")
    model_name: str = Field(default=DEFAULT_MODEL_NAME, validation_alias="OLLAMA_MODEL")
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
        validation_alias="OLLAMA_TIMEOUT_SECONDS",
    )
    generation_max_tokens: int = Field(
        default=DEFAULT_GENERATION_MAX_TOKENS,
        gt=0,
        validation_alias="OLLAMA_GENERATION_MAX_TOKENS",
    )

    @field_validator("base_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    @classmethod
    def integration_from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        return cls.from_env(env).model_copy(
            update={
                "model_name": INTEGRATION_MODEL_NAME,
                "timeout_seconds": INTEGRATION_REQUEST_TIMEOUT_SECONDS,
                "generation_max_tokens": INTEGRATION_GENERATION_MAX_TOKENS,
            }
        )


class AnthropicSettings(ProviderSettings):
    """Native API settings; credentials never appear in dumps or repr."""

    provider: Literal["anthropic"] = "anthropic"
    api_key: str = Field(
        repr=False, exclude=True, min_length=1, validation_alias="ANTHROPIC_API_KEY"
    )
    model_name: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL_NAME, validation_alias="ANTHROPIC_MODEL"
    )
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        allow_inf_nan=False,
        validation_alias="ANTHROPIC_TIMEOUT_SECONDS",
    )
    generation_max_tokens: int = Field(
        default=DEFAULT_GENERATION_MAX_TOKENS,
        gt=0,
        validation_alias="ANTHROPIC_GENERATION_MAX_TOKENS",
    )


type LLMSettings = OllamaSettings | AnthropicSettings


class ProviderSelection(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    provider: Literal["ollama", "anthropic"] = Field(
        default="ollama", validation_alias="LLM_PROVIDER"
    )


def settings_from_env(env: Mapping[str, str] | None = None) -> LLMSettings:
    selection = (
        ProviderSelection()
        if env is None
        else ProviderSelection.model_validate(
            {"LLM_PROVIDER": env.get("LLM_PROVIDER", "ollama")}
        )
    )
    if selection.provider == "ollama":
        return OllamaSettings.from_env(env)
    return AnthropicSettings.from_env(env)


class RunSettings(BaseSettings):
    """The full experiment configuration saved directly into run metadata."""

    model_config = SettingsConfigDict(
        env_prefix="RUN_", env_file=".env", extra="ignore", frozen=True
    )

    input_path: Path = Path("data/private_qa.json")
    runs_dir: Path = Path("data/runs")
    question_limit: int | None = Field(default=None, gt=0)
    completions_per_question: int = Field(default=1, gt=0)
    llm: LLMSettings = Field(
        default_factory=settings_from_env, discriminator="provider"
    )
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)
    instructions: str = ANSWER_INSTRUCTIONS
