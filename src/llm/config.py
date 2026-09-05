"""Connection settings for the LLM providers supported by this project."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "qwen3.5:4b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_GENERATION_MAX_TOKENS = 1536
DEFAULT_ANTHROPIC_MODEL_NAME = "claude-sonnet-4-5"

# Live integration tests exercise the same agent path with a smaller member of
# the production model family and a deliberately tight generation budget.
INTEGRATION_MODEL_NAME = "qwen3.5:0.8b"
INTEGRATION_REQUEST_TIMEOUT_SECONDS = 30.0
INTEGRATION_GENERATION_MAX_TOKENS = 64
INTEGRATION_TEST_TIMEOUT_SECONDS = 30.0

TIMEOUT_ENV_VAR = "OLLAMA_TIMEOUT_SECONDS"
GENERATION_MAX_TOKENS_ENV_VAR = "OLLAMA_GENERATION_MAX_TOKENS"
PROVIDER_ENV_VAR = "LLM_PROVIDER"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
ANTHROPIC_TIMEOUT_ENV_VAR = "ANTHROPIC_TIMEOUT_SECONDS"
ANTHROPIC_GENERATION_MAX_TOKENS_ENV_VAR = "ANTHROPIC_GENERATION_MAX_TOKENS"

type ProviderName = Literal["anthropic", "ollama"]


@dataclass(frozen=True)
class OllamaSettings:
    """Where the Ollama server lives and which model to talk to."""

    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def openai_base_url(self) -> str:
        """The OpenAI-compatible endpoint that pydantic-ai talks to."""
        return f"{self.base_url}/v1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "OllamaSettings":
        """Build settings from environment variables, falling back to defaults."""
        if env is None:
            load_dotenv()
            env = os.environ

        return cls(
            base_url=env.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
            model_name=env.get("OLLAMA_MODEL", DEFAULT_MODEL_NAME),
            timeout_seconds=_read_timeout(env, TIMEOUT_ENV_VAR),
            generation_max_tokens=_read_generation_max_tokens(
                env, GENERATION_MAX_TOKENS_ENV_VAR
            ),
        )

    @classmethod
    def integration_from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "OllamaSettings":
        """Build the integration profile from the same runtime settings shape.

        The local server location comes from the environment; model, request
        timeout and generation budget are pinned centrally so integration tests
        cannot silently drift from the runtime names used by agents.
        """
        return replace(
            cls.from_env(env),
            model_name=INTEGRATION_MODEL_NAME,
            timeout_seconds=INTEGRATION_REQUEST_TIMEOUT_SECONDS,
            generation_max_tokens=INTEGRATION_GENERATION_MAX_TOKENS,
        )


def _read_timeout(env: Mapping[str, str], variable_name: str) -> float:
    """Parse a provider timeout setting into a positive number of seconds."""
    raw = env.get(variable_name)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        timeout = float(raw)
    except ValueError:
        message = f"{variable_name} must be a number, got {raw!r}."
        raise ValueError(message) from None

    if timeout <= 0:
        message = f"{variable_name} must be greater than zero, got {timeout}."
        raise ValueError(message)

    return timeout


def _read_generation_max_tokens(env: Mapping[str, str], variable_name: str) -> int:
    """Parse the maximum number of generated tokens, including thinking."""
    raw = env.get(variable_name)
    if raw is None:
        return DEFAULT_GENERATION_MAX_TOKENS

    try:
        max_tokens = int(raw)
    except ValueError:
        message = f"{variable_name} must be an integer, got {raw!r}."
        raise ValueError(message) from None

    if max_tokens <= 0:
        message = f"{variable_name} must be greater than zero, got {max_tokens}."
        raise ValueError(message)

    return max_tokens


@dataclass(frozen=True)
class AnthropicSettings:
    """Credentials and generation settings for Anthropic's native API."""

    api_key: str = field(repr=False)
    model_name: str = DEFAULT_ANTHROPIC_MODEL_NAME
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    generation_max_tokens: int = DEFAULT_GENERATION_MAX_TOKENS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AnthropicSettings":
        """Build settings from ``ANTHROPIC_*`` environment variables.

        The key is required here, rather than at module import time, so Ollama
        remains fully usable on machines that do not have Anthropic credentials.
        """
        if env is None:
            load_dotenv()
            env = os.environ

        api_key = env.get(ANTHROPIC_API_KEY_ENV_VAR)
        if not api_key:
            message = f"Set {ANTHROPIC_API_KEY_ENV_VAR} to use the Anthropic provider."
            raise ValueError(message)

        return cls(
            api_key=api_key,
            model_name=env.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL_NAME),
            timeout_seconds=_read_timeout(env, ANTHROPIC_TIMEOUT_ENV_VAR),
            generation_max_tokens=_read_generation_max_tokens(
                env, ANTHROPIC_GENERATION_MAX_TOKENS_ENV_VAR
            ),
        )


type LLMSettings = OllamaSettings | AnthropicSettings


def settings_from_env(env: Mapping[str, str] | None = None) -> LLMSettings:
    """Select a provider from ``LLM_PROVIDER`` and load its settings.

    Ollama stays the default to retain the existing local-development workflow.
    """
    if env is None:
        load_dotenv()
        env = os.environ

    provider = env.get(PROVIDER_ENV_VAR, "ollama")
    if provider == "ollama":
        return OllamaSettings.from_env(env)
    if provider == "anthropic":
        return AnthropicSettings.from_env(env)

    message = f"{PROVIDER_ENV_VAR} must be 'ollama' or 'anthropic', got {provider!r}."
    raise ValueError(message)
