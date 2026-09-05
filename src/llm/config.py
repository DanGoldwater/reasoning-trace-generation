"""Connection settings for a local Ollama server."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "qwen3.5:4b"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_GENERATION_MAX_TOKENS = 1536

# Live integration tests exercise the same agent path with a smaller member of
# the production model family and a deliberately tight generation budget.
INTEGRATION_MODEL_NAME = "qwen3.5:0.8b"
INTEGRATION_REQUEST_TIMEOUT_SECONDS = 30.0
INTEGRATION_GENERATION_MAX_TOKENS = 64
INTEGRATION_TEST_TIMEOUT_SECONDS = 30.0

TIMEOUT_ENV_VAR = "OLLAMA_TIMEOUT_SECONDS"
GENERATION_MAX_TOKENS_ENV_VAR = "OLLAMA_GENERATION_MAX_TOKENS"


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
            timeout_seconds=_read_timeout(env),
            generation_max_tokens=_read_generation_max_tokens(env),
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


def _read_timeout(env: Mapping[str, str]) -> float:
    """Parse ``OLLAMA_TIMEOUT_SECONDS`` into a positive number of seconds."""
    raw = env.get(TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        timeout = float(raw)
    except ValueError:
        message = f"{TIMEOUT_ENV_VAR} must be a number, got {raw!r}."
        raise ValueError(message) from None

    if timeout <= 0:
        message = f"{TIMEOUT_ENV_VAR} must be greater than zero, got {timeout}."
        raise ValueError(message)

    return timeout


def _read_generation_max_tokens(env: Mapping[str, str]) -> int:
    """Parse the maximum number of generated tokens, including thinking."""
    raw = env.get(GENERATION_MAX_TOKENS_ENV_VAR)
    if raw is None:
        return DEFAULT_GENERATION_MAX_TOKENS

    try:
        max_tokens = int(raw)
    except ValueError:
        message = f"{GENERATION_MAX_TOKENS_ENV_VAR} must be an integer, got {raw!r}."
        raise ValueError(message) from None

    if max_tokens <= 0:
        message = (
            f"{GENERATION_MAX_TOKENS_ENV_VAR} must be greater than zero, "
            f"got {max_tokens}."
        )
        raise ValueError(message)

    return max_tokens
