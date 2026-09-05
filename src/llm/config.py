"""Connection settings for a local Ollama server."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "qwen3:0.6b"
DEFAULT_TIMEOUT_SECONDS = 120.0
TIMEOUT_ENV_VAR = "OLLAMA_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class OllamaSettings:
    """Where the Ollama server lives and which model to talk to."""

    base_url: str = DEFAULT_BASE_URL
    model_name: str = DEFAULT_MODEL_NAME
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

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
