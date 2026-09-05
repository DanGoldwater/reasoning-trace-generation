"""Fixtures binding the integration suite to a real, running Ollama server.

These tests are deliberately not skipped when Ollama is missing: a broken local
setup should fail loudly rather than pass silently.
"""

import pytest

from src.llm.config import OllamaSettings
from src.llm.health import require_ready

INTEGRATION_MODEL = "qwen3.5:4b"


@pytest.fixture(scope="session")
def settings() -> OllamaSettings:
    """Real settings pinned to the model the integration suite expects."""
    configured = OllamaSettings.from_env()
    return OllamaSettings(
        base_url=configured.base_url,
        model_name=INTEGRATION_MODEL,
        timeout_seconds=configured.timeout_seconds,
    )


@pytest.fixture(scope="session", autouse=True)
def _ollama_is_ready(settings: OllamaSettings) -> None:
    """Fail the whole integration suite up front if Ollama is not usable."""
    require_ready(settings)
