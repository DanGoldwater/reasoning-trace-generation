"""Fixtures binding the integration suite to a real, running Ollama server.

These tests are deliberately not skipped when Ollama is missing: a broken local
setup should fail loudly rather than pass silently.
"""

import pytest

from src.llm.config import OllamaSettings
from src.llm.health import require_ready


@pytest.fixture(scope="session")
def settings() -> OllamaSettings:
    """Real settings from the centrally-defined integration profile."""
    return OllamaSettings.integration_from_env()


@pytest.fixture(scope="session", autouse=True)
def _ollama_is_ready(settings: OllamaSettings) -> None:
    """Fail the whole integration suite up front if Ollama is not usable."""
    require_ready(settings)
