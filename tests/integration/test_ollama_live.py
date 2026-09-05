"""Server-level checks against a real local Ollama, unmocked.

Only ``live_text`` costs a model call, and it is shared; everything else here
talks to the server's HTTP API, which is free.
"""

import pytest

from src.llm.config import INTEGRATION_TEST_TIMEOUT_SECONDS, OllamaSettings
from src.llm.health import (
    OllamaUnavailableError,
    list_installed_models,
    require_ready,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.timeout(INTEGRATION_TEST_TIMEOUT_SECONDS),
]


def test_the_model_is_installed_on_the_live_server(settings: OllamaSettings) -> None:
    assert settings.model_name in list_installed_models(settings)


def test_require_ready_rejects_a_model_that_is_not_installed(
    settings: OllamaSettings,
) -> None:
    missing = OllamaSettings(
        base_url=settings.base_url, model_name="definitely-not-pulled:0b"
    )

    with pytest.raises(
        OllamaUnavailableError, match="ollama pull definitely-not-pulled:0b"
    ):
        require_ready(missing)


def test_the_live_model_answers_an_unstructured_prompt(live_text: str) -> None:
    """The plain-text path, which structured output bypasses entirely."""
    assert "paris" in live_text.lower()
