"""Unit tests for the Ollama health check, stubbed at the HTTP boundary."""

from collections.abc import Callable

import httpx
import pytest

from src.llm.health import (
    OllamaUnavailableError,
    list_installed_models,
    require_ready,
)
from src.settings import OllamaSettings

SETTINGS = OllamaSettings(base_url="http://localhost:11434", model_name="qwen3:0.6b")


Handler = Callable[[httpx.Request], httpx.Response]


def client_returning(handler: Handler) -> httpx.Client:
    """A real httpx client whose transport is stubbed at the HTTP boundary."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_installed_models_are_read_from_the_tags_endpoint() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:0.6b"}, {"name": "gemma3:1b"}]},
        )

    with client_returning(handler) as client:
        models = list_installed_models(SETTINGS, client=client)

    assert models == ["qwen3:0.6b", "gemma3:1b"]
    assert requested == ["http://localhost:11434/api/tags"]


def test_unreachable_server_raises_with_the_base_url_in_the_message() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with client_returning(handler) as client:
        with pytest.raises(OllamaUnavailableError, match="http://localhost:11434"):
            require_ready(SETTINGS, client=client)


def test_missing_model_raises_with_the_pull_command() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "gemma3:1b"}]})

    with client_returning(handler) as client:
        with pytest.raises(OllamaUnavailableError, match="ollama pull qwen3:0.6b"):
            require_ready(SETTINGS, client=client)


def test_ready_server_with_the_model_installed_passes_silently() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:0.6b"}]})

    with client_returning(handler) as client:
        require_ready(SETTINGS, client=client)
