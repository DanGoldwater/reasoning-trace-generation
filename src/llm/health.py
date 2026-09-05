"""Readiness checks for a local Ollama server."""

import contextlib
from collections.abc import Iterator

import httpx

from src.llm.config import OllamaSettings


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or the requested model is not usable."""


@contextlib.contextmanager
def _client_for(
    settings: OllamaSettings,
    client: httpx.Client | None,
) -> Iterator[httpx.Client]:
    """Yield the caller's client, or a short-lived one built from the settings."""
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=settings.timeout_seconds) as owned_client:
        yield owned_client


def list_installed_models(
    settings: OllamaSettings,
    client: httpx.Client | None = None,
) -> list[str]:
    """Return the names of every model the Ollama server has pulled."""
    with _client_for(settings, client) as active_client:
        response = active_client.get(f"{settings.base_url}/api/tags")
    response.raise_for_status()
    return [model["name"] for model in response.json()["models"]]


def require_ready(
    settings: OllamaSettings,
    client: httpx.Client | None = None,
) -> None:
    """Raise :class:`OllamaUnavailableError` unless the server is usable."""
    try:
        installed = list_installed_models(settings, client=client)
    except httpx.HTTPError as error:
        message = (
            f"Could not reach the Ollama server at {settings.base_url}: {error}. "
            "Start it with `ollama serve`."
        )
        raise OllamaUnavailableError(message) from error

    if settings.model_name not in installed:
        message = (
            f"The Ollama server at {settings.base_url} does not have "
            f"{settings.model_name} installed. Pull it with "
            f"`ollama pull {settings.model_name}`. Installed models: "
            f"{', '.join(installed) or 'none'}."
        )
        raise OllamaUnavailableError(message)
