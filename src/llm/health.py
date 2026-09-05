"""Readiness checks for a local Ollama server."""

import httpx

from src.settings import OllamaSettings


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server or the requested model is not usable."""


def list_installed_models(settings: OllamaSettings, client: httpx.Client) -> list[str]:
    """Return the names of every model the Ollama server has pulled."""
    response = client.get(f"{settings.base_url}/api/tags")
    response.raise_for_status()
    return [model["name"] for model in response.json()["models"]]


def require_ready(settings: OllamaSettings, client: httpx.Client) -> None:
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
