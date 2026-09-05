"""Local-LLM plumbing: Ollama connection settings, health checks and agents."""

from src.llm.agents import build_agent, build_model
from src.llm.config import OllamaSettings
from src.llm.health import OllamaUnavailableError, list_installed_models, require_ready

__all__ = [
    "OllamaSettings",
    "OllamaUnavailableError",
    "build_agent",
    "build_model",
    "list_installed_models",
    "require_ready",
]
