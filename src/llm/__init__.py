"""Local-LLM plumbing: Ollama connection settings, health checks and agents."""

from src.llm.agents import build_agent, build_model
from src.llm.config import OllamaSettings
from src.llm.health import OllamaUnavailableError, list_installed_models, require_ready
from src.llm.reasoning import MissingReasoningError, Reasoned, run_reasoned

__all__ = [
    "MissingReasoningError",
    "OllamaSettings",
    "OllamaUnavailableError",
    "Reasoned",
    "build_agent",
    "build_model",
    "list_installed_models",
    "require_ready",
    "run_reasoned",
]
