"""Pydantic AI plumbing for Ollama and Anthropic."""

from src.llm.agents import build_agent, build_model
from src.llm.health import OllamaUnavailableError, list_installed_models, require_ready
from src.llm.reasoning import MissingReasoningError, Reasoned, run_reasoned
from src.settings import (
    AnthropicSettings,
    LLMSettings,
    OllamaSettings,
    settings_from_env,
)

__all__ = [
    "MissingReasoningError",
    "AnthropicSettings",
    "LLMSettings",
    "OllamaSettings",
    "OllamaUnavailableError",
    "Reasoned",
    "build_agent",
    "build_model",
    "list_installed_models",
    "require_ready",
    "run_reasoned",
    "settings_from_env",
]
