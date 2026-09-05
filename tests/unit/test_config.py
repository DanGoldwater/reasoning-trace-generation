"""Unit tests for Ollama connection settings."""

import pytest

from src.llm.config import OllamaSettings


def test_defaults_when_no_env_vars_are_set() -> None:
    settings = OllamaSettings.from_env(env={})

    assert settings.base_url == "http://localhost:11434"
    assert settings.model_name == "qwen3:0.6b"
    assert settings.timeout_seconds == 120.0


def test_environment_variables_override_defaults() -> None:
    settings = OllamaSettings.from_env(
        env={
            "OLLAMA_BASE_URL": "http://ollama.internal:9999",
            "OLLAMA_MODEL": "llama3.2:1b",
            "OLLAMA_TIMEOUT_SECONDS": "12.5",
        }
    )

    assert settings.base_url == "http://ollama.internal:9999"
    assert settings.model_name == "llama3.2:1b"
    assert settings.timeout_seconds == 12.5


def test_openai_base_url_appends_the_v1_compatibility_path() -> None:
    settings = OllamaSettings(base_url="http://localhost:11434")

    assert settings.openai_base_url == "http://localhost:11434/v1"


def test_trailing_slashes_are_stripped_before_urls_are_derived() -> None:
    settings = OllamaSettings.from_env(
        env={"OLLAMA_BASE_URL": "http://localhost:11434/"}
    )

    assert settings.base_url == "http://localhost:11434"
    assert settings.openai_base_url == "http://localhost:11434/v1"


def test_unparseable_timeout_names_the_offending_variable() -> None:
    with pytest.raises(ValueError, match="OLLAMA_TIMEOUT_SECONDS"):
        OllamaSettings.from_env(env={"OLLAMA_TIMEOUT_SECONDS": "soon"})


def test_non_positive_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="OLLAMA_TIMEOUT_SECONDS"):
        OllamaSettings.from_env(env={"OLLAMA_TIMEOUT_SECONDS": "0"})
