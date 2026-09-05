"""Unit tests for Ollama connection settings."""

import pytest

from src.settings import (
    DEFAULT_GENERATION_MAX_TOKENS,
    INTEGRATION_GENERATION_MAX_TOKENS,
    INTEGRATION_MODEL_NAME,
    INTEGRATION_REQUEST_TIMEOUT_SECONDS,
    AnthropicSettings,
    OllamaSettings,
    settings_from_env,
)


def test_defaults_when_no_env_vars_are_set() -> None:
    settings = OllamaSettings.from_env(env={})

    assert settings.base_url == "http://localhost:11434"
    assert settings.model_name == "qwen3.5:4b"
    assert settings.timeout_seconds == 120.0
    assert settings.generation_max_tokens == DEFAULT_GENERATION_MAX_TOKENS


def test_environment_variables_override_defaults() -> None:
    settings = OllamaSettings.from_env(
        env={
            "OLLAMA_BASE_URL": "http://ollama.internal:9999",
            "OLLAMA_MODEL": "llama3.2:1b",
            "OLLAMA_TIMEOUT_SECONDS": "12.5",
            "OLLAMA_GENERATION_MAX_TOKENS": "321",
        }
    )

    assert settings.base_url == "http://ollama.internal:9999"
    assert settings.model_name == "llama3.2:1b"
    assert settings.timeout_seconds == 12.5
    assert settings.generation_max_tokens == 321


def test_integration_settings_share_the_runtime_setting_names() -> None:
    settings = OllamaSettings.integration_from_env(
        env={
            "OLLAMA_BASE_URL": "http://ollama.internal:9999",
            "OLLAMA_MODEL": "ignored-by-the-integration-profile:1b",
            "OLLAMA_TIMEOUT_SECONDS": "12.5",
            "OLLAMA_GENERATION_MAX_TOKENS": "321",
        }
    )

    assert settings.base_url == "http://ollama.internal:9999"
    assert settings.model_name == INTEGRATION_MODEL_NAME
    assert settings.timeout_seconds == INTEGRATION_REQUEST_TIMEOUT_SECONDS
    assert settings.generation_max_tokens == INTEGRATION_GENERATION_MAX_TOKENS


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


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_generation_budget_names_the_offending_variable(value: str) -> None:
    with pytest.raises(ValueError, match="OLLAMA_GENERATION_MAX_TOKENS"):
        OllamaSettings.from_env(env={"OLLAMA_GENERATION_MAX_TOKENS": value})


def test_provider_selection_defaults_to_ollama() -> None:
    settings = settings_from_env(env={})

    assert settings == OllamaSettings()


def test_anthropic_settings_are_loaded_from_the_environment() -> None:
    settings = settings_from_env(
        env={
            "LLM_PROVIDER": "anthropic",
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL": "claude-test",
            "ANTHROPIC_TIMEOUT_SECONDS": "12.5",
            "ANTHROPIC_GENERATION_MAX_TOKENS": "321",
        }
    )

    assert settings == AnthropicSettings(
        api_key="test-key",
        model_name="claude-test",
        timeout_seconds=12.5,
        generation_max_tokens=321,
    )


def test_anthropic_provider_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        settings_from_env(env={"LLM_PROVIDER": "anthropic"})


def test_unknown_provider_is_rejected_with_the_setting_name() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        settings_from_env(env={"LLM_PROVIDER": "unknown"})
