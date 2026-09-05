# Anthropic through Pydantic AI: implementation notes

Research date: 2026-09-05. Sources below are the current official Pydantic AI
and Anthropic documentation.

## Dependency and configuration

The project currently depends on `pydantic-ai-slim[openai]`. Anthropic support
is supplied by Pydantic AI's `anthropic` extra, so retain the existing extra
and change the dependency to `pydantic-ai-slim[openai,anthropic]>=2.40.0`.
The normal direct-Anthropic path does not require a separately declared SDK
dependency. [Pydantic AI's Anthropic installation guide](https://pydantic.dev/docs/ai/models/anthropic/#install)
specifies `uv add "pydantic-ai-slim[anthropic]"`.

Use the same `load_dotenv()`-then-`os.environ` pattern as `OllamaSettings`.
The native provider reads `ANTHROPIC_API_KEY`; a local, uncommitted `.env` may
therefore contain `ANTHROPIC_API_KEY=...`. The key should be required only when
the Anthropic provider is selected, so the default Ollama path remains usable
without it. Pydantic AI documents both the variable and automatic construction
from it. [Pydantic AI: Anthropic environment-variable configuration](https://pydantic.dev/docs/ai/models/anthropic/#environment-variable)

Provider selection should be explicit (for example, an `LLM_PROVIDER` setting
whose default is `ollama`) and settings/factory code should return Pydantic AI
model objects. That preserves the application's single interaction surface:
`Agent`, model settings, structured output, and `Agent.override`, rather than
introducing direct calls to either vendor SDK. The default and the Anthropic
model name are project policy decisions; they are not prescribed by either API.

## Pydantic AI construction

For native Anthropic API access, the documented direct construction is:

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    "claude-sonnet-4-5",
    provider=AnthropicProvider(api_key="your-api-key"),
)
```

When the key is loaded into the environment, `AnthropicModel("claude-sonnet-4-5")`
is sufficient. Pydantic AI also supports the string identifier
`"anthropic:claude-sonnet-4-6"` when constructing an agent. [Pydantic AI:
Anthropic model/provider usage](https://pydantic.dev/docs/ai/models/anthropic/#provider-argument)

Keep the existing Ollama `OpenAIChatModel` plus `OllamaProvider` factory unless
changing it is necessary for the selected abstraction: it is already a Pydantic
AI model/provider pairing and existing tests assert it. Pydantic AI also has
an `OllamaModel`, but introducing it would be a behavioural/type change beyond
the provider-selection work and should be covered separately. [Pydantic AI:
Ollama model documentation](https://pydantic.dev/docs/ai/models/ollama/)

The existing provider-neutral `ModelSettings` fields (`timeout`, `temperature`,
and `max_tokens`) can continue to be supplied to the agent. If Anthropic-only
settings become necessary, use `AnthropicModelSettings`; Pydantic AI documents
provider-specific settings such as `top_k` and service tier. [Pydantic AI:
Anthropic model settings](https://pydantic.dev/docs/ai/models/anthropic/#model-settings)

If custom transport injection is added later, use `httpx2.AsyncClient`, not
the legacy `httpx.AsyncClient`: the current Anthropic integration requires the
former. The SDK client retries twice by default; when application transport
retries are intentionally authoritative, inject a client with `max_retries=0`
to avoid retry multiplication. [Pydantic AI: custom HTTP client and retries](https://pydantic.dev/docs/ai/models/anthropic/#custom-http-client)

## Unit-test boundary

Do not make a real Anthropic request in unit tests and do not require an API
key to import or exercise the factory. Test provider selection/configuration
directly (including a clear missing-key error), then test agent behaviour at
Pydantic AI's model seam. Pydantic AI recommends `TestModel` or `FunctionModel`
and `Agent.override`; set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` in
unit tests to prevent accidental live model requests. `FunctionModel` is the
right choice where the test must control a response or assert the messages sent.
[Pydantic AI unit-testing guide](https://pydantic.dev/docs/ai/guides/testing/)

This matches the repository's existing hermetic `FunctionModel` tests and lets
the Ollama integration suite stay live and unchanged. A focused Anthropic
factory test may mock/patch the provider constructor to assert model name and
key hand-off; it should never mock the application's reasoning/structured-output
logic, which should remain tested through `Agent.override`.
