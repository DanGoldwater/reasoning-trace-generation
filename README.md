# reasoning-trace-generation

## Dataset access

Create a local `.env` file containing a Hugging Face token that can read the
private dataset:

```sh
echo 'HF_TOKEN=hf_your_token_here' > .env
```

Put your real token in that file, then fetch the sample data with:

```sh
uv run python src/data_fetching.py
```

The first 50 records are saved to `./data/private_qa.json`. The `data/`
directory is intentionally ignored by Git because it may contain private data.

Environment variables already set in the shell take precedence over `.env`.

## LLM providers

Generation always runs through [Pydantic AI](https://ai.pydantic.dev), which
provides one `build_model` / `build_agent` interface over a local
[Ollama](https://ollama.com) server and the native Anthropic API. Ollama is the
default, so the existing local workflow remains unchanged. Install Ollama, then
pull the production model and the smaller integration-test model:

```sh
ollama pull qwen3.5:4b
ollama pull qwen3.5:0.8b
```

`src/llm/` holds the plumbing:

| Module | Purpose |
|---|---|
| `config.py` | provider settings and `settings_from_env` — read from the environment |
| `health.py` | `list_installed_models` / `require_ready` — fail early with an actionable message |
| `agents.py` | `build_model` / `build_agent` — pydantic-ai objects wired to the local server |

```python
from pydantic import BaseModel

from src.llm import build_agent, settings_from_env

class City(BaseModel):
    name: str
    country: str

settings = settings_from_env()

agent = build_agent(settings, output_type=City, instructions="Extract the city.")
result = await agent.run("The conference was held in Paris, France.")
```

For Ollama, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` and
`OLLAMA_GENERATION_MAX_TOKENS` override the defaults in `src/llm/config.py`.
For Anthropic, set these in `.env` (or in the shell):

```sh
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-sonnet-4-5  # optional
```

`ANTHROPIC_TIMEOUT_SECONDS` and `ANTHROPIC_GENERATION_MAX_TOKENS` optionally
override the corresponding Anthropic defaults. `.env` is loaded without
overwriting values already present in the shell. Agents sample at temperature 0
so traces are reproducible, and any non-`str` output type is requested with
schema-constrained decoding, which small models follow far more reliably than a
tool call.

## Tests

```sh
uv run pytest                      # everything
uv run pytest -m "not integration" # unit tests only, no server needed
```

`tests/unit/` is hermetic: HTTP is stubbed at the transport boundary and the
model at pydantic-ai's `FunctionModel` seam. `tests/integration/` runs against
the real Ollama server with the real `qwen3.5:0.8b`, unmocked. It derives its
model, request timeout and generation budget from the same `OllamaSettings`
shape as production, with smaller values defined centrally. Each live test has
a 30-second process timeout. Those tests deliberately **error rather than
skip** when the server is down or the model is missing, so a broken local setup
can never pass silently.

## Development checks

Install the development tools and Git hook:

```sh
uv sync --group dev
uv run pre-commit install
```

Run every check manually, including Gitleaks secret detection:

```sh
uv run pre-commit run --all-files
```

The hook runs Gitleaks secret detection, Ruff linting and formatting, and
basedpyright and ty type checking. Ruff and basedpyright are configured in
`pyproject.toml`; ty uses the project's `requires-python` setting.

To scan the full Git history (for example, before publishing the repository),
install the Gitleaks CLI and run:

```sh
gitleaks git --redact
```
