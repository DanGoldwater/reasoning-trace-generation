# reasoning-trace-generation

Generate and filter reasoning traces from the private biology Q&A dataset using
Pydantic AI and a local Ollama model.

## Setup and execution

Install Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com), then run:

```sh
uv sync --group dev
ollama pull qwen3.5:4b
```

Start Ollama (`ollama serve` if the desktop app is not running). For dataset
access, create `.env` in the repository root:

```dotenv
HF_TOKEN=hf_your_token_here
```

The token must have read access to `owkin/technical_test`. Run an experiment:

```sh
uv run python main.py --llm-judge off --question-limit 1
uv run python main.py --llm-judge on --question-limit 50 --completions-per-question 2
```

`--llm-judge on|off` is required on every run. It turns the Anthropic
hallucination gate on or off, and there is deliberately no default: judging
changes what a run measures and what it costs, so each run states the decision
outright. `on` requires `ANTHROPIC_API_KEY`.

The runner uses `data/private_qa.json` if it exists. Otherwise it fetches up to
1,000 rows from Hugging Face and saves the cache before generation. A failed fetch
reports the need to set `HF_TOKEN` in `.env` and check access/connectivity. An
existing but invalid input file produces a validation error; it is not replaced.
To fetch explicitly:

```sh
uv run python -m src.data_fetching
```

Use `--input-path`, `--runs-dir`, `--question-limit`,
`--completions-per-question`, and `--temperature` to override run defaults.
These have defaults; `--llm-judge` does not.
Without a limit, every cached question is processed. Question IDs are zero-based
positions in the input file; completion IDs restart at zero for each question.
Multiple completions at temperature zero may be identical.

## Run artifacts

Each run gets a unique three-word petname and a new directory:

```text
data/runs/<petname>/
    passed.jsonl
    failed.jsonl
    run.json
```

The processing loop is question → generation → all gates → disk, repeated for
each completion. Each JSONL append is closed before the next generation begins;
metadata is then replaced atomically. Ctrl-C preserves completed records and
marks the run interrupted. A hard kill can leave metadata marked running or its
counts one record behind; JSONL files are the source of truth. This prototype
does not resume runs or guarantee persistence through a power failure.

Passing rows match the assignment schema exactly:

```json
{
  "question_id": 0,
  "completion_id": 0,
  "hf_sample": {
    "question": "According to the screen, is this cell line sensitive?",
    "answer": "B",
    "options": {"A": "No", "B": "Yes"}
  },
  "completion": {
    "reasoning": "The screen reports sensitivity, so the answer is yes.",
    "answer": "B"
  },
  "prompting": {
    "full_prompt": "[instructions and question sent to the model]"
  }
}
```

Failed rows contain `record` (the same nested fields), `failures` (a list of
`{"gate": "...", "reason": "..."}` objects), and `raw_response` when generation
failed. Missing answers are `null`; missing reasoning is an empty string. This
preserves incorrect answers and partial traces for error analysis. Only the final
model response is associated with an answer; reasoning from a discarded
validation retry is never attached to a later successful answer.

`run.json` records the complete validated settings, enabled gate names, source
file SHA-256, timestamps, status, and accepted/rejected counts. Settings are dumped
directly, so new configuration fields automatically appear in metadata.
Anthropic credentials are excluded from serialization and representations.

## Configuration and design

`src/settings.py` owns Pydantic Settings models for provider selection, each
provider, and the experiment. Explicit values override environment variables,
which override `.env` and defaults. Run settings use the `RUN_` prefix, for
example `RUN_QUESTION_LIMIT=10` and `RUN_TEMPERATURE=0.3`.

The one exception is the judging decision. `RUN_LLM_JUDGE` is ignored, and
`RunSettings` cannot be constructed without `llm_judge`, so no ambient variable
can decide judging for a run that never asked for it.

Ollama is the default provider. Its environment variables are `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, and `OLLAMA_GENERATION_MAX_TOKENS`.
Defaults are `http://localhost:11434`, `qwen3.5:4b`, 120 seconds per completion
(including validation retries), and 1,536 tokens including thinking.

The existing native Anthropic path remains available for comparison experiments:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_api_key
ANTHROPIC_MODEL=claude-sonnet-4-5
```

`ANTHROPIC_TIMEOUT_SECONDS` and `ANTHROPIC_GENERATION_MAX_TOKENS` override the
same default budgets. Use Ollama for the assignment's local-model requirement.

The code separates the main responsibilities:

| Module | Responsibility |
|---|---|
| `src/settings.py` | Validate settings and load environment values |
| `src/dataset/` | Validate source/accepted records and read/write data |
| `src/llm/` | Build provider-neutral agents, check readiness, capture reasoning |
| `src/generation/` | Prompting, option-constrained answers, recover failed attempts |
| `src/quality.py` | Candidate/failure models and extensible quality gates |
| `src/experiments.py` | Sequential orchestration and run metadata |

Pydantic models validate external data, settings, and persisted artifacts. The
answer model adds each question's allowed keys to its JSON schema and validates
them at runtime, without dynamic `Literal` construction or casts. The internal
generic reasoning result stays a small immutable dataclass.

## Filtering and model trade-offs

Both default gates run for every candidate:

- `non_empty_reasoning`: reject missing or whitespace-only reasoning.
- `correct_answer`: require exact equality with the ground-truth option key.

Malformed answers, truncated generation, timeouts, and provider connection errors
also produce a `generation` failure and the run continues. Authentication and
missing-model errors stop the run; Ollama readiness is checked before creating a
run directory. Disk errors and unexpected programming errors stop the run too.
Pydantic AI can retry malformed structured output once within each completion's
deadline; quality-gate failures do not trigger regeneration.

To extend filtering, subclass `QualityGate`, give it a unique `name`, and implement
`check(record) -> str | None`: return a rejection reason or `None` to accept.
Supply gates through `run_experiment(settings, gates=[...])`; this explicit list
replaces the defaults, so include `NonEmptyReasoning()` and `CorrectAnswer()` when
adding another check. Metadata records the supplied names, and failures retain
all applicable rejection reasons.

These gates establish output usability and answer agreement, not scientific
correctness of the reasoning. A model can guess the right answer and justify it
poorly; a mislabeled source answer can also reject a sound response. The gold
answer is never placed in the generation prompt. A later gate could inspect
unsupported claims or use a separate verifier, at an additional inference cost.

The 4B model is a pragmatic size for the assignment's consumer-laptop budget.
A smaller model reduces memory and latency but has less capacity for specialist
biology reasoning; a larger model can improve capacity at greater memory and
inference cost. Schema-constrained answers reduce formatting failures but do not
correct scientific mistakes. The token cap bounds cost and catches truncated
thinking, at the risk of rejecting questions needing longer reasoning. Sequential
execution limits concurrent memory pressure. Temperature zero reduces sampling
variation but is not a guarantee of reproducibility across model/runtime changes.

## Tests and checks

```sh
uv run pytest
uv run pytest -m integration
uv run pytest -m "not integration"
uv run pre-commit run --all-files
uv run pre-commit install
```

The single live integration test runs one synthetic, dataset-shaped question
through the production runner and both gates into a temporary run directory. It
asserts the exact artifact layout, IDs, answer, reasoning, prompt, metadata,
counts, input identity, and JSON round-trip. It uses `qwen3.5:4b` with a 1,536-token
budget, a 60-second completion deadline, and a 90-second test timeout. Missing
Ollama or its model fails rather than skips. The test verifies the pipeline
contract, not biomedical accuracy. No extra live queries exercise rejection paths.

Unit tests use Pydantic AI's `FunctionModel` at the inference boundary and real
temporary files. They cover failures, incremental writes, interruption, custom
gates, fetching, and configuration without contacting a model provider.

Pre-commit runs Gitleaks, Ruff lint/format, basedpyright, and ty. Explicit `Any`
is banned across source and tests by basedpyright's
[`reportExplicitAny`](https://docs.basedpyright.com/latest/benefits-over-pyright/new-diagnostic-rules/#reportexplicitany)
and Ruff's banned imports. Third-party inferred types are not globally banned;
our data boundaries validate them into concrete models.

The `data/` directory and `.env` are ignored by Git. To scan Git history before
publishing, install Gitleaks and run `gitleaks git --redact`.
