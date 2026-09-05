# Reasoning Trace Dataset Generation

Generate and filter reasoning traces from the private biology Q&A dataset using
Pydantic AI and a local Ollama model, with an optional Anthropic judge that
rejects hallucinated reasoning.

>[!Note] This readme is about how to get things running. Largely Claude written. Info about architecture, design decisions, etc, are in `docs/slides.md`. My intention is to use these slides in the presentation component. 

## Setup and execution

Install uv and Ollama:
```sh
brew install ollama
brew install uv
```

```sh
uv sync --group dev
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
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
`--completions-per-question`, and `--temperature` to override run defaults, and
`--verbose-ollama` to dump every Ollama response after each attempt. These have
defaults; `--llm-judge` does not.
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
`run.json` is then rewritten in full. Ctrl-C preserves completed records and
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

Failed rows contain `record` (the same nested fields), `failures`, and
`raw_response` when generation failed. Each failure carries the `gate` that
rejected the row, a human-readable `reason`, a `failure_type` for grouping
(`no_answer`, `wrong_answer`, `missing_reasoning`, `unsupported_reasoning`,
`generation_error`, `judge_error`, or `custom`), and the structured
`judge_verdict` when the hallucination gate rejected it. Missing answers are
`null`; missing reasoning is an empty string. This preserves incorrect answers
and partial traces for error analysis. Only the final model response is
associated with an answer; reasoning from a discarded validation retry is never
attached to a later successful answer.

`run.json` records the complete validated settings, enabled gate names, source
file SHA-256, timestamps, status, and accepted/rejected counts. Settings are dumped
directly, so new configuration fields automatically appear in metadata, judge
configuration included. Anthropic credentials, for both the provider and the
judge, are excluded from serialization and representations.

## Configuration and design

`src/settings.py` owns Pydantic Settings models for provider selection, each
provider, the judge, and the experiment. Explicit values override environment
variables, which override `.env` and defaults. Run settings use the `RUN_`
prefix, for example `RUN_QUESTION_LIMIT=10` and `RUN_TEMPERATURE=0.3`.

The one exception is the judging decision. `RUN_LLM_JUDGE` is ignored, and
`RunSettings` cannot be constructed without `llm_judge`, so no ambient variable
can decide judging for a run that never asked for it.

Judge settings live in the same file under the `JUDGE_` prefix and are saved
into run metadata with everything else. The defaults are `claude-sonnet-5`, a
60-second deadline, 1,536 tokens, thinking disabled, and one structured-output
retry; `JUDGE_MODEL_NAME`, `JUDGE_TIMEOUT_SECONDS`, `JUDGE_MAX_TOKENS`,
`JUDGE_THINKING`, and `JUDGE_INSTRUCTIONS` override them. The key itself is read
from `ANTHROPIC_API_KEY`, so the judge shares credentials with the Anthropic
provider without requiring it to be the generating provider.

Ollama is the default provider. Its main environment variables are
`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, and
`OLLAMA_GENERATION_MAX_TOKENS`.
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
| `src/judging.py` | The Anthropic hallucination gate over completed traces |
| `src/experiments.py` | Sequential orchestration and run metadata |

Pydantic models validate external data, settings, and persisted artifacts. The
answer model adds each question's allowed keys to its JSON schema and validates
them at runtime, without dynamic `Literal` construction or casts. The internal
generic reasoning result stays a small immutable dataclass.

## Filtering and model trade-offs

Three gates run for every candidate by default:

- `non_empty_answer`: reject a missing or whitespace-only answer.
- `non_empty_reasoning`: reject missing or whitespace-only reasoning.
- `correct_answer`: require exact equality with the ground-truth option key,
  leaving a missing answer to `non_empty_answer` alone.

`--llm-judge on` appends a fourth, `reasoning_hallucination`: an Anthropic model
reads the question, the options, and the trace, and rejects reasoning whose
significant claims are fabricated. It sees the answer the local model chose but
never the gold answer, so label agreement cannot bias the verdict. It runs only
on completions that generated cleanly, and skips rows with no reasoning or an
off-menu answer. A judge that times out or returns nothing usable rejects that
one row with a `judge_error` failure rather than ending the run.

Malformed answers, truncated generation, timeouts, and provider connection errors
also produce a `generation` failure and the run continues. Authentication and
missing-model errors stop the run; Ollama readiness is checked before creating a
run directory. Disk errors and unexpected programming errors stop the run too.
Pydantic AI can retry malformed structured output once within each completion's
deadline; quality-gate failures do not trigger regeneration.

To extend filtering, subclass `QualityGate`, give it a unique `name`, and
implement `async check(record) -> GateFailure | None`: return `self.reject(...)`
with a reason, or `None` to accept. Set `failure_type` to group its rejections,
and `requires_complete_generation = True` to skip candidates whose generation
already failed. Supply gates through `run_experiment(settings, gates=[...])`;
this explicit list replaces the defaults, so include `NonEmptyAnswer()`,
`NonEmptyReasoning()`, and `CorrectAnswer()` when adding another check. The judge
is appended to whatever list is supplied whenever `--llm-judge on`. Metadata
records the resulting names, and every applicable rejection is kept.

The default gates establish output usability and answer agreement, not scientific
correctness of the reasoning. A model can guess the right answer and justify it
poorly; a mislabeled source answer can also reject a sound response. The gold
answer is never placed in the generation prompt. The hallucination judge is the
step past agreement — a second model's opinion on whether the trace is
supportable — and it is a model assessment, not literature verification. It
costs an API call per completed candidate, which is why it is decided per run.

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

[just](https://github.com/casey/just) wraps the routine commands: `just test`
runs pytest, `just check` runs the tests and then pre-commit, `just plots`
regenerates the figures behind `docs/eda_report.md`, and `just slides`
re-exports the Excalidraw diagrams and builds `docs/slides.html` from
`docs/slides.md` (both of the last two need `npx`).

The single live integration test runs one synthetic, dataset-shaped question
through the production runner and the three default gates, judging off, into a
temporary run directory. It asserts the exact artifact layout, IDs, answer,
reasoning, prompt, metadata, counts, input identity, and JSON round-trip. It
uses `qwen3.5:4b` with a 1,536-token budget, a 60-second completion deadline,
and a 90-second test timeout. Missing Ollama or its model fails rather than
skips. The 60-second deadline is not
generous next to a real completion on a busy laptop, so under load the test can
fail with a `TimeoutError` generation failure written to `failed.jsonl`; that is
the deadline, not a broken contract.
The test verifies the pipeline contract, not biomedical accuracy. No extra live
queries exercise rejection paths.

Unit tests use Pydantic AI's `FunctionModel` at the inference boundary and real
temporary files. They cover failures, incremental writes, interruption, custom
gates, verbose output, fetching, and configuration without contacting a model
provider. The judge gate is covered at the settings level only; no offline test
exercises its prompt.

Pre-commit runs Gitleaks, Ruff lint/format, basedpyright, and ty. Explicit `Any`
is banned across source and tests by basedpyright's
[`reportExplicitAny`](https://docs.basedpyright.com/latest/benefits-over-pyright/new-diagnostic-rules/#reportexplicitany)
and Ruff's banned imports. Third-party inferred types are not globally banned;
our data boundaries validate them into concrete models.

The `data/` directory and `.env` are ignored by Git. To scan Git history before
publishing, install Gitleaks and run `gitleaks git --redact`.
