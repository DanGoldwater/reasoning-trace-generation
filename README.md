# reasoning-trace-generation

## Dataset access

Create a local `.env` file containing a Hugging Face token that can read the
private dataset:

```sh
cp .env.example .env
```

Set `HF_TOKEN` in that file, then fetch the sample data with:

```sh
uv run python src/data_fetching.py
```

The first 50 records are saved to `./data/private_qa.json`. The `data/`
directory is intentionally ignored by Git because it may contain private data.

Environment variables already set in the shell take precedence over `.env`.

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
