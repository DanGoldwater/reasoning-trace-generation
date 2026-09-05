# reasoning-trace-generation

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
basedpyright type checking. Ruff and basedpyright are configured in
`pyproject.toml`.

To scan the full Git history (for example, before publishing the repository),
install the Gitleaks CLI and run:

```sh
gitleaks git --redact
```
