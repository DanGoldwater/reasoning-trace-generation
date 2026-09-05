set shell := ["bash", "-euo", "pipefail", "-c"]

MARP := "@marp-team/marp-cli@4.5.0"
EXCALIDRAW := "@moona3k/excalidraw-export@0.2.1"
NPX := "npx --yes --registry=https://registry.npmjs.org"

# List the available recipes.
default:
    just --list

# Run the test suite.
test:
    uv run pytest

# Run the full definition-of-done checks.
check: test
    uv run pre-commit run --all-files

# Regenerate the figures used by the EDA report.
plots:
    uv run python docs/make_plots.py

# Re-export the source diagrams and build a self-contained HTML slide deck.
[working-directory('docs')]
slides:
    for drawing in figures/*.excalidraw; do {{ NPX }} {{ EXCALIDRAW }} "$drawing" --svg --output "${drawing%.excalidraw}.svg"; done
    {{ NPX }} {{ MARP }} slides.md --html --output slides.html
