"""Command-line entry point for a reasoning-trace experiment."""

import argparse
import asyncio

from pydantic import ValidationError

from src.experiments import run_experiment
from src.settings import RunSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", help="Cached Q&A JSON; fetched if missing.")
    parser.add_argument("--runs-dir", help="Parent directory for pet-named runs.")
    parser.add_argument(
        "--question-limit", type=int, help="Maximum questions to process."
    )
    parser.add_argument("--completions-per-question", type=int)
    parser.add_argument("--temperature", type=float)
    arguments = vars(parser.parse_args())
    try:
        settings = RunSettings(
            **{key: value for key, value in arguments.items() if value is not None}
        )
        directory = asyncio.run(run_experiment(settings))
    except (ValueError, RuntimeError, OSError, ValidationError) as error:
        parser.exit(1, f"Experiment failed: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Experiment interrupted; completed records remain on disk.\n")
    else:
        print(f"Run saved to {directory}")


if __name__ == "__main__":
    main()
