# Agent Working Agreement

## Definition of done

Work is **not done**, and must not be handed back, until both of these pass:

```sh
uv run pytest
uv run pre-commit run --all-files
```

Both must be run *after your final edit*. Passing pre-commit is part of the
definition of done, not a formality on the way out — a green test run is not
evidence that pre-commit passes, because the hooks also run Gitleaks, Ruff lint
and format, basedpyright and ty, none of which pytest exercises.

If you cannot get them green, say so plainly and stop. Never describe work as
complete, finished, or ready while a check is failing or unrun.

## Verifying the checks honestly

- **Judge the result by the exit code, not by the output you happened to read.**
  Do not pipe these commands through `tail`, `head` or `grep` and conclude they
  passed because the visible lines looked fine. Failures appear mid-output and
  are trivially truncated away.
- Report what the commands actually printed. If you did not run one, say you did
  not run it.

## While working

- Run both checks frequently — after each meaningful change, not once at the end.
- Type errors are cheap to fix in the slice that introduced them and expensive to
  fix as a batch afterwards, because later work gets built on top of a signature
  the type checker was going to reject anyway.

## Commits

- Do not commit changes. Leave commits for the user.

## Configuration

- Collect runtime configuration and defaults in `src/settings.py`. Do not scatter
  tunable values (paths, model parameters, timeouts, retries, dataset limits, or
  output verbosity) through implementation code or duplicate them in argparse.
- Expose experiment and provider options through the validated settings models
  so the effective configuration is captured in run metadata. CLI arguments
  should override settings only when explicitly supplied.
