# CLAUDE.md — Project Guide for fluxopt

## Philosophy

This project uses **modern Python tooling exclusively**. Every tool in the chain is
fast, standards-based, and configured in a single `pyproject.toml`.

- **uv is the single entry point** for all Python operations — installing, running,
  locking, building, adding/removing deps. No pip, no setuptools CLI, no tox.
- **pyproject.toml is the single source of truth** — build config, dependencies,
  linter settings, test settings. No setup.py, setup.cfg, tox.ini, or .flake8.
- **src layout** — source code lives in `src/fluxopt/`, enforcing proper installation
  for testing and preventing accidental imports from the working directory.
- **hatchling + hatch-vcs** for building — version comes from git tags automatically.
- **PEP 735 dependency groups** for contributor tooling (dev, docs). These are *not*
  installable by end users and won't leak into the published package.
- **ruff** replaces flake8, isort, pyupgrade, and black — one tool, zero config files.
- **The lock file is committed** — reproducible CI and contributor environments.
- **Never hand-edit dependency versions** — always use `uv add` / `uv remove` so the
  lock file stays in sync.

## Tooling

| Role | Tool |
|------|------|
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Build backend | hatchling + hatch-vcs |
| Linter + formatter | ruff |
| Tests | pytest |
| Docs | mkdocs-material |
| Pre-commit | ruff + pre-commit-hooks + nbstripout |

## Common Commands

```bash
uv sync                  # Install runtime deps
uv sync --group dev      # Install runtime + dev deps
uv sync --group docs     # Install runtime + docs deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv build                 # Build wheel + sdist
uv lock --check          # Verify lockfile is up to date
uv add <pkg>             # Add runtime dependency
uv add --group dev <pkg> # Add dev dependency
uv add --optional full <pkg>  # Add to [full] extra
uv remove <pkg>          # Remove a dependency
```

## Package Layout

- **PyPI name**: `fluxopt`
- **Import name**: `fluxopt`
- **Source**: `src/fluxopt/` (src layout)
- **Tests**: `tests/`
- **Docs**: `docs/` + `mkdocs.yml`

## Dependency Management

- **Runtime deps**: `[project].dependencies` — add via `uv add <pkg>`
- **Dev/docs groups**: `[dependency-groups]` (PEP 735) — add via `uv add --group <name> <pkg>`
- **User extras**: `[project.optional-dependencies]` — add via `uv add --optional <name> <pkg>`
- **Lock file**: `uv.lock` is committed — never hand-edit, always use `uv add` / `uv remove`
- Python >= 3.12 required
