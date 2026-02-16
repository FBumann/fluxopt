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
- **mypy strict** — enforced from day one, runs in CI and pre-commit.
- **Dev/docs deps are pinned** (`==`) — Dependabot bumps them weekly.
- **No lock file** — `uv.lock` is gitignored. Pinned deps in pyproject.toml are the source of truth for dev tooling.

## Tooling

| Role | Tool |
|------|------|
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Build backend | hatchling + hatch-vcs |
| Linter + formatter | ruff |
| Tests | pytest |
| Docs | mkdocs-material |
| Type checker | mypy (strict) |
| Pre-commit | ruff + mypy + pre-commit-hooks + nbstripout |

## Common Commands

```bash
uv sync                  # Install runtime deps
uv sync --group dev      # Install runtime + dev deps
uv sync --group docs     # Install runtime + docs deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run mypy src/         # Type check
uv build                 # Build wheel + sdist
uv add <pkg>             # Add runtime dependency (uses >=)
uv add --group dev <pkg> # Add dev dependency (pin to == manually)
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
- **Dev/docs groups**: `[dependency-groups]` (PEP 735) — pinned with `==`, Dependabot bumps weekly
- **User extras**: `[project.optional-dependencies]` — add via `uv add --optional <name> <pkg>`
- Python >= 3.12 required

## Code Style

- **Docstrings**: Google style, brief, on all functions
  - No types in docstrings (types live in signatures only)
  - Always include `Args` section when there are parameters
  - `Returns` / `Raises` only when non-obvious
