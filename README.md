# fluxopt

Energy system optimization with [linopy](https://github.com/PyPSA/linopy) — detailed dispatch, scaled to multi period planning.

[![PyPI](https://img.shields.io/pypi/v/fluxopt)](https://pypi.org/project/fluxopt/)
[![Downloads](https://img.shields.io/pypi/dm/fluxopt)](https://pypi.org/project/fluxopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Early development** — the API may change between releases.
> Planned features and progress are tracked in [Issues](https://github.com/FBumann/fluxopt/issues).

## Installation

```bash
pip install fluxopt
```

Includes the [HiGHS](https://highs.dev/) solver out of the box.

## Quick Start

<!--quickstart-start-->
```python
# A gas boiler covers a heat demand, minimizing fuel cost
from datetime import datetime
from fluxopt import Carrier, Converter, Effect, Flow, Port, optimize

result = optimize(
    timesteps=[datetime(2024, 1, 1, h) for h in range(4)],
    carriers=[Carrier(id='gas'), Carrier(id='heat')],
    effects=[Effect(id='cost')],
    ports=[
        Port(id='grid', imports=[Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': 0.04})]),
        Port(id='demand', exports=[Flow(carrier='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])]),
    ],
    converters=[
        Converter.boiler(
            'boiler',
            thermal_efficiency=0.9,
            fuel_flow=Flow(carrier='gas', size=300),
            thermal_flow=Flow(carrier='heat', size=200),
        )
    ],
    objective='cost',
)

print(f'Total cost: {result.objective:.2f}')
print(result.flow_rates)
```
<!--quickstart-end-->

## One API, four levels of control

Every level returns the same `Result`; each one only adds control — pick the
lowest rung that does the job.

**1. One-shot** — `optimize(...)` as above. Elements in, `Result` out, with
fail-fast validation of ids and references.

**2. Declarative** — gather the same arguments into a reusable, serializable
system. Time series can stay out of the structure as `ProfileRef`s and be
supplied at solve time via `profiles`:

```python
spec = fx.FlowSystem.from_yaml('system.yaml')  # or FlowSystem(...) in Python
result = spec.optimize(profiles={'load': demand_ds})
spec.to_yaml('system.yaml')  # round-trips
```

**3. Inspectable** — materialize the solver model without solving, inspect or
extend the underlying linopy model, retarget the objective, then solve:

```python
model = spec.build_model(profiles={'load': demand_ds})  # unbuilt FlowSystemModel
model.build()
model.m.add_constraints(...)  # full linopy access
result = model.solve()

model.objective = {'cost': 1, 'co2': 50}  # retarget…
model.build()  # …and rebuild
```

For a one-off tweak, stay on level 1/2 and pass
`customize=lambda m: m.m.add_constraints(...)` instead.

**4. Data-level** — build or load the xarray `ModelData` yourself and edit it
before modeling:

```python
data = fx.ModelData.build(...)  # or ModelData.from_netcdf(path)
data.flows.fixed_profile.loc[{'flow': 'demand(heat)'}] = 0.7
result = fx.FlowSystemModel(data, objective='cost').optimize()
```

Results close the loop: `result.flow_rates`, `result.effect_totals`,
`result.stats` (KPIs, effect contributions), `result.plot`, netCDF round-trip,
and `result.data` — the exact `ModelData` the solution came from.

## Roadmap

fluxopt is evolving into a family of packages with a lean core and optional companions:

```
                          ┌──────────────┐
                          │   fluxopt    │  core: model building, solving, results, IO
                          └──────┬───────┘
        ┌──────────────┬─────────┼──────────────┬──────────────┐
        │              │         │              │              │
 fluxopt-plot   fluxopt-yaml  fluxopt-tsam  fluxopt-marimo  (examples)
   plotting      YAML+CSV    time series    interactive     cross-package
   (plotly)       loader     aggregation       apps          notebooks
```

Companion packages depend on core — core has no knowledge of companions.

### Companion packages

| Package | Role | Versioning · Tier | `fluxopt` pin | Status |
|---------|------|-------------------|---------------|--------|
| `fluxopt-plot` | Result visualization (Plotly) | Semver · Experimental — method signatures may change | Tight (`>=A.B,<A.C`), validated per release | Scaffolded — [docs](https://fbumann.github.io/fluxopt-plot/latest/) · [#51](https://github.com/FBumann/fluxopt/issues/51) |
| `fluxopt-yaml` | Declarative model loader (YAML + CSV → `Element`s) | Semver · Experimental — YAML schema may change | Tight (`>=A.B,<A.C`), validated per release | Scaffolded — [docs](https://fbumann.github.io/fluxopt-yaml/latest/) · [#52](https://github.com/FBumann/fluxopt/issues/52) |
| `fluxopt-tsam` | Time series aggregation — input pre-processing, possibly result disaggregation | Semver · Experimental — round-trip schema may evolve | **Undecided** — depends on whether representative-period primitives live in core (→ loose) or in this package (→ tight) | Planned |
| `fluxopt-marimo` | Interactive exploration & dashboards (marimo apps) | CalVer (`YYYY.MM.PATCH`) · Experimental — apps are templates | Tight (`>=A.B,<A.C`), validated per release | Planned |

Tight-pinned companions release on every `fluxopt` minor; validation is
automated via scheduled CI. `fluxopt-tsam`'s pin policy is blocked on an
architectural decision — if representative-period primitives live in core, tsam stays
a thin adapter (loose pin); if they live in tsam, the package owns deep
round-trip behavior (tight pin).

### Milestones

Cross-cutting work not tied to a single companion package:

| Milestone | Description | Status | Issue |
|-----------|-------------|--------|-------|
| `Result.stats` accessor | Cached xarray properties for post-processing | Planned | [#49](https://github.com/FBumann/fluxopt/issues/49) |
| `.plot` stub on `Result` | Discoverable property, helpful error if plot package absent | Planned | [#50](https://github.com/FBumann/fluxopt/issues/50) |
| ReadTheDocs migration | Automatic versioned docs from git tags | Planned | [#53](https://github.com/FBumann/fluxopt/issues/53) |
| Remove plotly from core | Keep core lean — plotting deps in `fluxopt-plot` only | Planned | [#54](https://github.com/FBumann/fluxopt/issues/54) |

### Stability Tiers

| Component | Tier | Policy |
|-----------|------|--------|
| Core modeling API | **Stable** | Semver. Deprecation warnings before removal. |
| Stats accessor | **Semi-stable** | Breaking changes allowed between minor versions with changelog entry. |

Companion packages have their own stability policies — see the table above.

See [#47](https://github.com/FBumann/fluxopt/issues/47) for the full architecture discussion.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```bash
uv sync --group dev      # Install deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
```

## License

MIT
