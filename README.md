# fluxopt

Energy system optimization built on [pyoframe](https://github.com/Bravos-Power/pyoframe) and [polars](https://pola.rs/).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Installation

```bash
pip install fluxopt
```

Includes the [HiGHS](https://highs.dev/) solver out of the box.

## Quick Start

```python
import energysys as es

result = es.solve(
    timesteps=["t0", "t1", "t2"],
    buses=[es.Bus("electricity")],
    effects=[es.Effect("cost", is_objective=True)],
    components=[
        es.Source("grid", outputs=[
            es.Flow("elec", bus="electricity", size=200, effects_per_flow_hour={"cost": 0.04}),
        ]),
        es.Sink("demand", inputs=[
            es.Flow("elec", bus="electricity", size=100, fixed_relative_profile=[0.5, 0.8, 0.6]),
        ]),
    ],
)
```

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
