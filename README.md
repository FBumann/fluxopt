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

```python
from datetime import datetime, timedelta

import fluxopt as fx

timesteps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(4)]

result = fx.optimize(
    timesteps=timesteps,
    buses=[fx.Bus('electricity')],
    effects=[fx.Effect('cost', is_objective=True)],
    ports=[
        fx.Port('grid', imports=[
            fx.Flow(bus='electricity', size=200, effects_per_flow_hour={'cost': 0.04}),
        ]),
        fx.Port('demand', exports=[
            fx.Flow(bus='electricity', size=100, fixed_relative_profile=[0.5, 0.8, 1.0, 0.6]),
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
