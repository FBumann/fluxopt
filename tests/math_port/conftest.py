"""Shared helpers for ported mathematical correctness tests.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``solve``
    Baseline correctness check.
``save->reload->solve``
    Proves the ModelData definition survives IO.
``solve->save->reload``
    Proves the solution data survives IO.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fluxopt import Flow, FlowSystemModel, ModelData, Port, solve
from fluxopt.results import SolvedModel


def ts(n: int) -> list[datetime]:
    """Create n hourly timesteps starting 2024-01-01."""
    return [datetime(2024, 1, 1, h) for h in range(n)]


def _waste(bus: str) -> Port:
    """Create a dump port that absorbs excess on a bus for free."""
    return Port(f'_dump_{bus}', exports=[Flow(bus=bus)])


@pytest.fixture(
    params=[
        'solve',
        'save->reload->solve',
        'solve->save->reload',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs) -> SolvedModel:
        if request.param == 'solve':
            return solve(**kwargs)
        if request.param == 'save->reload->solve':
            data = ModelData.build(
                kwargs['timesteps'],
                kwargs['buses'],
                kwargs['effects'],
                kwargs['ports'],
                kwargs.get('converters'),
                kwargs.get('storages'),
                kwargs.get('dt'),
            )
            path = tmp_path / 'data.nc'
            data.to_netcdf(path, mode='w')
            loaded = ModelData.from_netcdf(path)
            model = FlowSystemModel(loaded, solver='highs')
            model.build()
            return model.solve(silent=True)
        # solve->save->reload
        result = solve(**kwargs)
        path = tmp_path / 'result.nc'
        result.to_netcdf(path)
        return SolvedModel.from_netcdf(path)

    return _optimize
