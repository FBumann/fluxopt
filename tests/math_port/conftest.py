"""Shared helpers for ported mathematical correctness tests.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``optimize``
    Baseline correctness check.
``save->reload->optimize``
    Proves the ModelData definition survives IO.
``optimize->save->reload``
    Proves the solution data survives IO.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fluxopt import Flow, FlowSystemModel, ModelData, Port
from fluxopt import optimize as fluxopt_optimize
from fluxopt.results import SolvedModel


def ts(n: int) -> list[datetime]:
    """Create n hourly timesteps starting 2024-01-01."""
    return [datetime(2024, 1, 1, h) for h in range(n)]


def _waste(bus: str) -> Port:
    """Create a dump port that absorbs excess on a bus for free."""
    return Port(f'_dump_{bus}', exports=[Flow(bus=bus)])


@pytest.fixture(
    params=[
        'optimize',
        'save->reload->optimize',
        'optimize->save->reload',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs) -> SolvedModel:
        if request.param == 'optimize':
            return fluxopt_optimize(**kwargs)
        if request.param == 'save->reload->optimize':
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
            model = FlowSystemModel(loaded)
            model.build()
            return model.solve()
        # optimize->save->reload
        result = fluxopt_optimize(**kwargs)
        path = tmp_path / 'result.nc'
        result.to_netcdf(path)
        return SolvedModel.from_netcdf(path)

    return _optimize
