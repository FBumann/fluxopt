"""Shared helpers for ported mathematical correctness tests.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``optimize``
    Baseline correctness check.
``save->reload->optimize``
    Proves the ModelData definition survives IO.
``optimize->save->reload->validate``
    Proves the solution data survives IO and contributions sum correctly.
"""

from __future__ import annotations

import pytest
from conftest import ts, waste  # noqa: F401 — re-exported for test imports

from fluxopt import FlowSystem, ModelData
from fluxopt import optimize as fluxopt_optimize
from fluxopt.results import Result


@pytest.fixture(
    params=[
        'optimize',
        'save->reload->optimize',
        'optimize->save->reload->validate',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs) -> Result:
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
            model = FlowSystem(loaded)
            model.build()
            return model.solve()
        # optimize->save->reload->validate
        result = fluxopt_optimize(**kwargs)
        path = tmp_path / 'result.nc'
        result.to_netcdf(path)
        loaded = Result.from_netcdf(path)
        loaded.effect_contributions()  # validate contributions survive IO roundtrip
        return loaded

    return _optimize
