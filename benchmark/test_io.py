"""NetCDF IO benchmarks — ModelData serialization round-trip.

Mirrors ``test_build.py``: the same feature matrix at one scale plus a
multi_node scaling curve, but timing/measuring ``ModelData.save`` (write)
and ``ModelData.load`` (read) instead of the build. Solve-free and
deterministic, so it fits the same CodSpeed / benchmem suite.

The solved-``Result`` round-trip is intentionally *not* here: building a Result
needs a HiGHS solve, which the suite excludes (non-deterministic, not ours).

Run from this pinned, standalone project (``cd benchmark``)::

    uv run pytest . --codspeed                        # CodSpeed
    uv run pytest . --benchmark-only --benchmark-memory  # local memory (benchmem)

See benchmark/README.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from systems import SCALES, SCENARIO_SCALE, SCENARIOS, make_model_data, multi_node

from fluxopt import ModelData

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=list(SCENARIOS), ids=list(SCENARIOS))
def scenario(request: pytest.FixtureRequest) -> object:
    return SCENARIOS[request.param]


@pytest.fixture(params=list(SCALES), ids=list(SCALES))
def scale(request: pytest.FixtureRequest) -> dict:
    return SCALES[request.param]


def test_feature_write(benchmark: object, scenario: object, tmp_path: Path) -> None:
    """ModelData → NetCDF, per feature archetype."""
    data = make_model_data(scenario, **SCENARIO_SCALE)  # type: ignore[arg-type]
    path = tmp_path / 'model'
    benchmark(data.save, path)  # type: ignore[operator]


def test_feature_read(benchmark: object, scenario: object, tmp_path: Path) -> None:
    """NetCDF → ModelData, per feature archetype."""
    data = make_model_data(scenario, **SCENARIO_SCALE)  # type: ignore[arg-type]
    path = tmp_path / 'model'
    data.save(path)
    benchmark(ModelData.load, path)  # type: ignore[operator]


def test_scaling_write(benchmark: object, scale: dict, tmp_path: Path) -> None:
    """ModelData → NetCDF for multi_node across scales."""
    data = make_model_data(multi_node, **scale)
    path = tmp_path / 'model'
    benchmark(data.save, path)  # type: ignore[operator]


def test_scaling_read(benchmark: object, scale: dict, tmp_path: Path) -> None:
    """NetCDF → ModelData for multi_node across scales."""
    data = make_model_data(multi_node, **scale)
    path = tmp_path / 'model'
    data.save(path)
    benchmark(ModelData.load, path)  # type: ignore[operator]
