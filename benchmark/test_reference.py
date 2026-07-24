"""Benchmarks over the bundled realistic reference systems (``fluxopt.benchmark``).

Unlike the archetypes in ``systems.py`` (which live in this directory and work
against any fluxopt version), these come from the installed package — so
comparing two branches is just running the suite on each (``uv run`` re-syncs
the editable install after a ``git switch``; see README.md, which also covers
``benchmem sweep`` for released versions).

Versions that predate ``fluxopt.benchmark`` skip this file (importorskip).
A quarter year keeps a multi-round pytest-benchmark run reasonable; the
full-year numbers come from ``python -m fluxopt.benchmark``.
"""

from __future__ import annotations

import pytest

fx_benchmark = pytest.importorskip('fluxopt.benchmark')

QUARTER_YEAR = 2190


@pytest.fixture(params=['district_heating', 'industry_park', 'green_city', 'energy_transition', 'stress'])
def reference_system(request: pytest.FixtureRequest) -> str:
    name = request.param
    if name not in fx_benchmark.SYSTEMS:
        pytest.skip(f'{name!r} is not in this fluxopt version')
    return name


def test_reference_build(benchmark: object, reference_system: str) -> None:
    """Full pipeline (Elements -> ModelData -> linopy model) for one reference system."""
    row = benchmark(fx_benchmark.measure, reference_system, QUARTER_YEAR)  # type: ignore[operator]
    extra_info = getattr(benchmark, 'extra_info', None)
    if extra_info is not None and isinstance(row, dict):
        extra_info['variables'] = row['variables']
        extra_info['constraints'] = row['constraints']
