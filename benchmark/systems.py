"""Synthetic systems for the build-pipeline benchmarks.

A registry of feature archetypes — each compiles to a different set of
variables/constraints, so each has its own time/memory fingerprint. Deterministic
(no randomness, no wall-clock), so ``benchmem compare`` diffs are meaningful.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from fluxopt import Carrier, Converter, Effect, Flow, ModelData, PiecewiseConversion, Port, Sizing, Status, Storage

if TYPE_CHECKING:
    from collections.abc import Callable

# (n, timesteps): n scales component count, timesteps the horizon. Bump locally
# to profile heavier models; kept modest to stay under a couple of CI minutes.
SCALES: dict[str, dict[str, int]] = {
    'small': {'n': 4, 'timesteps': 24},
    'medium': {'n': 15, 'timesteps': 96},
    'large': {'n': 40, 'timesteps': 336},
}
SCENARIO_SCALE = SCALES['medium']

Elements = dict[str, object]


def _timesteps(n: int) -> list[datetime]:
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def _profile(n: int) -> list[float]:
    return (0.55 + 0.35 * np.sin(np.arange(n) / 3.0)).clip(0.1, 1.0).tolist()


def multi_node(*, n: int, timesteps: int) -> Elements:
    """Gas/elec/heat buses, boilers + CHPs + heat pump + storage."""
    converters: list[Converter] = []
    for i in range(n):
        if i % 3 == 0:
            converters.append(
                Converter.chp(
                    f'chp{i}',
                    0.4,
                    0.45,
                    Flow(carrier='gas', size=300.0, effects_per_flow_hour={'cost': 0.03, 'co2': 0.2}),
                    Flow(carrier='elec', size=150.0),
                    Flow(carrier='heat', size=150.0),
                )
            )
        else:
            converters.append(
                Converter.boiler(
                    f'boiler{i}',
                    0.9,
                    Flow(carrier='gas', size=300.0, effects_per_flow_hour={'cost': 0.03, 'co2': 0.2}),
                    Flow(carrier='heat', size=200.0),
                )
            )
    converters.append(
        Converter.heat_pump(
            'hp',
            3.5,
            Flow(carrier='elec', size=100.0, effects_per_flow_hour={'cost': 0.08}),
            Flow(carrier='heat', short_id='ambient', size=1e6),
            Flow(carrier='heat', size=250.0),
        )
    )
    return {
        'timesteps': _timesteps(timesteps),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='heat')],
        'effects': [Effect(id='cost'), Effect(id='co2', unit='kg')],
        'ports': [
            Port(
                id='gas_grid', imports=[Flow(carrier='gas', size=1e6, effects_per_flow_hour={'cost': 0.03, 'co2': 0.2})]
            ),
            Port(id='elec_grid', imports=[Flow(carrier='elec', size=1e6, effects_per_flow_hour={'cost': 0.08})]),
            Port(id='elec_sink', exports=[Flow(carrier='elec', size=1e6)]),
            Port(
                id='heat_demand',
                exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))],
            ),
        ],
        'converters': converters,
        'storages': [
            Storage(
                id='heat_store',
                charging=Flow(carrier='heat', size=200.0),
                discharging=Flow(carrier='heat', size=200.0),
                capacity=1000.0,
            ),
        ],
    }


def status(*, n: int, timesteps: int) -> Elements:
    """Semi-continuous sources with on/off, min-uptime, and startup effects."""
    sources = [
        Flow(
            carrier='heat',
            size=100.0,
            relative_rate_min=0.4,
            effects_per_flow_hour={'cost': 1.0 + 0.1 * i},
            status=Status(uptime_min=3, effects_per_startup={'cost': 20.0}, effects_per_running_hour={'cost': 0.5}),
        )
        for i in range(n)
    ]
    return {
        'timesteps': _timesteps(timesteps),
        'carriers': [Carrier(id='heat')],
        'effects': [Effect(id='cost')],
        'ports': [
            Port(
                id='demand', exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))]
            ),
            *[Port(id=f'src{i}', imports=[f]) for i, f in enumerate(sources)],
            Port(id='backup', imports=[Flow(carrier='heat', size=1e6, effects_per_flow_hour={'cost': 50.0})]),
        ],
    }


def piecewise(*, n: int, timesteps: int) -> Elements:
    """Converters with piecewise-linear part-load curves."""
    converters = [
        Converter(
            id=f'boilerP{i}',
            inputs=[Flow(carrier='gas', short_id='fuel', effects_per_flow_hour={'cost': 1.0})],
            outputs=[Flow(carrier='heat', size=100.0)],
            conversion=PiecewiseConversion(points={'fuel': [0, 30, 60, 100], 'heat': [0, 30, 54, 70]}),
        )
        for i in range(n)
    ]
    return {
        'timesteps': _timesteps(timesteps),
        'carriers': [Carrier(id='gas'), Carrier(id='heat')],
        'effects': [Effect(id='cost')],
        'ports': [
            Port(id='gas_grid', imports=[Flow(carrier='gas', size=1e6, effects_per_flow_hour={'cost': 1.0})]),
            Port(
                id='demand', exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))]
            ),
        ],
        'converters': converters,
    }


def effects(*, n: int, timesteps: int) -> Elements:
    """Many effect families per flow, plus sizing effects."""
    names = ['cost', 'co2', 'nox', 'primary', 'land', 'water', 'noise', 'sox']
    per_hour = {k: 0.01 * (j + 1) for j, k in enumerate(names)}
    per_size = {k: 0.5 * (j + 1) for j, k in enumerate(names[:4])}
    fixed = {k: 5.0 * (j + 1) for j, k in enumerate(names[:4])}
    sources = [
        Flow(
            carrier='heat',
            size=Sizing(
                size_min=10,
                size_max=500,
                mandatory=(i % 2 == 0),
                effects_per_size=dict(per_size),
                effects_fixed=dict(fixed),
            ),
            effects_per_flow_hour=dict(per_hour),
        )
        for i in range(n)
    ]
    return {
        'timesteps': _timesteps(timesteps),
        'carriers': [Carrier(id='heat')],
        'effects': [Effect(id='cost'), *[Effect(id=k, unit='kg') for k in names[1:]]],
        'ports': [
            Port(
                id='demand', exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))]
            ),
            *[Port(id=f'src{i}', imports=[f]) for i, f in enumerate(sources)],
        ],
    }


def sizing(*, n: int, timesteps: int) -> Elements:
    """Investable flows with mandatory + optional Sizing (capacity + indicators)."""
    sources = [
        Flow(
            carrier='heat',
            size=Sizing(
                size_min=10,
                size_max=300,
                mandatory=(i % 2 == 0),
                effects_per_size={'cost': 100.0},
                effects_fixed={'cost': 50.0},
            ),
            effects_per_flow_hour={'cost': 1.0 + 0.05 * i},
        )
        for i in range(n)
    ]
    return {
        'timesteps': _timesteps(timesteps),
        'carriers': [Carrier(id='heat')],
        'effects': [Effect(id='cost')],
        'ports': [
            Port(
                id='demand', exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))]
            ),
            *[Port(id=f'src{i}', imports=[f]) for i, f in enumerate(sources)],
        ],
    }


def multi_period(*, n: int, timesteps: int) -> Elements:
    """Four investment periods over the features whose build cost is period-scoped.

    Exercises every code path the time-index representation prices differently:
    per-period aggregates (``flow_hours_max``, Effect ``periodic_max``),
    episode-reset chains (storage cyclic/prior, status windows), and per-period
    ``Sizing`` with size-scoped effects.
    """
    periods = [2030 + 5 * p for p in range(4)]
    sources = [
        Flow(
            carrier='heat',
            size=Sizing(size_min=10, size_max=300, mandatory=(i % 2 == 0), effects_per_size={'cost': 100.0}),
            effects_per_flow_hour={'cost': 1.0 + 0.05 * i, 'co2': 0.2},
            flow_hours_max=5000.0,
        )
        for i in range(n)
    ]
    status_sources = [
        Flow(
            carrier='heat',
            size=100.0,
            relative_rate_min=0.4,
            effects_per_flow_hour={'cost': 2.0 + 0.1 * i},
            status=Status(uptime_min=3, effects_per_startup={'cost': 20.0}),
        )
        for i in range(max(n // 2, 1))
    ]
    return {
        'timesteps': _timesteps(timesteps),
        'periods': periods,
        'period_weights': [5.0] * len(periods),
        'carriers': [Carrier(id='heat')],
        'effects': [Effect(id='cost'), Effect(id='co2', unit='kg', periodic_max=1e9)],
        'ports': [
            Port(
                id='demand', exports=[Flow(carrier='heat', size=100.0 * n, fixed_relative_profile=_profile(timesteps))]
            ),
            *[Port(id=f'src{i}', imports=[f]) for i, f in enumerate(sources)],
            *[Port(id=f'unit{i}', imports=[f]) for i, f in enumerate(status_sources)],
            Port(id='backup', imports=[Flow(carrier='heat', size=1e6, effects_per_flow_hour={'cost': 50.0})]),
        ],
        'storages': [
            Storage(
                id='heat_store',
                charging=Flow(carrier='heat', size=200.0),
                discharging=Flow(carrier='heat', size=200.0),
                capacity=1000.0,
                prior_level=500.0,
            ),
        ],
    }


SCENARIOS: dict[str, Callable[..., Elements]] = {
    'multi_node': multi_node,
    'status': status,
    'piecewise': piecewise,
    'effects': effects,
    'sizing': sizing,
    'multi_period': multi_period,
}


def make_model_data(builder: Callable[..., Elements], **scale: int) -> ModelData:
    """Elements → ModelData."""
    return ModelData.build(**builder(**scale))


def build_model(data: ModelData, objective: str = 'cost') -> Any:
    """ModelData → a built model, without solving (mirrors optimize() before solve)."""
    import lpspec

    from fluxopt.relational import MATH_PROGRAM, build_sources
    from fluxopt.relational.results import objective_weights

    sources, coords = build_sources(data, objective_weights(data, objective))
    return lpspec.build(MATH_PROGRAM, {**sources, **coords})
