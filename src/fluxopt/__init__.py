from collections.abc import Callable
from typing import Any

from fluxopt.components import Converter, Port
from fluxopt.elements import Bus, Effect, Flow, Sizing, Status, Storage
from fluxopt.model import FlowSystem
from fluxopt.model_data import ModelData
from fluxopt.results import Result
from fluxopt.types import (
    IdList,
    TimeSeries,
    Timesteps,
    as_dataarray,
    compute_dt,
    normalize_timesteps,
)
from fluxopt.yaml_loader import load_yaml, solve_yaml


def optimize(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    solver: str = 'highs',
    customize: Callable[[FlowSystem], None] | None = None,
    **kwargs: Any,
) -> Result:
    """Build data, build model, optimize, return results.

    Args:
        timesteps: Time index for the optimization horizon.
        buses: Energy buses in the system.
        effects: Effects to track (costs, emissions, etc.).
        ports: System boundary ports with imports/exports.
        converters: Linear converters between buses.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        solver: Solver backend name.
        customize: Optional callback to modify the linopy model between build and solve.
            Receives the built FlowSystem; use ``model.m`` to add variables/constraints.
        **kwargs: Passed through to ``linopy.Model.solve()``.
    """
    data = ModelData.build(timesteps, buses, effects, ports, converters, storages, dt)
    model = FlowSystem(data)
    return model.optimize(customize=customize, solver=solver, **kwargs)


__all__ = [
    'Bus',
    'Converter',
    'Effect',
    'Flow',
    'FlowSystem',
    'IdList',
    'ModelData',
    'Port',
    'Result',
    'Sizing',
    'Status',
    'Storage',
    'TimeSeries',
    'Timesteps',
    'as_dataarray',
    'compute_dt',
    'load_yaml',
    'normalize_timesteps',
    'optimize',
    'solve_yaml',
]
