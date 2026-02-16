from fluxopt.components import Converter, Port
from fluxopt.elements import Bus, Effect, Flow, Sizing, Storage
from fluxopt.model import FlowSystemModel
from fluxopt.model_data import ModelData
from fluxopt.results import SolvedModel
from fluxopt.types import (
    IdList,
    TimeSeries,
    Timesteps,
    as_dataarray,
    compute_dt,
    normalize_timesteps,
)


def solve(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    solver: str = 'highs',
    silent: bool = True,
) -> SolvedModel:
    """Build data, build model, solve, return results.

    Args:
        timesteps: Time index for the optimization horizon.
        buses: Energy buses in the system.
        effects: Effects to track (costs, emissions, etc.).
        ports: System boundary ports with imports/exports.
        converters: Linear converters between buses.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        solver: Solver backend name.
        silent: Suppress solver output.
    """
    data = ModelData.build(timesteps, buses, effects, ports, converters, storages, dt)
    model = FlowSystemModel(data, solver=solver)
    model.build()
    return model.solve(silent=silent)


__all__ = [
    'Bus',
    'Converter',
    'Effect',
    'Flow',
    'FlowSystemModel',
    'IdList',
    'ModelData',
    'Port',
    'Sizing',
    'SolvedModel',
    'Storage',
    'TimeSeries',
    'Timesteps',
    'as_dataarray',
    'compute_dt',
    'normalize_timesteps',
    'solve',
]
