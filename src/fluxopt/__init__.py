import polars as pl

from fluxopt.components import LinearConverter, Port
from fluxopt.elements import Bus, Effect, Flow, Storage
from fluxopt.model import FlowSystemModel
from fluxopt.results import SolvedModel
from fluxopt.tables import (
    BusesTable,
    ConvertersTable,
    EffectsTable,
    FlowsTable,
    ModelData,
    StoragesTable,
    build_model_data,
)
from fluxopt.types import TimeSeries, Timesteps, compute_dt, normalize_timesteps, to_polars_series


def solve(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    components: list[Port | LinearConverter],
    storages: list[Storage] | None = None,
    dt: float | list[float] | pl.Series | None = None,
    solver: str = 'highs',
    silent: bool = True,
) -> SolvedModel:
    """Convenience: build data, build model, solve, return results."""
    data = build_model_data(timesteps, buses, effects, components, storages, dt)
    model = FlowSystemModel(data, solver=solver)
    model.build()
    return model.solve(silent=silent)


__all__ = [
    'Bus',
    'BusesTable',
    'ConvertersTable',
    'Effect',
    'EffectsTable',
    'Flow',
    'FlowSystemModel',
    'FlowsTable',
    'LinearConverter',
    'ModelData',
    'Port',
    'SolvedModel',
    'Storage',
    'StoragesTable',
    'TimeSeries',
    'Timesteps',
    'build_model_data',
    'compute_dt',
    'normalize_timesteps',
    'solve',
    'to_polars_series',
]
