import polars as pl

from fluxopt.components import LinearConverter, Sink, Source
from fluxopt.data import (
    BusesTable,
    ConvertersTable,
    EffectsTable,
    FlowsTable,
    ModelData,
    StoragesTable,
    build_model_data,
)
from fluxopt.elements import Bus, Effect, Flow, Storage
from fluxopt.model import EnergySystemModel
from fluxopt.results import SolvedModel
from fluxopt.types import TimeSeries, Timesteps, compute_dt, normalize_timesteps, to_polars_series


def solve(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    components: list[Source | Sink | LinearConverter],
    storages: list[Storage] | None = None,
    dt: float | list[float] | pl.Series | None = None,
    solver: str = 'highs',
    silent: bool = True,
) -> SolvedModel:
    """Convenience: build data, build model, solve, return results."""
    data = build_model_data(timesteps, buses, effects, components, storages, dt)
    model = EnergySystemModel(data, solver=solver)
    model.build()
    return model.solve(silent=silent)


__all__ = [
    'Bus',
    'BusesTable',
    'ConvertersTable',
    'Effect',
    'EffectsTable',
    'EnergySystemModel',
    'Flow',
    'FlowsTable',
    'LinearConverter',
    'ModelData',
    'Sink',
    'SolvedModel',
    'Source',
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
