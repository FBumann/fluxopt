from energysys.components import LinearConverter, Sink, Source
from energysys.data import (
    BusesTable,
    ConvertersTable,
    EffectsTable,
    FlowsTable,
    ModelData,
    StoragesTable,
    build_model_data,
)
from energysys.elements import Bus, Effect, Flow, Storage
from energysys.model import EnergySystemModel
from energysys.results import SolvedModel
from energysys.types import TimeSeries, to_polars_series


def solve(
    timesteps,
    buses,
    effects,
    components,
    storages=None,
    dt=1.0,
    solver='highs',
    silent=True,
):
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
    'build_model_data',
    'solve',
    'to_polars_series',
]
