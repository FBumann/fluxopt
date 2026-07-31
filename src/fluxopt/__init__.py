from collections.abc import Callable, Mapping
from typing import Any

from fluxopt.components import Converter, Port
from fluxopt.elements import (
    PENALTY_EFFECT_ID,
    Carrier,
    Effect,
    Flow,
    Investment,
    PiecewiseConversion,
    Sizing,
    Status,
    Storage,
)
from fluxopt.flow_system import FlowSystem
from fluxopt.model import FlowSystemModel
from fluxopt.model_data import Dims, ModelData
from fluxopt.results import Result
from fluxopt.schema import all_element_schemas, element_schema, from_dict, to_dict
from fluxopt.types import (
    IdList,
    ProfileRef,
    TimeIndex,
    Timesteps,
    Variate,
    as_dataarray,
)


def optimize(
    timesteps: Timesteps | Mapping[int, Timesteps],
    carriers: list[Carrier],
    effects: list[Effect],
    ports: list[Port],
    objective: str | dict[str, float],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    periods: list[int] | None = None,
    period_weights: list[float] | None = None,
    profiles: Mapping[str, Any] | None = None,
    solver: str = 'highs',
    customize: Callable[[FlowSystemModel], None] | None = None,
    **kwargs: Any,
) -> Result:
    """Build data, build model, optimize, return results.

    Args:
        timesteps: Time index for the optimization horizon, or a
            ``{period: index}`` mapping for ragged multi-period grids.
        carriers: Carrier declarations.
        effects: Effects to track (costs, emissions, etc.).
        ports: System boundary ports with imports/exports.
        objective: Effect(s) to minimize. A single name, or a dict
            mapping effect names to objective weights
            (``{'cost': 1, 'co2': 50}``) — tracked effect totals are
            unaffected by the weighting. The built-in ``'penalty'`` effect
            is added at weight 1.0 unless the dict names it
            (``{'cost': 1, 'penalty': 0}`` opts out).
        converters: Linear converters between carriers.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        periods: Integer period labels for multi-period optimization.
        period_weights: Explicit weights per period. Inferred from gaps if None.
        profiles: Mapping from ``ProfileRef.dataset`` to a dataset (or mapping)
            holding referenced time series. Required if any element uses a
            ``ProfileRef``.
        solver: Solver backend name.
        customize: Optional callback to modify the linopy model between build and solve.
            Receives the built FlowSystemModel; use ``model.m`` to add variables/constraints.
        **kwargs: Passed through to ``linopy.Model.solve()``.
    """
    system = FlowSystem(
        timesteps=timesteps,
        carriers=carriers,
        effects=effects,
        ports=ports,
        objective=objective,
        converters=converters or [],
        storages=storages or [],
        dt=dt,
        periods=periods,
        period_weights=period_weights,
    )
    return system.optimize(profiles, customize=customize, solver=solver, **kwargs)


__all__ = [
    'PENALTY_EFFECT_ID',
    'Carrier',
    'Converter',
    'Dims',
    'Effect',
    'Flow',
    'FlowSystem',
    'FlowSystemModel',
    'IdList',
    'Investment',
    'ModelData',
    'PiecewiseConversion',
    'Port',
    'ProfileRef',
    'Result',
    'Sizing',
    'Status',
    'Storage',
    'TimeIndex',
    'Timesteps',
    'Variate',
    'all_element_schemas',
    'as_dataarray',
    'element_schema',
    'from_dict',
    'optimize',
    'to_dict',
]
