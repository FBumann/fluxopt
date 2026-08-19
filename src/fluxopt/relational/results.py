"""Turn an lpspec result into a :class:`~fluxopt.results.Result`.

Both lanes answer with the same object, which is what lets the parity test
compare answers rather than shapes — and what makes deleting the linopy
builder a deletion rather than a migration.

The work is relabelling. The program indexes ``time``, ``period`` and
``build_period`` by ordinal, because the engine joins on them and a timestamp
is a poor join key; :class:`~fluxopt.results.Result` is read by humans and
indexes them by the labels the element layer used. Everything else is a
rename: one program variable to one ``Var`` name.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from fluxopt.contract import Var
from fluxopt.results import Result

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: Program variable -> the name it answers to in a solution. Variables the
#: program shares between two families (``running`` covers a flow's own status
#: and a component's) are split afterwards, by which entity a row belongs to.
_DIRECT: dict[str, str] = {
    'rate': Var.FLOW_RATE,
    'level': Var.STORAGE_LEVEL,
    'size': Var.FLOW_SIZE,
    'size_built': Var.FLOW_SIZE_INDICATOR,
    'capacity': Var.STORAGE_CAPACITY,
    'capacity_built': Var.STORAGE_SIZE_INDICATOR,
    'charging': Var.STORAGE_CHARGING,
    'invest_size': Var.INVEST_SIZE,
    'invest_build': Var.INVEST_BUILD,
    'invest_active': Var.INVEST_ACTIVE,
    'invest_size_at_build': Var.INVEST_SIZE_AT_BUILD,
    'effect_total': Var.EFFECT_TOTAL,
}

#: The status variables, and the pair of names each splits into.
_STATUS: dict[str, tuple[str, str]] = {
    'running': (Var.FLOW_ON, Var.COMPONENT_ON),
    'startup': (Var.FLOW_STARTUP, Var.COMPONENT_STARTUP),
    'shutdown': (Var.FLOW_SHUTDOWN, Var.COMPONENT_SHUTDOWN),
}


def objective_weights(data: ModelData, objective: str | dict[str, float]) -> dict[str, float]:
    """The weights the objective is actually minimised with.

    Not simply what the caller passed: the built-in penalty effect is added at
    1.0 unless the caller named it, which is what makes a soft constraint cost
    something. Both lanes have to agree here or their objectives are different
    numbers for the same request — and ``Result.objective_weights`` is the
    provenance a reader checks that against.
    """
    from fluxopt.elements import PENALTY_EFFECT_ID

    weights = {objective: 1.0} if isinstance(objective, str) else {k: float(v) for k, v in objective.items()}
    effect_ids = {str(e) for e in data.effects.total_min.coords['effect'].values}
    if PENALTY_EFFECT_ID not in weights and PENALTY_EFFECT_ID in effect_ids:
        weights[PENALTY_EFFECT_ID] = 1.0
    return weights


def _entity_order(data: ModelData) -> dict[str, list[str]]:
    """The element layer's own order for each entity axis.

    The engine returns labels sorted, because a label is a join key there.
    Downstream this order is load-bearing rather than cosmetic: ``Result``
    round-trips through NetCDF alongside ``ModelData``, and the contributions
    check compares positionally *on purpose*, so that a misalignment fails
    loudly instead of being silently reindexed away.
    """
    order = {
        'flow': [str(f) for f in data.flows.size.coords['flow'].values],
        'effect': [str(e) for e in data.effects.total_min.coords['effect'].values],
    }
    if data.storages is not None:
        order['storage'] = [str(s) for s in data.storages.capacity.coords['storage'].values]
    return order


def _relabel(arr: xr.DataArray, data: ModelData) -> xr.DataArray:
    """Put the element layer's own labels and order back on the axes."""
    dims = data.dims
    periods = list(dims.period.values) if dims.period is not None else [0]
    for name, labels in (
        ('time', list(dims.time.values)),
        ('period', periods),
        ('build_period', periods),
    ):
        if name in arr.dims:
            arr = arr.assign_coords({name: [labels[int(i)] for i in arr.coords[name].values]})
    for name, labels in _entity_order(data).items():
        if name in arr.dims:
            have = {str(x) for x in arr.coords[name].values}
            present = [v for v in labels if v in have]
            arr = arr.sel({name: present})
    # A single-period model never named a period, so it does not carry one out.
    if dims.period is None and 'period' in arr.dims:
        arr = arr.squeeze('period', drop=True)
    return arr


#: Axes whose membership a `where:` mask can thin out. Densifying over one of
#: these invents an entity, so the primal frame decides who is really there.
_ENTITY_DIMS = ('flow', 'storage', 'status_entity', 'converter', 'effect')


def _read(result: Any, name: str, data: ModelData) -> xr.DataArray | None:
    """One program variable as a labelled array, or None if it has no columns.

    ``to_dataarray`` is dense over the declared dimensions, so a variable the
    program masked down to one flow still comes back spanning every flow. The
    padding is NaN and a solved value never is, which is what makes the two
    tellable apart — an entity with no column anywhere is one the model never
    gave the decision to, and carrying it would read as decided-at-zero.
    """
    try:
        arr = result.to_dataarray(name)
    except Exception:  # absent variable, whatever the lane calls it
        return None
    if not arr.size:
        return None
    for dim in _ENTITY_DIMS:
        if dim not in arr.dims:
            continue
        others = [d for d in arr.dims if d != dim]
        has_column = arr.notnull().any(others)
        arr = arr.sel({dim: arr.coords[dim].values[has_column.values]})
    return _relabel(arr, data) if arr.size else None


def _split_status(arr: xr.DataArray, data: ModelData) -> tuple[xr.DataArray | None, xr.DataArray | None]:
    """Divide one status array into its flow rows and its component rows.

    The program keeps both on one ``status_entity`` axis because they obey the
    same math; a solution keeps them apart because they are different things to
    read. Membership decides: an entity that is a flow is a flow.
    """
    flow_ids = {str(f) for f in data.flows.size.coords['flow'].values}
    entities = [str(e) for e in arr.coords['status_entity'].values]
    parts: list[xr.DataArray | None] = []
    for wanted, dim in ((True, 'flow'), (False, 'component')):
        ids = [e for e in entities if (e in flow_ids) is wanted]
        parts.append(
            arr.sel(status_entity=ids).rename({'status_entity': dim}).assign_coords({dim: ids}) if ids else None
        )
    return parts[0], parts[1]


def to_result(result: Any, data: ModelData, weights: dict[str, float]) -> Result:
    """Build a :class:`~fluxopt.results.Result` from a solved lpspec model.

    Args:
        result: The lpspec result — primal frames and named expressions.
        data: The model data that was bound. Carries the labels the program
            indexes by ordinal, and the tables ``Result`` reads alongside the
            solution.
        weights: Effect ids mapped to their objective weight, for provenance.

    Returns:
        The same object the linopy lane returns.

    Raises:
        RuntimeError: If the solve did not produce a primal solution — there
            is no result to read, and an empty one would read as zeros.
    """
    if not result.has_primal:
        raise RuntimeError(f'no primal solution to read: the solver terminated {result.termination_condition!r}')

    solution: dict[str, xr.DataArray] = {}
    for program_name, var_name in _DIRECT.items():
        if (arr := _read(result, program_name, data)) is not None:
            solution[var_name] = arr
    for program_name, (flow_var, component_var) in _STATUS.items():
        if (arr := _read(result, program_name, data)) is None:
            continue
        on_flows, on_components = _split_status(arr, data)
        if on_flows is not None:
            solution[flow_var] = on_flows
        if on_components is not None:
            solution[component_var] = on_components

    # The lump half of the ledger is a named expression rather than a variable:
    # nothing in the model decides it on its own, so it is read back through
    # the same compiler that built the accounting row.
    lump = result.expression('effect_lump')
    effects = [str(e) for e in data.effects.total_min.coords['effect'].values]
    solution[Var.EFFECT_LUMP] = _relabel(_frame_to_array(lump, 'effect_lump', effects, data), data)

    return Result(
        solution=xr.Dataset(
            solution,
            attrs={'objective': float(result.objective), 'objective_weights': json.dumps(weights)},
        ),
        data=data,
    )


def _frame_to_array(frame: Any, name: str, effects: list[str], data: ModelData) -> xr.DataArray:
    """A tidy ``(period, effect, value)`` frame as a dense ``(effect, period)`` array."""
    periods = list(range(len(data.dims.period.values))) if data.dims.period is not None else [0]
    values = np.zeros((len(effects), len(periods)))
    index = {e: i for i, e in enumerate(effects)}
    for row in frame.iter_rows(named=True):
        values[index[row['effect']], periods.index(int(row['period']))] = row['value']
    return xr.DataArray(
        values,
        dims=['effect', 'period'],
        coords={'effect': effects, 'period': periods},
        name=name,
    )
