"""Per-contributor effect breakdown, read off the program.

Every contribution the ledger sums is a named expression in
``math/program.yaml``, declared with the entity it came from and reduced only
where ``effect_accounting`` adds it up. So the breakdown is not a second
implementation of the effect math that has to be checked against the first —
it is the same declaration, read one step before the sum.

Two views, as before:

- **with cross-effects** (default): each contributor is charged the full
  priced-in cost, CO2 through to cost. This is what the expressions give
  directly, because the coefficients bound to the program already carry the
  Leontief inverse (:mod:`fluxopt.leontief`) — the model never multiplies it
  at build time.
- **direct**: each contributor shows only what it directly emits, recovered
  as ``(I - C) . charged``. A forward multiply, and the exact inverse of the
  fold the binder applied.

The contributor axis is a presentation choice rather than model math: flows
and storages share one dimension, and a component-level cost is attributed to
the first flow its status governs, having no single natural flow of its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from fluxopt.contract import Var

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: Named expressions the program declares per contributor, and the entity each
#: is keyed by. ``effect_temporal`` and ``effect_lump`` sum exactly these.
TEMPORAL: dict[str, str] = {
    'contribution_flow_hour': 'flow',
    'contribution_running': 'status_entity',
    'contribution_startup': 'status_entity',
}
LUMP: dict[str, str] = {
    'contribution_flow_per_size': 'flow',
    'contribution_flow_fixed': 'flow',
    'contribution_storage_per_capacity': 'storage',
    'contribution_storage_fixed': 'storage',
    'contribution_invest_per_size_at_build': 'flow',
    'contribution_invest_fixed_at_build': 'flow',
    'contribution_invest_per_size_recurring': 'flow',
    'contribution_invest_fixed_recurring': 'flow',
}


def _first_governed_flow(data: ModelData) -> dict[str, str]:
    """Map each component-status component to its first governed flow.

    Component-level costs have no single natural flow, so the decomposition
    attributes them to the component's first governed flow (a presentation
    policy of the breakdown, not part of the model math).
    """
    governed = data.flows.governed_by
    if governed is None:
        return {}
    first: dict[str, str] = {}
    for fid, owner in zip(governed.coords['flow'].values, governed.values, strict=True):
        if str(owner):
            first.setdefault(str(owner), str(fid))
    return first


def _onto_contributor(arr: xr.DataArray, entity: str, data: ModelData, flow_ids: list[str]) -> xr.DataArray:
    """Rename an entity axis to ``contributor``, folding status entities onto flows."""
    if entity != 'status_entity':
        return arr.rename({entity: 'contributor'})
    governed = _first_governed_flow(data)
    labels = [str(e) for e in arr.coords[entity].values]
    known = set(flow_ids)
    # A status entity is either a flow (carrying its own Status) or a component
    # (whose Status governs several); the second is charged to a flow it governs.
    mapped = [label if label in known else governed.get(label, '') for label in labels]
    keep = [i for i, target in enumerate(mapped) if target]
    arr = arr.isel({entity: keep}).assign_coords({entity: [mapped[i] for i in keep]})
    return arr.rename({entity: 'contributor'}).groupby('contributor').sum()


def _gather(
    read: Any, names: dict[str, str], data: ModelData, all_ids: list[str], collapse: str | None
) -> xr.DataArray:
    """Sum the named contributions onto one contributor axis."""
    flow_ids = [str(f) for f in data.flows.flow_id.values]
    total: xr.DataArray | None = None
    for name, entity in names.items():
        arr = read(name)
        if arr is None:
            continue
        if collapse and collapse in arr.dims:
            arr = arr.sum(collapse)
        # `fillna` as well as `fill_value`: the expressions are stored in one
        # Dataset, so a name whose entity axis is narrower than another's was
        # aligned to the union and padded. A solved product is never NaN, so
        # padding there means the entity had no row — which contributes zero.
        part = _onto_contributor(arr, entity, data, flow_ids).reindex(contributor=all_ids, fill_value=0.0).fillna(0.0)
        total = part if total is None else total + part
    if total is None:
        effects = data.effects.ids
        total = xr.DataArray(
            np.zeros((len(all_ids), len(effects))),
            dims=['contributor', 'effect'],
            coords={'contributor': all_ids, 'effect': effects},
        )
    return total


def _undo_cross_effects(
    temporal: xr.DataArray, lump: xr.DataArray, data: ModelData
) -> tuple[xr.DataArray, xr.DataArray]:
    """Recover the direct view: ``(I - C) . charged``.

    The exact inverse of the fold the binder applied, and a forward multiply
    rather than another inversion — so the two views cannot disagree about
    anything but floating point.
    """
    periods = list(data.dims.period.values) if data.dims.period is not None else None
    cf = data.effects.cf_matrix(periods)
    if cf is None:
        return temporal, lump

    def unfold(arr: xr.DataArray, matrix: xr.DataArray) -> xr.DataArray:
        n = matrix.sizes['effect']
        identity = xr.DataArray(
            np.eye(n),
            dims=['effect', 'source_effect'],
            coords={'effect': matrix.coords['effect'], 'source_effect': matrix.coords['source_effect']},
        )
        out: xr.DataArray = xr.dot(
            identity - matrix, arr.rename({'effect': 'source_effect'}), dim='source_effect', optimize=True
        )
        return out

    return unfold(temporal, cf), unfold(lump, cf.mean('time'))


def _finalize(temporal: xr.DataArray, lump: xr.DataArray, all_ids: list[str], data: ModelData) -> xr.Dataset:
    """Combine temporal + lump into the public ``(temporal, lump, total)`` Dataset."""
    total = (temporal * data.dims.weights).sum('time').reindex(contributor=all_ids, fill_value=0.0) + lump.reindex(
        contributor=all_ids, fill_value=0.0
    )

    def lead(arr: xr.DataArray) -> xr.DataArray:
        """Contributor first, then effect — the order the accessors document."""
        front = [d for d in ('contributor', 'effect') if d in arr.dims]
        return arr.transpose(*front, *[d for d in arr.dims if d not in front])

    return xr.Dataset({'temporal': lead(temporal), 'lump': lead(lump), 'total': lead(total)})


def contributions_from(read: Any, data: ModelData, *, cross_effects: bool = True) -> xr.Dataset:
    """The breakdown, assembled from the program's own contribution expressions.

    Args:
        read: Reads one named expression as a labelled array, or None where
            the program declares it but this system has no rows for it.
        data: The model data that was bound.
        cross_effects: Charge each contributor the full priced-in cost
            (default), or show only what it directly emits.

    Returns:
        Dataset with ``temporal`` (contributor, effect, time), ``lump``
        (contributor, effect) and ``total`` (contributor, effect).
    """
    flow_ids = [str(f) for f in data.flows.flow_id.values]
    stor_ids = [str(s) for s in data.storages.capacity.coords['storage'].values] if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    temporal = _gather(read, TEMPORAL, data, all_ids, None)
    lump = _gather(read, LUMP, data, all_ids, 'build_period')
    if not cross_effects:
        temporal, lump = _undo_cross_effects(temporal, lump, data)
    return _finalize(temporal, lump, all_ids, data)


def validate_against_solver(total: xr.DataArray, solution: xr.Dataset) -> None:
    """The breakdown must add up to the totals the solver reported.

    Kept although the breakdown and the ledger are now one declaration: what
    it guards is no longer a second implementation of the math, but the
    assembly above it — the contributor mapping, the reindexing, and the
    coordinate order the comparison is positional in.
    """
    solver = solution[Var.EFFECT_TOTAL]
    # Aligned by dim *name* — which axis comes first is arbitrary and differs
    # between the solution and an expression's own declaration order. What
    # stays positional is the coordinate order *within* each dim, because a
    # misordering there is a real pipeline bug rather than a presentation one.
    computed = total.sum('contributor').transpose(*solver.dims)
    if not np.allclose(computed.values, solver.values, atol=1e-6):
        diff = abs(computed - solver)
        raise ValueError(
            f'Effect contributions do not sum to solver totals. Max deviation: {float(diff.max().values):.6g}'
        )
