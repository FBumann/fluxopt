"""Per-contributor effect breakdown.

Decomposes solver effect totals into per-contributor (flow/storage) parts,
split into temporal (per-timestep) and lump (sizing + investment costs) domains —
matching the model's own temporal/lump structure.

Two views are supported via the ``cross_effects`` parameter on
``compute_effect_contributions``:

- **with cross-effects** (default): propagates ``contribution_from`` chains
  via the Leontief inverse — ``total = (I - C)^-1 * direct`` — so each
  contributor is charged the full priced-in cost (e.g. CO₂ → cost).
- **direct**: skips Leontief; each contributor shows only effects it
  directly emits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from fluxopt.contract import Dim, Var
from fluxopt.effect_terms import effect_terms

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


def _leontief(cf: xr.DataArray) -> xr.DataArray:
    """Compute Leontief inverse (I - C)^-1 from cross-effect coefficients.

    Args:
        cf: Cross-effect coefficients with dims ``(effect, source_effect)``
            and optionally extra batch dims (e.g. ``time``).
    """
    n = cf.sizes['effect']
    batch_dims = [d for d in cf.dims if d not in ('effect', 'source_effect')]
    ordered = [*batch_dims, 'effect', 'source_effect']
    mat = np.eye(n) - cf.transpose(*ordered).values  # (..., n, n)
    return xr.DataArray(_inverse_per_batch(mat), dims=ordered, coords=cf.coords).transpose(*cf.dims)


def _inverse_per_batch(mat: np.ndarray) -> np.ndarray:
    """Invert a batched matrix array ``(..., n, n)``.

    The cross-effect factors are usually constant along ``time``/``period``,
    so when every batch element holds the same matrix it is decomposed once
    and broadcast instead of once per element.

    Raises:
        ValueError: If any matrix is singular (circular contribution_from chains).
    """
    n = mat.shape[-1]
    stack = mat.reshape(-1, n, n)
    if len(stack) > 1 and bool((stack == stack[0]).all()):
        stack = stack[:1]
    if np.any(np.linalg.matrix_rank(stack) < n):
        raise ValueError('Cross-effect matrix (I - C) is singular — check for circular contribution_from chains')
    inverse = np.linalg.inv(stack)
    return np.broadcast_to(inverse[0], mat.shape) if len(inverse) == 1 else inverse.reshape(mat.shape)


def _apply_leontief(
    leontief: xr.DataArray,
    arr: xr.DataArray,
) -> xr.DataArray:
    """Apply Leontief inverse to an array with an ``effect`` dimension.

    Args:
        leontief: Leontief inverse ``(effect, source_effect[, ...])``.
        arr: Array whose ``effect`` dim is contracted over.
    """
    result: xr.DataArray = xr.dot(leontief, arr.rename({'effect': 'source_effect'}), dim='source_effect', optimize=True)
    return result


def _first_governed_flow(data: ModelData) -> dict[str, str]:
    """Map each component-status component to its first governed flow.

    Component-level costs have no single natural flow, so the decomposition
    attributes them to the component's first governed flow (a presentation
    policy of the breakdown, not part of the model math).
    """
    cst = data.flows.cstatus
    if cst is None or cst.governed_flows is None:
        return {}
    gf = cst.governed_flows
    return {
        str(comp_id): str(gf.sel(cstatus_component=comp_id).values[0])
        for comp_id in gf.coords[Dim.CSTATUS_COMPONENT].values
        if str(gf.sel(cstatus_component=comp_id).values[0])
    }


def _compute_direct(solution: xr.Dataset, data: ModelData) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Compute direct (no cross-effect propagation) per-contributor temporal and lump.

    Evaluates the same term declarations the model built its expressions
    from (:func:`fluxopt.effect_terms.effect_terms`), with solved variable
    values in place of linopy variables. Returns ``(temporal, lump,
    all_ids)`` where each contributor's effects are only those it directly
    emits — independent of ``contribution_from`` chains.
    """
    flow_ids: list[str] = list(data.flows.effect_coeff.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.total_min.coords['effect'].values)
    stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values) if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    dt = data.dims.dt  # (time,)
    first_flow_per_comp = _first_governed_flow(data)

    temporal_flow = xr.zeros_like(data.flows.effect_coeff * dt)  # (flow, effect, time[, period])
    lump: dict[str, xr.DataArray] = {
        entity: xr.DataArray(
            np.zeros((len(ids), len(effect_ids))),
            dims=[entity, 'effect'],
            coords={entity: ids, 'effect': effect_ids},
        )
        for entity, ids in (('flow', flow_ids), ('storage', stor_ids))
        if ids
    }

    for term in effect_terms(data):
        if term.var is None:
            values = term.coeff
        else:
            var = solution[term.var]
            if term.select is not None:
                var = var.sel({term.entity_dim: list(term.select)})
            values = term.coeff * var
        if term.domain == 'temporal' and term.scale_dt:
            values = values * dt

        if term.domain == 'lump':
            lump[term.entity_dim] = lump[term.entity_dim] + values.reindex(
                {term.entity_dim: lump[term.entity_dim].coords[term.entity_dim].values}, fill_value=0.0
            )
        elif term.entity_dim == 'flow':
            temporal_flow = temporal_flow + values.reindex(flow=flow_ids, fill_value=0.0)
        else:  # component-level temporal: attribute to the first governed flow
            for comp_id, fid in first_flow_per_comp.items():
                if fid in flow_ids:
                    add = values.sel(component=comp_id).drop_vars('component')
                    temporal_flow.loc[{'flow': fid}] = temporal_flow.sel(flow=fid) + add

    temporal = temporal_flow.rename({'flow': 'contributor'})
    lump_parts = [arr.rename({entity: 'contributor'}) for entity, arr in lump.items()]
    lump_all = xr.concat(lump_parts, dim='contributor') if len(lump_parts) > 1 else lump_parts[0]
    return temporal, lump_all, all_ids


def _apply_cross_effects(
    temporal: xr.DataArray, lump: xr.DataArray, data: ModelData
) -> tuple[xr.DataArray, xr.DataArray]:
    """Propagate effects along ``contribution_from`` chains via Leontief inverse.

    Caller must ensure ``data.effects.cf_temporal is not None``. Time-varying
    ``contribution_from`` is averaged over time for the lump domain (mirroring
    the model's own treatment in ``model.py``).
    """
    assert data.effects.cf_temporal is not None
    temporal_out = _apply_leontief(_leontief(data.effects.cf_temporal), temporal)
    cf_lump = data.effects.cf_temporal.mean('time')
    lump_out = _apply_leontief(_leontief(cf_lump), lump)
    return temporal_out, lump_out


def _validate_against_solver(total: xr.DataArray, solution: xr.Dataset) -> None:
    """Sanity check: per-contributor totals must sum to solver ``effect--total``.

    Comparison is positional — coordinate misordering or mismatch is a real
    pipeline bug that should fail loudly here rather than be silently aligned.
    """
    solver = solution[Var.EFFECT_TOTAL]
    computed = total.sum('contributor')
    if not np.allclose(computed.values, solver.values, atol=1e-6):
        diff = abs(computed - solver)
        raise ValueError(
            f'Effect contributions do not sum to solver totals. Max deviation: {float(diff.max().values):.6g}'
        )


def _finalize(
    temporal: xr.DataArray,
    lump: xr.DataArray,
    all_ids: list[str],
    data: ModelData,
) -> xr.Dataset:
    """Combine temporal + lump into the public ``(temporal, lump, total)`` Dataset."""
    total = (temporal * data.dims.weights).sum('time').reindex(contributor=all_ids, fill_value=0.0) + lump.reindex(
        contributor=all_ids, fill_value=0.0
    )
    return xr.Dataset({'temporal': temporal, 'lump': lump, 'total': total})


def _with_cross_effects(direct: xr.Dataset, data: ModelData, solution: xr.Dataset) -> xr.Dataset:
    """Apply Leontief cross-effects on top of a precomputed direct contributions Dataset.

    Validates the resulting totals against solver ``effect--total``. When the model has
    no ``contribution_from`` chains, the direct Dataset is already the with-cross
    answer — we just validate and return it.

    Args:
        direct: Output of :func:`compute_effect_contributions` with ``cross_effects=False``.
        data: Model data the ``direct`` was built from.
        solution: Solved variable dataset (used for validation).
    """
    if data.effects.cf_temporal is None:
        _validate_against_solver(direct['total'], solution)
        return direct
    temporal, lump = _apply_cross_effects(direct['temporal'], direct['lump'], data)
    all_ids = list(direct['total'].coords['contributor'].values)
    out = _finalize(temporal, lump, all_ids, data)
    _validate_against_solver(out['total'], solution)
    return out


def compute_effect_contributions(
    solution: xr.Dataset,
    data: ModelData,
    *,
    cross_effects: bool = True,
) -> xr.Dataset:
    """Compute per-contributor effect breakdown from solved values.

    Decomposes effect totals into per-contributor parts on a unified
    ``contributor`` dimension (flow IDs + storage IDs).

    Args:
        solution: Solved variable dataset from ``Result.solution``.
        data: Model data used to build the optimization.
        cross_effects: When True (default), propagates effects along
            ``contribution_from`` chains via the Leontief inverse so each
            contributor is charged the full priced-in cost (e.g. CO₂ → cost).
            When False, returns *direct* contributions only — each contributor
            shows only effects it directly emits, ignoring cross-effects.

    Returns:
        Dataset with:
        - ``temporal`` (contributor, effect, time) — per-timestep contributions
        - ``lump`` (contributor, effect) — lump contributions (flows + storages)
        - ``total`` (contributor, effect) — temporal summed over time + lump

    Raises:
        ValueError: if ``cross_effects=True`` and the contributions don't
            match solver totals (a sanity check on the breakdown).
    """
    temporal, lump, all_ids = _compute_direct(solution, data)
    direct = _finalize(temporal, lump, all_ids, data)
    if cross_effects:
        return _with_cross_effects(direct, data, solution)
    return direct
