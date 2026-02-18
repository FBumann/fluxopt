"""Per-contributor effect breakdown.

Decomposes solver effect totals into per-contributor (flow/storage) parts,
split into temporal (per-timestep) and periodic (sizing, fixed costs) domains — matching
the model's own temporal/periodic structure.

Cross-effects use the Leontief inverse: total = (I - C)^-1 * direct,
where C is the cross-effect coefficient matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


def _leontief(cf: xr.DataArray) -> xr.DataArray:
    """Compute Leontief inverse (I - C)^-1 from cross-effect coefficients.

    Args:
        cf: Cross-effect coefficients with dims ``(effect, source_effect)``
            and optionally extra batch dims (e.g. ``time``).
    """
    n = cf.sizes['effect']
    other_dims = [d for d in cf.dims if d not in ('effect', 'source_effect')]
    ordered = [*other_dims, 'effect', 'source_effect']
    vals = cf.transpose(*ordered).values  # (..., n, n)
    inv = np.linalg.inv(np.eye(n) - vals)  # (..., n, n)
    return xr.DataArray(inv, dims=ordered, coords=cf.coords)


def _apply_leontief(
    leontief: xr.DataArray,
    arr: xr.DataArray,
) -> xr.DataArray:
    """Apply Leontief inverse to an array with an ``effect`` dimension.

    Args:
        leontief: Leontief inverse ``(effect, source_effect[, ...])``.
        arr: Array whose ``effect`` dim is contracted over.
    """
    result: xr.DataArray = xr.dot(leontief, arr.rename({'effect': 'source_effect'}), dim='source_effect')
    return result


def _compute_periodic(
    effects_per_size: xr.DataArray | None,
    effects_fixed: xr.DataArray | None,
    mandatory: xr.DataArray | None,
    solution: xr.Dataset,
    contributor_ids: list[str],
    effect_ids: list[str],
    sizing_dim: str,
    entity_dim: str,
    size_var: str,
    indicator_var: str,
) -> xr.DataArray:
    """Compute periodic contributions for flows or storages.

    Args:
        effects_per_size: Per-unit sizing costs ``(sizing_dim, effect)`` or None.
        effects_fixed: Fixed sizing costs ``(sizing_dim, effect)`` or None.
        mandatory: Boolean mask for mandatory sizing ``(sizing_dim,)`` or None.
        solution: Solved variable dataset.
        contributor_ids: Contributor ids for this entity type.
        effect_ids: All effect ids.
        sizing_dim: Sizing dimension name (``sizing_flow`` or ``sizing_storage``).
        entity_dim: Entity dimension name (``flow`` or ``storage``).
        size_var: Solution variable name for size.
        indicator_var: Solution variable name for binary indicator.
    """
    result = xr.DataArray(
        np.zeros((len(contributor_ids), len(effect_ids))),
        dims=['contributor', 'effect'],
        coords={'contributor': contributor_ids, 'effect': effect_ids},
    )
    rename = {sizing_dim: entity_dim}

    # Per-size costs
    if effects_per_size is not None and size_var in solution:
        eps = effects_per_size.rename(rename)
        term = (eps * solution[size_var]).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — optional (binary indicator * cost)
    if effects_fixed is not None and indicator_var in solution:
        indicator = solution[indicator_var]
        opt_ids = list(indicator.coords[entity_dim].values)
        ef = effects_fixed.rename(rename).sel({entity_dim: opt_ids})
        term = (ef * indicator).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — mandatory (constant)
    if effects_fixed is not None and mandatory is not None:
        mand_mask = mandatory.values
        if mand_mask.any():
            mand_ids = list(mandatory.coords[sizing_dim].values[mand_mask])
            ef_mand = effects_fixed.sel({sizing_dim: mand_ids}).rename(rename)
            term = ef_mand.reindex({entity_dim: contributor_ids}, fill_value=0.0)
            result = result + term.rename({entity_dim: 'contributor'})

    return result


def compute_effect_contributions(solution: xr.Dataset, data: ModelData) -> xr.Dataset:
    """Compute per-contributor effect breakdown from solved values.

    Decomposes solver totals into per-contributor parts on a unified
    ``contributor`` dimension (flow IDs + storage IDs).

    Args:
        solution: Solved variable dataset from ``Result.solution``.
        data: Model data used to build the optimization.

    Returns:
        Dataset with:
        - ``temporal`` (contributor, effect, time) — per-timestep contributions (flows only)
        - ``periodic`` (contributor, effect) — periodic contributions (flows + storages)
        - ``total`` (contributor, effect) — temporal summed over time + periodic
    """
    flow_ids: list[str] = list(data.flows.effect_coeff.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.min_total.coords['effect'].values)
    stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values) if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    rate = solution['flow--rate']  # (flow, time)
    dt = data.dt  # (time,)

    # --- Temporal: per-flow contributions (flow, effect, time) ---
    temporal_flow = data.flows.effect_coeff * rate * dt

    # Status running costs
    if data.flows.status_effects_running is not None and 'flow--on' in solution:
        er = data.flows.status_effects_running.rename({'status_flow': 'flow'})
        temporal_flow = temporal_flow + (er * solution['flow--on'] * dt).reindex(flow=flow_ids, fill_value=0.0)

    # Status startup costs
    if data.flows.status_effects_startup is not None and 'flow--startup' in solution:
        es = data.flows.status_effects_startup.rename({'status_flow': 'flow'})
        temporal_flow = temporal_flow + (es * solution['flow--startup']).reindex(flow=flow_ids, fill_value=0.0)

    # Cross-effects on temporal via Leontief inverse
    if data.effects.cf_temporal is not None:
        temporal_flow = _apply_leontief(_leontief(data.effects.cf_temporal), temporal_flow)

    # Rename to contributor dim
    temporal = temporal_flow.rename({'flow': 'contributor'})

    # --- Periodic: flow costs ---
    flow_periodic = _compute_periodic(
        data.flows.sizing_effects_per_size,
        data.flows.sizing_effects_fixed,
        data.flows.sizing_mandatory,
        solution,
        flow_ids,
        effect_ids,
        'sizing_flow',
        'flow',
        'flow--size',
        'flow--size_indicator',
    )

    # --- Periodic: storage costs ---
    if stor_ids:
        stor_periodic = _compute_periodic(
            data.storages.sizing_effects_per_size,  # type: ignore[union-attr]
            data.storages.sizing_effects_fixed,  # type: ignore[union-attr]
            data.storages.sizing_mandatory,  # type: ignore[union-attr]
            solution,
            stor_ids,
            effect_ids,
            'sizing_storage',
            'storage',
            'storage--capacity',
            'storage--size_indicator',
        )
        periodic = xr.concat([flow_periodic, stor_periodic], dim='contributor')
    else:
        periodic = flow_periodic

    # Cross-effects on periodic via Leontief inverse
    if data.effects.cf_periodic is not None:
        periodic = _apply_leontief(_leontief(data.effects.cf_periodic), periodic)

    # --- Total: temporal (weighted sum over time) + periodic ---
    total = (temporal * data.weights).sum('time').reindex(contributor=all_ids, fill_value=0.0) + periodic.reindex(
        contributor=all_ids, fill_value=0.0
    )

    # --- Validate: contributions must sum to solver effect totals ---
    solver = solution['effect--total']
    computed = total.sum('contributor')
    if not np.allclose(computed.values, solver.values, atol=1e-6):
        diff = abs(computed - solver)
        raise ValueError(
            f'Effect contributions do not sum to solver totals. Max deviation: {float(diff.max().values):.6g}'
        )

    return xr.Dataset({'temporal': temporal, 'periodic': periodic, 'total': total})
