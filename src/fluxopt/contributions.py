"""Per-flow effect contribution breakdown.

Computes how much each flow contributes to each effect, broken down into
operational and investment parts. Storage investment costs are reported
separately on the storage dimension. All math is post-hoc xarray arithmetic
on solved values — no linopy variables.

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
            or ``(effect, source_effect, time)``.
    """
    n = cf.sizes['effect']
    eye = np.eye(n)

    if 'time' not in cf.dims:
        # 2D: single matrix inverse
        inv = np.linalg.inv(eye - cf.values)
        return xr.DataArray(inv, dims=cf.dims, coords=cf.coords)

    # 3D: batch inverse over time axis
    mat = eye[np.newaxis] - cf.transpose('time', 'effect', 'source_effect').values
    inv = np.linalg.inv(mat)  # (T, n, n)
    return xr.DataArray(
        inv.transpose((1, 2, 0)),  # → (effect, source_effect, time)
        dims=['effect', 'source_effect', 'time'],
        coords=cf.coords,
    )


def _apply_leontief(
    leontief: xr.DataArray,
    arr: xr.DataArray,
) -> xr.DataArray:
    """Apply Leontief inverse to an array with an ``effect`` dimension.

    Args:
        leontief: Leontief inverse ``(effect, source_effect[, time])``.
        arr: Array whose ``effect`` dim is contracted over.
    """
    result: xr.DataArray = xr.dot(leontief, arr.rename({'effect': 'source_effect'}), dim='source_effect')
    return result


def _compute_investment(
    effects_per_size: xr.DataArray | None,
    effects_fixed: xr.DataArray | None,
    mandatory: xr.DataArray | None,
    solution: xr.Dataset,
    entity_ids: list[str],
    effect_ids: list[str],
    sizing_dim: str,
    entity_dim: str,
    size_var: str,
    indicator_var: str,
) -> xr.DataArray:
    """Compute investment contributions for flows or storages.

    Args:
        effects_per_size: Per-unit sizing costs ``(sizing_dim, effect)`` or None.
        effects_fixed: Fixed sizing costs ``(sizing_dim, effect)`` or None.
        mandatory: Boolean mask for mandatory sizing ``(sizing_dim,)`` or None.
        solution: Solved variable dataset.
        entity_ids: All entity ids (flow or storage).
        effect_ids: All effect ids.
        sizing_dim: Sizing dimension name (``sizing_flow`` or ``sizing_storage``).
        entity_dim: Entity dimension name (``flow`` or ``storage``).
        size_var: Solution variable name for size.
        indicator_var: Solution variable name for binary indicator.
    """
    inv = xr.DataArray(
        np.zeros((len(entity_ids), len(effect_ids))),
        dims=[entity_dim, 'effect'],
        coords={entity_dim: entity_ids, 'effect': effect_ids},
    )
    rename = {sizing_dim: entity_dim}

    # Per-size costs
    if effects_per_size is not None and size_var in solution:
        eps = effects_per_size.rename(rename)
        inv = inv + (eps * solution[size_var]).reindex({entity_dim: entity_ids}, fill_value=0.0)

    # Fixed costs — optional (binary indicator * cost)
    if effects_fixed is not None and indicator_var in solution:
        indicator = solution[indicator_var]
        opt_ids = list(indicator.coords[entity_dim].values)
        ef = effects_fixed.rename(rename).sel({entity_dim: opt_ids})
        inv = inv + (ef * indicator).reindex({entity_dim: entity_ids}, fill_value=0.0)

    # Fixed costs — mandatory (constant)
    if effects_fixed is not None and mandatory is not None:
        mand_mask = mandatory.values
        if mand_mask.any():
            mand_ids = list(mandatory.coords[sizing_dim].values[mand_mask])
            ef_mand = effects_fixed.sel({sizing_dim: mand_ids}).rename(rename)
            inv = inv + ef_mand.reindex({entity_dim: entity_ids}, fill_value=0.0)

    return inv


def compute_effect_contributions(solution: xr.Dataset, data: ModelData) -> xr.Dataset:
    """Compute per-flow effect contributions from solved values.

    Args:
        solution: Solved variable dataset from ``Result.solution``.
        data: Model data used to build the optimization.

    Returns:
        Dataset with:
        - ``operational`` (flow, effect, time) — per-flow operational contributions
        - ``investment`` (flow, effect) — per-flow investment (flow sizing)
        - ``storage_investment`` (storage, effect) — storage sizing investment
        - ``total`` (flow, effect) — operational summed over time + flow investment
    """
    flow_ids: list[str] = list(data.flows.effect_coeff.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.min_total.coords['effect'].values)

    if len(effect_ids) == 0:
        return xr.Dataset()

    rate = solution['flow--rate']  # (flow, time)
    dt = data.dt  # (time,)

    # --- Operational: direct per-flow contributions (flow, effect, time) ---
    flow_op = data.flows.effect_coeff * rate * dt

    # Status running costs
    if data.flows.status_effects_running is not None and 'flow--on' in solution:
        er = data.flows.status_effects_running.rename({'status_flow': 'flow'})
        flow_op = flow_op + (er * solution['flow--on'] * dt).reindex(flow=flow_ids, fill_value=0.0)

    # Status startup costs
    if data.flows.status_effects_startup is not None and 'flow--startup' in solution:
        es = data.flows.status_effects_startup.rename({'status_flow': 'flow'})
        flow_op = flow_op + (es * solution['flow--startup']).reindex(flow=flow_ids, fill_value=0.0)

    # Cross-effects on operational via Leontief inverse
    if data.effects.cf_temporal is not None:
        flow_op = _apply_leontief(_leontief(data.effects.cf_temporal), flow_op)

    # --- Flow investment ---
    flow_inv = _compute_investment(
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
    if data.effects.cf_periodic is not None:
        flow_inv = _apply_leontief(_leontief(data.effects.cf_periodic), flow_inv)

    # --- Storage investment ---
    stor_inv: xr.DataArray | None = None
    if data.storages is not None:
        stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values)
        stor_inv = _compute_investment(
            data.storages.sizing_effects_per_size,
            data.storages.sizing_effects_fixed,
            data.storages.sizing_mandatory,
            solution,
            stor_ids,
            effect_ids,
            'sizing_storage',
            'storage',
            'storage--capacity',
            'storage--size_indicator',
        )
        if data.effects.cf_periodic is not None:
            stor_inv = _apply_leontief(_leontief(data.effects.cf_periodic), stor_inv)

    # --- Total per flow: operational (weighted sum over time) + flow investment ---
    total = (flow_op * data.weights).sum('time') + flow_inv

    result_vars: dict[str, xr.DataArray] = {
        'operational': flow_op,
        'investment': flow_inv,
        'total': total,
    }
    if stor_inv is not None:
        result_vars['storage_investment'] = stor_inv

    return xr.Dataset(result_vars)
