"""Per-flow effect contribution breakdown.

Computes how much each flow contributes to each effect, broken down into
operational and investment parts. Storage investment costs are reported
separately on the storage dimension. All math is post-hoc xarray arithmetic
on solved values — no linopy variables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


def _topological_order(cf: xr.DataArray, dim: str = 'effect', src_dim: str = 'source_effect') -> list[str]:
    """Return effects in dependency order (leaves first).

    Args:
        cf: Cross-effect coefficient array with *dim* and *src_dim* axes.
        dim: Target effect dimension name.
        src_dim: Source effect dimension name.
    """
    effect_ids: list[str] = list(cf.coords[dim].values)
    # Build in-degree map: effect k depends on j when cf[k, j] != 0
    in_degree: dict[str, int] = dict.fromkeys(effect_ids, 0)
    dependents: dict[str, list[str]] = {e: [] for e in effect_ids}
    for k in effect_ids:
        for j in effect_ids:
            if k == j:
                continue
            sel = cf.sel({dim: k, src_dim: j})
            if float(np.abs(sel).max()) > 0:
                in_degree[k] += 1
                dependents[j].append(k)

    # Kahn's algorithm
    queue = [e for e in effect_ids if in_degree[e] == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)
    return order


def _apply_cross_effects_2d(
    arr: xr.DataArray,
    cf: xr.DataArray,
    effect_ids: list[str],
    entity_dim: str,
) -> None:
    """Propagate cross-effects in-place on a 2D (entity, effect) array.

    Args:
        arr: Mutable array with dims (*entity_dim*, 'effect').
        cf: Cross-effect coefficients (effect, source_effect).
        effect_ids: All effect ids.
        entity_dim: Name of the non-effect dimension ('flow' or 'storage').
    """
    order = _topological_order(cf)
    for k in order:
        for j in effect_ids:
            if k == j:
                continue
            cf_kj = float(cf.sel(effect=k, source_effect=j))
            if abs(cf_kj) > 0:
                arr.loc[{entity_dim: slice(None), 'effect': k}] = arr.sel(effect=k) + cf_kj * arr.sel(effect=j)


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

    n_flows = len(flow_ids)
    n_eff = len(effect_ids)

    rate = solution['flow--rate']  # (flow, time)
    dt = data.dt  # (time,)

    # --- Operational: direct per-flow contributions (flow, effect, time) ---
    flow_op = data.flows.effect_coeff * rate * dt

    # Status running costs
    if data.flows.status_effects_running is not None and 'flow--on' in solution:
        er = data.flows.status_effects_running.rename({'status_flow': 'flow'})
        on = solution['flow--on']
        flow_op = flow_op + (er * on * dt).reindex(flow=flow_ids, fill_value=0.0)

    # Status startup costs
    if data.flows.status_effects_startup is not None and 'flow--startup' in solution:
        es = data.flows.status_effects_startup.rename({'status_flow': 'flow'})
        startup = solution['flow--startup']
        flow_op = flow_op + (es * startup).reindex(flow=flow_ids, fill_value=0.0)

    # Cross-effects on operational (topological order)
    if data.effects.cf_per_hour is not None:
        cf = data.effects.cf_per_hour  # (effect, source_effect, time)
        order = _topological_order(cf)
        for k in order:
            for j in effect_ids:
                if k == j:
                    continue
                cf_kj = cf.sel(effect=k, source_effect=j)
                if float(np.abs(cf_kj).max()) > 0:
                    flow_op.loc[:, k, :] = flow_op.sel(effect=k) + cf_kj * flow_op.sel(effect=j)

    # --- Investment: per-flow (flow sizing) ---
    flow_inv = xr.DataArray(
        np.zeros((n_flows, n_eff)),
        dims=['flow', 'effect'],
        coords={'flow': flow_ids, 'effect': effect_ids},
    )

    # Flow sizing: per-size costs
    if data.flows.sizing_effects_per_size is not None and 'flow--size' in solution:
        eps = data.flows.sizing_effects_per_size.rename({'sizing_flow': 'flow'})  # (flow, effect)
        flow_size = solution['flow--size']  # (flow,)
        flow_inv = flow_inv + (eps * flow_size).reindex(flow=flow_ids, fill_value=0.0)

    # Flow sizing: fixed costs — optional (binary * cost)
    if data.flows.sizing_effects_fixed is not None and 'flow--size_indicator' in solution:
        indicator = solution['flow--size_indicator']  # (flow,)
        opt_ids = list(indicator.coords['flow'].values)
        ef = data.flows.sizing_effects_fixed.rename({'sizing_flow': 'flow'}).sel(flow=opt_ids)
        flow_inv = flow_inv + (ef * indicator).reindex(flow=flow_ids, fill_value=0.0)

    # Flow sizing: fixed costs — mandatory (constant)
    if data.flows.sizing_effects_fixed is not None and data.flows.sizing_mandatory is not None:
        mand_mask = data.flows.sizing_mandatory.values
        if mand_mask.any():
            mand_ids = list(data.flows.sizing_mandatory.coords['sizing_flow'].values[mand_mask])
            ef_mand = data.flows.sizing_effects_fixed.sel(sizing_flow=mand_ids).rename({'sizing_flow': 'flow'})
            flow_inv = flow_inv + ef_mand.reindex(flow=flow_ids, fill_value=0.0)

    # Cross-effects on flow investment (topological order)
    if data.effects.cf_invest is not None:
        _apply_cross_effects_2d(flow_inv, data.effects.cf_invest, effect_ids, 'flow')

    # --- Storage investment ---
    stor_inv: xr.DataArray | None = None
    if data.storages is not None:
        stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values)
        n_stor = len(stor_ids)
        stor_inv = xr.DataArray(
            np.zeros((n_stor, n_eff)),
            dims=['storage', 'effect'],
            coords={'storage': stor_ids, 'effect': effect_ids},
        )

        # Storage sizing: per-size costs
        if data.storages.sizing_effects_per_size is not None and 'storage--capacity' in solution:
            eps_s = data.storages.sizing_effects_per_size.rename({'sizing_storage': 'storage'})
            cap = solution['storage--capacity']
            stor_inv = stor_inv + (eps_s * cap).reindex(storage=stor_ids, fill_value=0.0)

        # Storage sizing: fixed costs — optional (binary * cost)
        if data.storages.sizing_effects_fixed is not None and 'storage--size_indicator' in solution:
            s_indicator = solution['storage--size_indicator']
            opt_ids_s = list(s_indicator.coords['storage'].values)
            ef_s = data.storages.sizing_effects_fixed.rename({'sizing_storage': 'storage'}).sel(storage=opt_ids_s)
            stor_inv = stor_inv + (ef_s * s_indicator).reindex(storage=stor_ids, fill_value=0.0)

        # Storage sizing: fixed costs — mandatory (constant)
        if data.storages.sizing_effects_fixed is not None and data.storages.sizing_mandatory is not None:
            mand_mask_s = data.storages.sizing_mandatory.values
            if mand_mask_s.any():
                mand_ids_s = list(data.storages.sizing_mandatory.coords['sizing_storage'].values[mand_mask_s])
                ef_mand_s = data.storages.sizing_effects_fixed.sel(sizing_storage=mand_ids_s).rename(
                    {'sizing_storage': 'storage'}
                )
                stor_inv = stor_inv + ef_mand_s.reindex(storage=stor_ids, fill_value=0.0)

        # Cross-effects on storage investment
        if data.effects.cf_invest is not None:
            _apply_cross_effects_2d(stor_inv, data.effects.cf_invest, effect_ids, 'storage')

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
