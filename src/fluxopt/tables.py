from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.types import normalize_timesteps, to_data_array

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Storage
    from fluxopt.types import Timesteps


@dataclass
class ModelData:
    flows: xr.Dataset
    buses: xr.Dataset
    converters: xr.Dataset
    effects: xr.Dataset
    storages: xr.Dataset
    time: pd.Index
    dt: xr.DataArray  # (time,)
    weights: xr.DataArray  # (time,)
    time_extra: pd.Index  # N+1 times for storage charge state


def build_flows_dataset(flows: list[Flow], time: pd.Index, effects: list[Effect]) -> xr.Dataset:
    """Build flows xr.Dataset from element objects."""
    flow_ids = [f.id for f in flows]
    effect_ids = [e.id for e in effects]
    n_time = len(time)
    n_flows = len(flows)

    rel_lb = np.zeros((n_flows, n_time))
    rel_ub = np.zeros((n_flows, n_time))
    fixed_profile = np.full((n_flows, n_time), np.nan)
    size = np.full(n_flows, np.nan)
    effect_coeff = np.zeros((n_flows, len(effect_ids), n_time))

    for i, f in enumerate(flows):
        lb_da = to_data_array(f.relative_minimum, time)
        ub_da = to_data_array(f.relative_maximum, time)
        rel_lb[i] = lb_da.values
        rel_ub[i] = ub_da.values

        if f.size is not None:
            size[i] = float(f.size)

        if f.fixed_relative_profile is not None:
            profile = to_data_array(f.fixed_relative_profile, time)
            fixed_profile[i] = profile.values

        for effect_label, factor in f.effects_per_flow_hour.items():
            if effect_label in effect_ids:
                j = effect_ids.index(effect_label)
                factor_da = to_data_array(factor, time)
                effect_coeff[i, j] = factor_da.values

    # Validate
    if np.any(rel_lb < -1e-12):
        bad = [flow_ids[i] for i in range(n_flows) if np.any(rel_lb[i] < -1e-12)]
        raise ValueError(f'Negative lower bounds on flows: {bad}')
    if np.any(rel_lb > rel_ub + 1e-12):
        bad = [flow_ids[i] for i in range(n_flows) if np.any(rel_lb[i] > rel_ub[i] + 1e-12)]
        raise ValueError(f'Lower bound > upper bound on flows: {bad}')

    ds = xr.Dataset(
        {
            'rel_lb': (['flow', 'time'], rel_lb),
            'rel_ub': (['flow', 'time'], rel_ub),
            'fixed_profile': (['flow', 'time'], fixed_profile),
            'size': (['flow'], size),
            'effect_coeff': (['flow', 'effect', 'time'], effect_coeff),
        },
        coords={'flow': flow_ids, 'time': time, 'effect': effect_ids},
    )
    return ds


def build_buses_dataset(buses: list[Bus], flows: list[Flow]) -> xr.Dataset:
    """Build buses xr.Dataset with flow coefficients."""
    bus_ids = [b.id for b in buses]
    flow_ids = [f.id for f in flows]

    coeff = np.full((len(bus_ids), len(flow_ids)), np.nan)
    for f in flows:
        if f.bus in bus_ids:
            bi = bus_ids.index(f.bus)
            fi = flow_ids.index(f.id)
            coeff[bi, fi] = -1.0 if f._is_input else 1.0

    return xr.Dataset(
        {'flow_coeff': (['bus', 'flow'], coeff)},
        coords={'bus': bus_ids, 'flow': flow_ids},
    )


def build_converters_dataset(converters: list[Converter], time: pd.Index) -> xr.Dataset:
    """Build converters xr.Dataset with conversion coefficients."""
    if not converters:
        return xr.Dataset()

    conv_ids = [c.id for c in converters]
    # Collect all flow ids across all converters
    all_flow_ids: list[str] = []
    for c in converters:
        for f in c.inputs:
            if f.id not in all_flow_ids:
                all_flow_ids.append(f.id)
        for f in c.outputs:
            if f.id not in all_flow_ids:
                all_flow_ids.append(f.id)

    max_eq = max(len(c.conversion_factors) for c in converters)
    n_time = len(time)

    coeff = np.full((len(conv_ids), max_eq, len(all_flow_ids), n_time), np.nan)
    eq_mask = np.zeros((len(conv_ids), max_eq), dtype=bool)

    for ci, conv in enumerate(converters):
        for eq_idx, equation in enumerate(conv.conversion_factors):
            eq_mask[ci, eq_idx] = True
            for flow_obj, factor in equation.items():
                fi = all_flow_ids.index(flow_obj.id)
                factor_da = to_data_array(factor, time)
                coeff[ci, eq_idx, fi, :] = factor_da.values

    return xr.Dataset(
        {
            'flow_coeff': (['converter', 'eq_idx', 'flow', 'time'], coeff),
            'eq_mask': (['converter', 'eq_idx'], eq_mask),
        },
        coords={
            'converter': conv_ids,
            'eq_idx': list(range(max_eq)),
            'flow': all_flow_ids,
            'time': time,
        },
    )


def build_effects_dataset(effects: list[Effect], time: pd.Index) -> xr.Dataset:
    """Build effects xr.Dataset."""
    effect_ids = [e.id for e in effects]
    n = len(effects)
    n_time = len(time)

    # Find objective
    objective_effects = [e for e in effects if e.is_objective]
    if not objective_effects:
        raise ValueError('No objective effect found. Include an Effect with is_objective=True.')
    if len(objective_effects) > 1:
        ids = [e.id for e in objective_effects]
        raise ValueError(f'Multiple objective effects: {ids}. Only one is allowed.')
    objective_effect = objective_effects[0].id

    min_total = np.full(n, np.nan)
    max_total = np.full(n, np.nan)
    min_per_hour = np.full((n, n_time), np.nan)
    max_per_hour = np.full((n, n_time), np.nan)
    is_objective = np.zeros(n, dtype=bool)

    for i, e in enumerate(effects):
        if e.minimum_total is not None:
            min_total[i] = e.minimum_total
        if e.maximum_total is not None:
            max_total[i] = e.maximum_total
        if e.minimum_per_hour is not None:
            min_per_hour[i] = to_data_array(e.minimum_per_hour, time).values
        if e.maximum_per_hour is not None:
            max_per_hour[i] = to_data_array(e.maximum_per_hour, time).values
        is_objective[i] = e.is_objective

    ds = xr.Dataset(
        {
            'min_total': (['effect'], min_total),
            'max_total': (['effect'], max_total),
            'min_per_hour': (['effect', 'time'], min_per_hour),
            'max_per_hour': (['effect', 'time'], max_per_hour),
            'is_objective': (['effect'], is_objective),
        },
        coords={'effect': effect_ids, 'time': time},
    )
    ds.attrs['objective_effect'] = objective_effect
    return ds


def build_storages_dataset(
    storages: list[Storage], time: pd.Index, time_extra: pd.Index, dt: xr.DataArray
) -> xr.Dataset:
    """Build storages xr.Dataset."""
    if not storages:
        return xr.Dataset()

    stor_ids = [s.id for s in storages]
    n = len(storages)
    n_time = len(time)
    n_extra = len(time_extra)

    capacity = np.full(n, np.nan)
    eta_c = np.ones((n, n_time))
    eta_d = np.ones((n, n_time))
    loss = np.zeros((n, n_time))
    rel_cs_lb = np.zeros((n, n_extra))
    rel_cs_ub = np.ones((n, n_extra))
    initial_charge = np.zeros(n)
    cyclic = np.zeros(n, dtype=bool)
    charge_flow: list[str] = []
    discharge_flow: list[str] = []

    for i, s in enumerate(storages):
        if s.capacity is not None:
            capacity[i] = s.capacity

        eta_c[i] = to_data_array(s.eta_charge, time).values
        eta_d[i] = to_data_array(s.eta_discharge, time).values
        loss[i] = to_data_array(s.relative_loss_per_hour, time).values

        # Charge state bounds — broadcast to time_extra (replicate last for extra point)
        cs_lb_t = to_data_array(s.relative_minimum_charge_state, time).values
        cs_ub_t = to_data_array(s.relative_maximum_charge_state, time).values
        rel_cs_lb[i, :n_time] = cs_lb_t
        rel_cs_lb[i, n_time:] = cs_lb_t[-1]
        rel_cs_ub[i, :n_time] = cs_ub_t
        rel_cs_ub[i, n_time:] = cs_ub_t[-1]

        is_cyclic = s.initial_charge_state == 'cyclic'
        cyclic[i] = is_cyclic
        if not is_cyclic:
            initial_charge[i] = float(s.initial_charge_state) if s.initial_charge_state is not None else 0.0

        charge_flow.append(s.charging.id)
        discharge_flow.append(s.discharging.id)

    # Validate
    bad_cap = [stor_ids[i] for i in range(n) if not np.isnan(capacity[i]) and capacity[i] < 0]
    if bad_cap:
        raise ValueError(f'Negative capacity on storages: {bad_cap}')
    bad_eta_c = [stor_ids[i] for i in range(n) if np.any(eta_c[i] <= 0) or np.any(eta_c[i] > 1)]
    if bad_eta_c:
        raise ValueError(f'eta_charge must be in (0, 1] on storages: {bad_eta_c}')
    bad_eta_d = [stor_ids[i] for i in range(n) if np.any(eta_d[i] <= 0) or np.any(eta_d[i] > 1)]
    if bad_eta_d:
        raise ValueError(f'eta_discharge must be in (0, 1] on storages: {bad_eta_d}')
    bad_loss = [stor_ids[i] for i in range(n) if np.any(loss[i] < 0)]
    if bad_loss:
        raise ValueError(f'Negative relative_loss_per_hour on storages: {bad_loss}')

    return xr.Dataset(
        {
            'capacity': (['storage'], capacity),
            'eta_c': (['storage', 'time'], eta_c),
            'eta_d': (['storage', 'time'], eta_d),
            'loss': (['storage', 'time'], loss),
            'rel_cs_lb': (['storage', 'time_extra'], rel_cs_lb),
            'rel_cs_ub': (['storage', 'time_extra'], rel_cs_ub),
            'initial_charge': (['storage'], initial_charge),
            'cyclic': (['storage'], cyclic),
            'charge_flow': (['storage'], charge_flow),
            'discharge_flow': (['storage'], discharge_flow),
        },
        coords={'storage': stor_ids, 'time': time, 'time_extra': time_extra},
    )


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> list[Flow]:
    flows: list[Flow] = []
    for port in ports:
        flows.extend(port.imports)
        flows.extend(port.exports)
    for conv in converters:
        flows.extend(conv.inputs)
        flows.extend(conv.outputs)
    for s in storages or []:
        flows.append(s.charging)
        flows.append(s.discharging)
    return flows


def _validate_system(
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage],
    flows: list[Flow],
) -> None:
    """Cross-cutting validation."""
    # Unique component IDs
    all_ids: list[str] = [b.id for b in buses]
    all_ids.extend(e.id for e in effects)
    all_ids.extend(p.id for p in ports)
    all_ids.extend(c.id for c in converters)
    all_ids.extend(s.id for s in storages)
    seen: set[str] = set()
    for id_ in all_ids:
        if id_ in seen:
            raise ValueError(f'Duplicate id: {id_!r}')
        seen.add(id_)

    # Bus references
    bus_ids = {b.id for b in buses}
    for flow in flows:
        if flow.bus not in bus_ids:
            raise ValueError(f'Flow {flow.id!r} references unknown bus {flow.bus!r}')

    # Unique flow IDs
    flow_seen: set[str] = set()
    for flow in flows:
        if flow.id in flow_seen:
            raise ValueError(f'Duplicate flow id: {flow.id!r}')
        flow_seen.add(flow.id)


def _compute_time_extra(time: pd.Index, dt: xr.DataArray) -> pd.Index:
    """Compute N+1 time index for storage charge state."""
    if isinstance(time, pd.DatetimeIndex):
        last_dt_hours: float = float(dt.values[-1])
        end_time = time[-1] + timedelta(hours=last_dt_hours)
        result = time.append(pd.DatetimeIndex([end_time]))
    else:
        # Integer index
        last_val: int = int(time[-1])
        result = time.append(pd.Index([last_val + 1]))
    result.name = 'time_extra'
    return result


def build_model_data(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
) -> ModelData:
    """Build ModelData from element objects."""
    from fluxopt.types import compute_dt as _compute_dt

    converters = converters or []
    stor_list = storages or []
    time = normalize_timesteps(timesteps)
    dt_da = _compute_dt(time, dt)

    flows = _collect_flows(ports, converters, stor_list)
    _validate_system(buses, effects, ports, converters, stor_list, flows)

    time_extra = _compute_time_extra(time, dt_da)
    weights = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time}, name='weight')

    flows_ds = build_flows_dataset(flows, time, effects)
    buses_ds = build_buses_dataset(buses, flows)
    converters_ds = build_converters_dataset(converters, time)
    effects_ds = build_effects_dataset(effects, time)
    storages_ds = build_storages_dataset(stor_list, time, time_extra, dt_da)

    return ModelData(
        flows=flows_ds,
        buses=buses_ds,
        converters=converters_ds,
        effects=effects_ds,
        storages=storages_ds,
        time=time,
        dt=dt_da,
        weights=weights,
        time_extra=time_extra,
    )
