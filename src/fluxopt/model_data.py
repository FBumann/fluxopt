from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.types import normalize_timesteps, to_data_array

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Sizing, Storage
    from fluxopt.types import Timesteps

_NC_GROUPS = {
    'flows': 'model/flows',
    'buses': 'model/buses',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
}


def _to_dataset(obj: object) -> xr.Dataset:
    """Convert a typed data dataclass to an xr.Dataset."""
    data_vars: dict[str, xr.DataArray] = {}
    attrs: dict[str, object] = {}
    for f in fields(obj):  # type: ignore[arg-type]
        val = getattr(obj, f.name)
        if val is None:
            continue
        if isinstance(val, xr.DataArray):
            data_vars[f.name] = val
        else:
            attrs[f.name] = val
    ds = xr.Dataset(data_vars)
    ds.attrs.update(attrs)
    return ds


@dataclass
class _SizingArrays:
    min: xr.DataArray | None = None
    max: xr.DataArray | None = None
    mandatory: xr.DataArray | None = None
    effects_per_size: xr.DataArray | None = None
    effects_fixed: xr.DataArray | None = None

    def __post_init__(self) -> None:
        if self.min is not None:
            mask = self.min < 0
            if mask.any():
                raise ValueError(f'Sizing.min_size < 0 on {list(self.min.coords[self.min.dims[0]][mask].values)}')
        if self.min is not None and self.max is not None:
            mask = self.max < self.min
            if mask.any():
                dim = self.min.dims[0]
                raise ValueError(f'Sizing.max_size < min_size on {list(self.min.coords[dim][mask].values)}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Sizing]],
        effect_ids: list[str],
        dim: str,
    ) -> Self:
        """Validate Sizing objects and collect into DataArrays."""
        if not items:
            return cls()

        effect_set = set(effect_ids)
        n_effects = len(effect_ids)

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        eps_rows: list[np.ndarray] = []
        ef_rows: list[np.ndarray] = []

        for item_id, s in items:
            ids.append(item_id)
            mins.append(s.min_size)
            maxs.append(s.max_size)
            mandatories.append(s.mandatory)
            eps_row = np.zeros(n_effects)
            ef_row = np.zeros(n_effects)
            for ek, ev in s.effects_per_size.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_per_size on {item_id!r}')
                eps_row[effect_ids.index(ek)] = ev
            for ek, ev in s.effects_fixed.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_fixed on {item_id!r}')
                ef_row[effect_ids.index(ek)] = ev
            eps_rows.append(eps_row)
            ef_rows.append(ef_row)

        coords = {dim: ids}
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            effects_per_size=xr.DataArray(
                np.array(eps_rows), dims=[dim, 'effect'], coords={dim: ids, 'effect': effect_ids}
            ),
            effects_fixed=xr.DataArray(
                np.array(ef_rows), dims=[dim, 'effect'], coords={dim: ids, 'effect': effect_ids}
            ),
        )


@dataclass
class FlowsData:
    bound_type: xr.DataArray  # (flow,) — 'unsized' | 'bounded' | 'profile'
    rel_lb: xr.DataArray  # (flow, time)
    rel_ub: xr.DataArray  # (flow, time)
    fixed_profile: xr.DataArray  # (flow, time) — NaN where not fixed
    size: xr.DataArray  # (flow,) — NaN for unsized
    effect_coeff: xr.DataArray  # (flow, effect, time)
    sizing_min: xr.DataArray | None = None  # (sizing_flow,)
    sizing_max: xr.DataArray | None = None  # (sizing_flow,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_flow,)
    sizing_effects_per_size: xr.DataArray | None = None  # (sizing_flow, effect)
    sizing_effects_fixed: xr.DataArray | None = None  # (sizing_flow, effect)

    def __post_init__(self) -> None:
        bad_neg = (self.rel_lb < -1e-12).any('time')
        if bad_neg.any():
            raise ValueError(f'Negative lower bounds on flows: {list(self.rel_lb.coords["flow"][bad_neg].values)}')
        bad_order = (self.rel_lb > self.rel_ub + 1e-12).any('time')
        if bad_order.any():
            raise ValueError(
                f'Lower bound > upper bound on flows: {list(self.rel_lb.coords["flow"][bad_order].values)}'
            )

    def to_dataset(self) -> xr.Dataset:
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        kwargs: dict[str, xr.DataArray | None] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def build(cls, flows: list[Flow], time: pd.Index, effects: list[Effect]) -> Self:
        """Build FlowsData from element objects."""
        from fluxopt.elements import Sizing

        flow_ids = [f.id for f in flows]
        effect_ids = [e.id for e in effects]
        n_time = len(time)
        n_flows = len(flows)
        n_effects = len(effect_ids)

        bound_type: list[str] = []
        rel_lb = np.zeros((n_flows, n_time))
        rel_ub = np.zeros((n_flows, n_time))
        fixed_profile = np.full((n_flows, n_time), np.nan)
        size = np.full(n_flows, np.nan)
        effect_coeff = np.zeros((n_flows, n_effects, n_time))
        sizing_items: list[tuple[str, Sizing]] = []

        for i, f in enumerate(flows):
            lb_da = to_data_array(f.relative_minimum, time)
            ub_da = to_data_array(f.relative_maximum, time)
            rel_lb[i] = lb_da.values
            rel_ub[i] = ub_da.values

            if isinstance(f.size, Sizing):
                sizing_items.append((f.id, f.size))
            elif f.size is not None:
                size[i] = float(f.size)

            if f.fixed_relative_profile is not None:
                profile = to_data_array(f.fixed_relative_profile, time)
                fixed_profile[i] = profile.values
                bound_type.append('profile')
            elif f.size is None:
                bound_type.append('unsized')
            else:
                bound_type.append('bounded')

            for effect_label, factor in f.effects_per_flow_hour.items():
                if effect_label in effect_ids:
                    j = effect_ids.index(effect_label)
                    factor_da = to_data_array(factor, time)
                    effect_coeff[i, j] = factor_da.values

        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_flow')

        return cls(
            bound_type=xr.DataArray(bound_type, dims=['flow'], coords={'flow': flow_ids}),
            rel_lb=xr.DataArray(rel_lb, dims=['flow', 'time'], coords={'flow': flow_ids, 'time': time}),
            rel_ub=xr.DataArray(rel_ub, dims=['flow', 'time'], coords={'flow': flow_ids, 'time': time}),
            fixed_profile=xr.DataArray(fixed_profile, dims=['flow', 'time'], coords={'flow': flow_ids, 'time': time}),
            size=xr.DataArray(size, dims=['flow'], coords={'flow': flow_ids}),
            effect_coeff=xr.DataArray(
                effect_coeff,
                dims=['flow', 'effect', 'time'],
                coords={'flow': flow_ids, 'effect': effect_ids, 'time': time},
            ),
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
        )


@dataclass
class BusesData:
    flow_coeff: xr.DataArray  # (bus, flow)

    def to_dataset(self) -> xr.Dataset:
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        return cls(flow_coeff=ds['flow_coeff'])

    @classmethod
    def build(cls, buses: list[Bus], flows: list[Flow], bus_coeff: dict[str, float]) -> Self:
        """Build BusesData with flow coefficients."""
        bus_ids = [b.id for b in buses]
        flow_ids = [f.id for f in flows]

        coeff = np.full((len(bus_ids), len(flow_ids)), np.nan)
        for f in flows:
            if f.bus in bus_ids:
                bi = bus_ids.index(f.bus)
                fi = flow_ids.index(f.id)
                coeff[bi, fi] = bus_coeff[f.id]

        return cls(
            flow_coeff=xr.DataArray(coeff, dims=['bus', 'flow'], coords={'bus': bus_ids, 'flow': flow_ids}),
        )


@dataclass
class ConvertersData:
    flow_coeff: xr.DataArray  # (converter, eq_idx, flow, time)
    eq_mask: xr.DataArray  # (converter, eq_idx)

    def to_dataset(self) -> xr.Dataset:
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        return cls(flow_coeff=ds['flow_coeff'], eq_mask=ds['eq_mask'])

    @classmethod
    def build(cls, converters: list[Converter], time: pd.Index) -> Self | None:
        """Build ConvertersData with conversion coefficients."""
        if not converters:
            return None

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

        return cls(
            flow_coeff=xr.DataArray(
                coeff,
                dims=['converter', 'eq_idx', 'flow', 'time'],
                coords={
                    'converter': conv_ids,
                    'eq_idx': list(range(max_eq)),
                    'flow': all_flow_ids,
                    'time': time,
                },
            ),
            eq_mask=xr.DataArray(
                eq_mask,
                dims=['converter', 'eq_idx'],
                coords={'converter': conv_ids, 'eq_idx': list(range(max_eq))},
            ),
        )


@dataclass
class EffectsData:
    min_total: xr.DataArray  # (effect,)
    max_total: xr.DataArray  # (effect,)
    min_per_hour: xr.DataArray  # (effect, time)
    max_per_hour: xr.DataArray  # (effect, time)
    is_objective: xr.DataArray  # (effect,)
    objective_effect: str

    def __post_init__(self) -> None:
        n_obj = int(self.is_objective.sum())
        if n_obj == 0:
            raise ValueError('No objective effect found. Include an Effect with is_objective=True.')
        if n_obj > 1:
            raise ValueError(
                f'Multiple objective effects: {list(self.is_objective.coords["effect"][self.is_objective].values)}. Only one is allowed.'
            )

    def to_dataset(self) -> xr.Dataset:
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        kwargs: dict[str, object] = {
            f.name: ds[f.name] if f.name in ds.data_vars else ds.attrs[f.name] for f in fields(cls)
        }
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def build(cls, effects: list[Effect], time: pd.Index) -> Self:
        """Build EffectsData."""
        effect_ids = [e.id for e in effects]
        n = len(effects)
        n_time = len(time)
        objective_effect = next(e.id for e in effects if e.is_objective)

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

        return cls(
            min_total=xr.DataArray(min_total, dims=['effect'], coords={'effect': effect_ids}),
            max_total=xr.DataArray(max_total, dims=['effect'], coords={'effect': effect_ids}),
            min_per_hour=xr.DataArray(
                min_per_hour, dims=['effect', 'time'], coords={'effect': effect_ids, 'time': time}
            ),
            max_per_hour=xr.DataArray(
                max_per_hour, dims=['effect', 'time'], coords={'effect': effect_ids, 'time': time}
            ),
            is_objective=xr.DataArray(is_objective, dims=['effect'], coords={'effect': effect_ids}),
            objective_effect=objective_effect,
        )


@dataclass
class StoragesData:
    capacity: xr.DataArray  # (storage,)
    eta_c: xr.DataArray  # (storage, time)
    eta_d: xr.DataArray  # (storage, time)
    loss: xr.DataArray  # (storage, time)
    rel_cs_lb: xr.DataArray  # (storage, time_extra)
    rel_cs_ub: xr.DataArray  # (storage, time_extra)
    initial_charge: xr.DataArray  # (storage,)
    cyclic: xr.DataArray  # (storage,)
    charge_flow: xr.DataArray  # (storage,) — str
    discharge_flow: xr.DataArray  # (storage,) — str
    sizing_min: xr.DataArray | None = None  # (sizing_storage,)
    sizing_max: xr.DataArray | None = None  # (sizing_storage,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_storage,)
    sizing_effects_per_size: xr.DataArray | None = None  # (sizing_storage, effect)
    sizing_effects_fixed: xr.DataArray | None = None  # (sizing_storage, effect)

    def __post_init__(self) -> None:
        s = self.capacity.coords['storage']
        cap = self.capacity
        bad_cap = ~np.isnan(cap) & (cap < 0)
        if bad_cap.any():
            raise ValueError(f'Negative capacity on storages: {list(s[bad_cap].values)}')
        bad_eta_c = ((self.eta_c <= 0) | (self.eta_c > 1)).any('time')
        if bad_eta_c.any():
            raise ValueError(f'eta_charge must be in (0, 1] on storages: {list(s[bad_eta_c].values)}')
        bad_eta_d = ((self.eta_d <= 0) | (self.eta_d > 1)).any('time')
        if bad_eta_d.any():
            raise ValueError(f'eta_discharge must be in (0, 1] on storages: {list(s[bad_eta_d].values)}')
        bad_loss = (self.loss < 0).any('time')
        if bad_loss.any():
            raise ValueError(f'Negative relative_loss_per_hour on storages: {list(s[bad_loss].values)}')

    def to_dataset(self) -> xr.Dataset:
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        kwargs: dict[str, xr.DataArray | None] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def build(
        cls,
        storages: list[Storage],
        time: pd.Index,
        time_extra: pd.Index,
        dt: xr.DataArray,
        effects: list[Effect] | None = None,
    ) -> Self | None:
        """Build StoragesData."""
        from fluxopt.elements import Sizing

        if not storages:
            return None

        effect_ids = [e.id for e in effects] if effects else []
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
        sizing_items: list[tuple[str, Sizing]] = []

        for i, s in enumerate(storages):
            if isinstance(s.capacity, Sizing):
                sizing_items.append((s.id, s.capacity))
            elif s.capacity is not None:
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

        coords_time: dict[str, object] = {'storage': stor_ids, 'time': time}
        coords_extra: dict[str, object] = {'storage': stor_ids, 'time_extra': time_extra}
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_storage')

        return cls(
            capacity=xr.DataArray(capacity, dims=['storage'], coords={'storage': stor_ids}),
            eta_c=xr.DataArray(eta_c, dims=['storage', 'time'], coords=coords_time),
            eta_d=xr.DataArray(eta_d, dims=['storage', 'time'], coords=coords_time),
            loss=xr.DataArray(loss, dims=['storage', 'time'], coords=coords_time),
            rel_cs_lb=xr.DataArray(rel_cs_lb, dims=['storage', 'time_extra'], coords=coords_extra),
            rel_cs_ub=xr.DataArray(rel_cs_ub, dims=['storage', 'time_extra'], coords=coords_extra),
            initial_charge=xr.DataArray(initial_charge, dims=['storage'], coords={'storage': stor_ids}),
            cyclic=xr.DataArray(cyclic, dims=['storage'], coords={'storage': stor_ids}),
            charge_flow=xr.DataArray(charge_flow, dims=['storage'], coords={'storage': stor_ids}),
            discharge_flow=xr.DataArray(discharge_flow, dims=['storage'], coords={'storage': stor_ids}),
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
        )


@dataclass
class ModelData:
    flows: FlowsData
    buses: BusesData
    converters: ConvertersData | None  # None when no converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dt: xr.DataArray  # (time,)
    weights: xr.DataArray  # (time,)
    time: pd.Index = field(repr=False)
    time_extra: pd.Index = field(repr=False)

    def to_netcdf(self, path: str | Path, *, mode: Literal['w', 'a'] = 'a') -> None:
        """Write model data as NetCDF groups under /model/."""
        p = Path(path)
        dataset_fields: dict[str, FlowsData | BusesData | ConvertersData | EffectsData | StoragesData | None] = {
            'flows': self.flows,
            'buses': self.buses,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
        }
        for name, obj in dataset_fields.items():
            if obj is not None:
                obj.to_dataset().to_netcdf(p, mode=mode, group=_NC_GROUPS[name], engine='netcdf4')
        meta = xr.Dataset({'dt': self.dt, 'weights': self.weights})
        meta.to_netcdf(p, mode='a', group='model/meta', engine='netcdf4')

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ModelData | None:
        """Read model data from NetCDF groups. Returns None if not present."""
        p = Path(path)
        try:
            meta = xr.open_dataset(p, group='model/meta', engine='netcdf4')
        except OSError:
            return None

        datasets: dict[str, xr.Dataset] = {}
        for name, group in _NC_GROUPS.items():
            try:
                datasets[name] = xr.open_dataset(p, group=group, engine='netcdf4')
            except OSError:
                datasets[name] = xr.Dataset()

        dt = meta['dt']
        time = pd.Index(dt.coords['time'].values)

        flows = FlowsData.from_dataset(datasets['flows'])
        buses = BusesData.from_dataset(datasets['buses'])
        converters = ConvertersData.from_dataset(datasets['converters']) if datasets['converters'].data_vars else None
        effects = EffectsData.from_dataset(datasets['effects'])
        storages = StoragesData.from_dataset(datasets['storages']) if datasets['storages'].data_vars else None

        if storages is not None and 'time_extra' in storages.rel_cs_lb.coords:
            time_extra = pd.Index(storages.rel_cs_lb.coords['time_extra'].values)
            time_extra.name = 'time_extra'
        else:
            time_extra = _compute_time_extra(time, dt)

        return cls(
            flows=flows,
            buses=buses,
            converters=converters,
            effects=effects,
            storages=storages,
            dt=dt,
            weights=meta['weights'],
            time=time,
            time_extra=time_extra,
        )

    @classmethod
    def build(
        cls,
        timesteps: Timesteps,
        buses: list[Bus],
        effects: list[Effect],
        ports: list[Port],
        converters: list[Converter] | None = None,
        storages: list[Storage] | None = None,
        dt: float | list[float] | None = None,
    ) -> Self:
        """Build ModelData from element objects."""
        from fluxopt.types import compute_dt as _compute_dt

        converters = converters or []
        stor_list = storages or []
        time = normalize_timesteps(timesteps)
        dt_da = _compute_dt(time, dt)

        flows, bus_coeff = _collect_flows(ports, converters, stor_list)
        _validate_system(buses, effects, ports, converters, stor_list, flows)

        time_extra = _compute_time_extra(time, dt_da)
        weights = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time}, name='weight')

        flows_data = FlowsData.build(flows, time, effects)
        buses_data = BusesData.build(buses, flows, bus_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time)
        storages_data = StoragesData.build(stor_list, time, time_extra, dt_da, effects)

        return cls(
            flows=flows_data,
            buses=buses_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dt=dt_da,
            weights=weights,
            time=time,
            time_extra=time_extra,
        )


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> tuple[list[Flow], dict[str, float]]:
    """Gather all flows and assign bus-balance coefficients by direction.

    Returns (flows, bus_coeff) where bus_coeff maps flow id → +1 (produces
    into bus) or -1 (consumes from bus).
    """
    flows: list[Flow] = []
    bus_coeff: dict[str, float] = {}
    for port in ports:
        for f in port.imports:
            flows.append(f)
            bus_coeff[f.id] = 1.0  # imports add energy to bus
        for f in port.exports:
            flows.append(f)
            bus_coeff[f.id] = -1.0  # exports take energy from bus
    for conv in converters:
        for f in conv.inputs:
            flows.append(f)
            bus_coeff[f.id] = -1.0  # converter consumes from bus
        for f in conv.outputs:
            flows.append(f)
            bus_coeff[f.id] = 1.0  # converter produces to bus
    for s in storages or []:
        flows.append(s.charging)
        bus_coeff[s.charging.id] = -1.0  # charging takes from bus
        flows.append(s.discharging)
        bus_coeff[s.discharging.id] = 1.0  # discharging adds to bus
    return flows, bus_coeff


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
