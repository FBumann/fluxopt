from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.types import as_dataarray, fast_concat, normalize_timesteps

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Sizing, Status, Storage
    from fluxopt.types import Timesteps

_NC_GROUPS = {
    'flows': 'model/flows',
    'buses': 'model/buses',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
}


def _to_dataset(obj: object) -> xr.Dataset:
    """Convert a data dataclass to an xr.Dataset.

    Args:
        obj: Dataclass with DataArray fields and scalar attrs.
    """
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
        """Validate min >= 0 and max >= min."""
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
        """Validate Sizing objects and collect into DataArrays.

        Args:
            items: Pairs of (element_id, Sizing).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
        """
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
class _StatusArrays:
    min_uptime: xr.DataArray | None = None  # (status_flow,)
    max_uptime: xr.DataArray | None = None  # (status_flow,)
    min_downtime: xr.DataArray | None = None  # (status_flow,)
    max_downtime: xr.DataArray | None = None  # (status_flow,)
    initial: xr.DataArray | None = None  # (status_flow,) — NaN = free
    effects_running: xr.DataArray | None = None  # (status_flow, effect, time)
    effects_startup: xr.DataArray | None = None  # (status_flow, effect, time)
    previous_uptime: xr.DataArray | None = None  # (status_flow,) — hours, NaN = no prior
    previous_downtime: xr.DataArray | None = None  # (status_flow,) — hours, NaN = no prior

    def __post_init__(self) -> None:
        """Validate durations >= 0 and max >= min where both given."""
        for name in ('min_uptime', 'max_uptime', 'min_downtime', 'max_downtime'):
            arr: xr.DataArray | None = getattr(self, name)
            if arr is not None:
                mask = (~np.isnan(arr)) & (arr < 0)
                if mask.any():
                    dim = arr.dims[0]
                    raise ValueError(f'Status.{name} < 0 on {list(arr.coords[dim][mask].values)}')

        if self.min_uptime is not None and self.max_uptime is not None:
            both = ~np.isnan(self.min_uptime) & ~np.isnan(self.max_uptime)
            bad = both & (self.max_uptime < self.min_uptime)
            if bad.any():
                dim = self.min_uptime.dims[0]
                raise ValueError(f'Status.max_uptime < min_uptime on {list(self.min_uptime.coords[dim][bad].values)}')

        if self.min_downtime is not None and self.max_downtime is not None:
            both = ~np.isnan(self.min_downtime) & ~np.isnan(self.max_downtime)
            bad = both & (self.max_downtime < self.min_downtime)
            if bad.any():
                dim = self.min_downtime.dims[0]
                raise ValueError(
                    f'Status.max_downtime < min_downtime on {list(self.min_downtime.coords[dim][bad].values)}'
                )

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Status]],
        effect_ids: list[str],
        time: pd.Index,
        dim: str,
        prior_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
    ) -> Self:
        """Validate Status objects and collect into DataArrays.

        Args:
            items: Pairs of (flow_id, Status).
            effect_ids: Known effect ids for validation.
            time: Time index for effect arrays.
            dim: Dimension name for the resulting arrays.
            prior_map: Flow id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
        """
        from fluxopt.constraints.status import compute_previous_duration

        if not items:
            return cls()

        prior_map = prior_map or {}
        effect_set = set(effect_ids)
        n_effects = len(effect_ids)
        n_time = len(time)

        ids: list[str] = []
        min_ups: list[float] = []
        max_ups: list[float] = []
        min_downs: list[float] = []
        max_downs: list[float] = []
        initials: list[float] = []
        prev_ups: list[float] = []
        prev_downs: list[float] = []
        er_slices: list[xr.DataArray] = []
        es_slices: list[xr.DataArray] = []

        for item_id, s in items:
            ids.append(item_id)
            min_ups.append(s.min_uptime if s.min_uptime is not None else np.nan)
            max_ups.append(s.max_uptime if s.max_uptime is not None else np.nan)
            min_downs.append(s.min_downtime if s.min_downtime is not None else np.nan)
            max_downs.append(s.max_downtime if s.max_downtime is not None else np.nan)

            prior = prior_map.get(item_id)
            if prior is not None:
                initials.append(1.0 if prior[-1] > 0 else 0.0)
                prior_da = xr.DataArray(prior, dims=['_prior_t'])
                prev_ups.append(compute_previous_duration(prior_da, target_state=1, dt=dt))
                prev_downs.append(compute_previous_duration(prior_da, target_state=0, dt=dt))
            else:
                initials.append(np.nan)
                prev_ups.append(np.nan)
                prev_downs.append(np.nan)

            # Effects per running hour — (effect, time)
            er = xr.DataArray(
                np.zeros((n_effects, n_time)),
                dims=['effect', 'time'],
                coords={'effect': effect_ids, 'time': time},
            )
            for ek, ev in s.effects_per_running_hour.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_running_hour on {item_id!r}')
                er.loc[ek] = as_dataarray(ev, {'time': time})
            er_slices.append(er)

            # Effects per startup — (effect, time)
            es = xr.DataArray(
                np.zeros((n_effects, n_time)),
                dims=['effect', 'time'],
                coords={'effect': effect_ids, 'time': time},
            )
            for ek, ev in s.effects_per_startup.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_startup on {item_id!r}')
                es.loc[ek] = as_dataarray(ev, {'time': time})
            es_slices.append(es)

        coords = {dim: ids}
        status_idx = pd.Index(ids, name=dim)

        prev_up_arr = np.array(prev_ups)
        prev_down_arr = np.array(prev_downs)

        return cls(
            min_uptime=xr.DataArray(np.array(min_ups), dims=[dim], coords=coords),
            max_uptime=xr.DataArray(np.array(max_ups), dims=[dim], coords=coords),
            min_downtime=xr.DataArray(np.array(min_downs), dims=[dim], coords=coords),
            max_downtime=xr.DataArray(np.array(max_downs), dims=[dim], coords=coords),
            initial=xr.DataArray(np.array(initials), dims=[dim], coords=coords),
            effects_running=fast_concat(er_slices, status_idx),
            effects_startup=fast_concat(es_slices, status_idx),
            previous_uptime=xr.DataArray(prev_up_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_up_arr))
            else None,
            previous_downtime=xr.DataArray(prev_down_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_down_arr))
            else None,
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
    status_min_uptime: xr.DataArray | None = None  # (status_flow,)
    status_max_uptime: xr.DataArray | None = None  # (status_flow,)
    status_min_downtime: xr.DataArray | None = None  # (status_flow,)
    status_max_downtime: xr.DataArray | None = None  # (status_flow,)
    status_initial: xr.DataArray | None = None  # (status_flow,)
    status_effects_running: xr.DataArray | None = None  # (status_flow, effect, time)
    status_effects_startup: xr.DataArray | None = None  # (status_flow, effect, time)
    status_previous_uptime: xr.DataArray | None = None  # (status_flow,)
    status_previous_downtime: xr.DataArray | None = None  # (status_flow,)

    def __post_init__(self) -> None:
        """Validate relative bounds: non-negative and lb <= ub."""
        bad_neg = (self.rel_lb < -1e-12).any('time')
        if bad_neg.any():
            raise ValueError(f'Negative lower bounds on flows: {list(self.rel_lb.coords["flow"][bad_neg].values)}')
        bad_order = (self.rel_lb > self.rel_ub + 1e-12).any('time')
        if bad_order.any():
            raise ValueError(
                f'Lower bound > upper bound on flows: {list(self.rel_lb.coords["flow"][bad_order].values)}'
            )

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with matching variable names.
        """
        kwargs: dict[str, Any] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def build(cls, flows: list[Flow], time: pd.Index, effects: list[Effect], dt: float = 1.0) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            time: Time index.
            effects: Effect definitions for cost coefficients.
            dt: Scalar timestep duration in hours for prior duration computation.
        """
        from fluxopt.elements import Sizing

        flow_ids = [f.id for f in flows]
        effect_ids = [e.id for e in effects]
        n_time = len(time)
        n_effects = len(effect_ids)

        bound_type: list[str] = []
        rel_lbs: list[xr.DataArray] = []
        rel_ubs: list[xr.DataArray] = []
        profiles: list[xr.DataArray] = []
        size_vals = np.full(len(flows), np.nan)
        effect_coeffs: list[xr.DataArray] = []
        sizing_items: list[tuple[str, Sizing]] = []
        status_items: list[tuple[str, Status]] = []
        prior_map: dict[str, list[float]] = {}

        nan_time = xr.DataArray(np.full(n_time, np.nan), dims=['time'], coords={'time': time})

        for i, f in enumerate(flows):
            rel_lbs.append(as_dataarray(f.relative_minimum, {'time': time}))
            rel_ubs.append(as_dataarray(f.relative_maximum, {'time': time}))

            if isinstance(f.size, Sizing):
                sizing_items.append((f.id, f.size))
            elif f.size is not None:
                size_vals[i] = float(f.size)

            if f.fixed_relative_profile is not None:
                profiles.append(as_dataarray(f.fixed_relative_profile, {'time': time}))
                bound_type.append('profile')
            elif f.size is None:
                profiles.append(nan_time)
                bound_type.append('unsized')
            else:
                profiles.append(nan_time)
                bound_type.append('bounded')

            # Effect coefficients for this flow
            ec = xr.DataArray(
                np.zeros((n_effects, n_time)),
                dims=['effect', 'time'],
                coords={'effect': effect_ids, 'time': time},
            )
            for effect_label, factor in f.effects_per_flow_hour.items():
                if effect_label in effect_ids:
                    ec.loc[effect_label] = as_dataarray(factor, {'time': time})
            effect_coeffs.append(ec)

            if f.status is not None:
                status_items.append((f.id, f.status))

            if f.prior is not None:
                prior_map[f.id] = f.prior

        flow_idx = pd.Index(flow_ids, name='flow')
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_flow')
        st = _StatusArrays.build(status_items, effect_ids, time, dim='status_flow', prior_map=prior_map, dt=dt)

        return cls(
            bound_type=xr.DataArray(bound_type, dims=['flow'], coords={'flow': flow_ids}),
            rel_lb=fast_concat(rel_lbs, flow_idx),
            rel_ub=fast_concat(rel_ubs, flow_idx),
            fixed_profile=fast_concat(profiles, flow_idx),
            size=xr.DataArray(size_vals, dims=['flow'], coords={'flow': flow_ids}),
            effect_coeff=fast_concat(effect_coeffs, flow_idx),
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
            status_min_uptime=st.min_uptime,
            status_max_uptime=st.max_uptime,
            status_min_downtime=st.min_downtime,
            status_max_downtime=st.max_downtime,
            status_initial=st.initial,
            status_effects_running=st.effects_running,
            status_effects_startup=st.effects_startup,
            status_previous_uptime=st.previous_uptime,
            status_previous_downtime=st.previous_downtime,
        )


@dataclass
class BusesData:
    flow_coeff: xr.DataArray  # (bus, flow)

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with ``flow_coeff`` variable.
        """
        return cls(flow_coeff=ds['flow_coeff'])

    @classmethod
    def build(cls, buses: list[Bus], flows: list[Flow], bus_coeff: dict[str, float]) -> Self:
        """Build BusesData with flow coefficients.

        Args:
            buses: Bus definitions.
            flows: All collected flows.
            bus_coeff: Mapping of flow id to +1 (produces) or -1 (consumes).
        """
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
    pair_coeff: xr.DataArray  # (pair, eq_idx, time) — non-zero coefficients only
    pair_converter: xr.DataArray  # (pair,) — converter id per pair
    pair_flow: xr.DataArray  # (pair,) — flow id per pair
    eq_mask: xr.DataArray  # (converter, eq_idx)

    @property
    def flow_coeff(self) -> xr.DataArray:
        """Dense (converter, eq_idx, flow, time) view for inspection."""
        conv_ids = list(dict.fromkeys(self.pair_converter.values))
        flow_ids = list(dict.fromkeys(self.pair_flow.values))
        eq_idx = list(self.pair_coeff.coords['eq_idx'].values)
        time = self.pair_coeff.coords['time']

        dense = xr.DataArray(
            np.full((len(conv_ids), len(eq_idx), len(flow_ids), len(time)), np.nan),
            dims=['converter', 'eq_idx', 'flow', 'time'],
            coords={'converter': conv_ids, 'eq_idx': eq_idx, 'flow': flow_ids, 'time': time},
        )
        for i in range(len(self.pair_converter)):
            conv_id = str(self.pair_converter.values[i])
            flow_id = str(self.pair_flow.values[i])
            dense.loc[conv_id, :, flow_id, :] = self.pair_coeff.isel(pair=i)
        return dense

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with pair-based converter coefficient variables.
        """
        return cls(
            pair_coeff=ds['pair_coeff'],
            pair_converter=ds['pair_converter'],
            pair_flow=ds['pair_flow'],
            eq_mask=ds['eq_mask'],
        )

    @classmethod
    def build(cls, converters: list[Converter], time: pd.Index) -> Self | None:
        """Build ConvertersData with sparse pair-based conversion coefficients.

        Args:
            converters: Converter definitions.
            time: Time index.
        """
        if not converters:
            return None

        conv_ids = [c.id for c in converters]
        max_eq = max(len(c.conversion_factors) for c in converters)
        n_time = len(time)
        eq_idx_list = list(range(max_eq))

        eq_mask_rows: list[np.ndarray] = []
        pairs_conv: list[str] = []
        pairs_flow: list[str] = []
        coeff_arrays: list[np.ndarray] = []

        for conv in converters:
            mask_row = np.zeros(max_eq, dtype=bool)
            for eq_i in range(len(conv.conversion_factors)):
                mask_row[eq_i] = True
            eq_mask_rows.append(mask_row)

            for flow in (*conv.inputs, *conv.outputs):
                eq_coeffs = np.zeros((max_eq, n_time))
                for eq_i, equation in enumerate(conv.conversion_factors):
                    if flow in equation:
                        eq_coeffs[eq_i] = as_dataarray(equation[flow], {'time': time}).values
                pairs_conv.append(conv.id)
                pairs_flow.append(flow.id)
                coeff_arrays.append(eq_coeffs)

        return cls(
            pair_coeff=xr.DataArray(
                np.array(coeff_arrays),
                dims=['pair', 'eq_idx', 'time'],
                coords={'eq_idx': eq_idx_list, 'time': time},
            ),
            pair_converter=xr.DataArray(pairs_conv, dims=['pair']),
            pair_flow=xr.DataArray(pairs_flow, dims=['pair']),
            eq_mask=xr.DataArray(
                np.array(eq_mask_rows),
                dims=['converter', 'eq_idx'],
                coords={'converter': conv_ids, 'eq_idx': eq_idx_list},
            ),
        )


def _detect_contribution_cycle(adjacency: dict[str, list[str]]) -> list[str] | None:
    """Return first cycle found in directed graph, or None.

    Args:
        adjacency: Mapping of node to list of neighbors (outgoing edges).
    """
    unvisited, in_stack, done = 0, 1, 2
    state: dict[str, int] = dict.fromkeys(adjacency, unvisited)
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = in_stack
        path.append(node)
        for neighbor in adjacency.get(node, []):
            if state[neighbor] == in_stack:
                i = path.index(neighbor)
                return [*path[i:], neighbor]
            if state[neighbor] == unvisited:
                result = dfs(neighbor)
                if result is not None:
                    return result
        path.pop()
        state[node] = done
        return None

    for node in adjacency:
        if state[node] == unvisited:
            cycle = dfs(node)
            if cycle is not None:
                return cycle
    return None


@dataclass
class EffectsData:
    min_total: xr.DataArray  # (effect,)
    max_total: xr.DataArray  # (effect,)
    min_per_hour: xr.DataArray  # (effect, time)
    max_per_hour: xr.DataArray  # (effect, time)
    is_objective: xr.DataArray  # (effect,)
    objective_effect: str
    cf_periodic: xr.DataArray | None = None  # (effect, source_effect)
    cf_temporal: xr.DataArray | None = None  # (effect, source_effect, time)

    def __post_init__(self) -> None:
        """Validate exactly one objective effect exists."""
        n_obj = int(self.is_objective.sum())
        if n_obj == 0:
            raise ValueError('No objective effect found. Include an Effect with is_objective=True.')
        if n_obj > 1:
            raise ValueError(
                f'Multiple objective effects: {list(self.is_objective.coords["effect"][self.is_objective].values)}. Only one is allowed.'
            )

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with effect variables and attrs.
        """
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name in ds.data_vars:
                kwargs[f.name] = ds[f.name]
            elif f.name in ds.attrs:
                kwargs[f.name] = ds.attrs[f.name]
            # else: rely on dataclass default (e.g. None for optional fields)
        return cls(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def build(cls, effects: list[Effect], time: pd.Index) -> Self:
        """Build EffectsData from element objects.

        Args:
            effects: Effect definitions.
            time: Time index.
        """
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)
        n = len(effects)
        n_time = len(time)
        objective_effect = next(e.id for e in effects if e.is_objective)

        min_total = np.full(n, np.nan)
        max_total = np.full(n, np.nan)
        min_per_hours: list[xr.DataArray] = []
        max_per_hours: list[xr.DataArray] = []
        is_objective = np.zeros(n, dtype=bool)

        nan_time = xr.DataArray(np.full(n_time, np.nan), dims=['time'], coords={'time': time})

        has_contributions = False
        for i, e in enumerate(effects):
            if e.minimum_total is not None:
                min_total[i] = e.minimum_total
            if e.maximum_total is not None:
                max_total[i] = e.maximum_total
            min_per_hours.append(
                as_dataarray(e.minimum_per_hour, {'time': time}) if e.minimum_per_hour is not None else nan_time
            )
            max_per_hours.append(
                as_dataarray(e.maximum_per_hour, {'time': time}) if e.maximum_per_hour is not None else nan_time
            )
            is_objective[i] = e.is_objective
            if e.contribution_from or e.contribution_from_per_hour:
                has_contributions = True

        # Build cross-effect contribution arrays
        cf_periodic: xr.DataArray | None = None
        cf_temporal: xr.DataArray | None = None
        if has_contributions:
            # Self-reference check
            for e in effects:
                for src_id in (*e.contribution_from, *e.contribution_from_per_hour):
                    if src_id == e.id:
                        raise ValueError(f'Effect {e.id!r} cannot reference itself in contribution_from')

            # Cycle check
            adjacency: dict[str, list[str]] = {eid: [] for eid in effect_ids}
            for e in effects:
                for src_id in {*e.contribution_from, *e.contribution_from_per_hour}:
                    adjacency[e.id].append(src_id)
            cycle = _detect_contribution_cycle(adjacency)
            if cycle is not None:
                raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')

            periodic_mat = np.zeros((n, n))
            temporal_mat = np.zeros((n, n, n_time))
            for i, e in enumerate(effects):
                for src_id, factor in e.contribution_from.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from on {e.id!r}')
                    j = effect_ids.index(src_id)
                    periodic_mat[i, j] = factor
                    temporal_mat[i, j, :] = factor  # default temporal = scalar
                for src_id, factor_ts in e.contribution_from_per_hour.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from_per_hour on {e.id!r}')
                    j = effect_ids.index(src_id)
                    temporal_mat[i, j, :] = as_dataarray(factor_ts, {'time': time}).values
            cf_periodic = xr.DataArray(
                periodic_mat,
                dims=['effect', 'source_effect'],
                coords={'effect': effect_ids, 'source_effect': effect_ids},
            )
            cf_temporal = xr.DataArray(
                temporal_mat,
                dims=['effect', 'source_effect', 'time'],
                coords={'effect': effect_ids, 'source_effect': effect_ids, 'time': time},
            )

        effect_idx = pd.Index(effect_ids, name='effect')

        return cls(
            min_total=xr.DataArray(min_total, dims=['effect'], coords={'effect': effect_ids}),
            max_total=xr.DataArray(max_total, dims=['effect'], coords={'effect': effect_ids}),
            min_per_hour=xr.concat(min_per_hours, dim=effect_idx),
            max_per_hour=xr.concat(max_per_hours, dim=effect_idx),
            is_objective=xr.DataArray(is_objective, dims=['effect'], coords={'effect': effect_ids}),
            objective_effect=objective_effect,
            cf_periodic=cf_periodic,
            cf_temporal=cf_temporal,
        )


@dataclass
class StoragesData:
    capacity: xr.DataArray  # (storage,)
    eta_c: xr.DataArray  # (storage, time)
    eta_d: xr.DataArray  # (storage, time)
    loss: xr.DataArray  # (storage, time)
    rel_level_lb: xr.DataArray  # (storage, time)
    rel_level_ub: xr.DataArray  # (storage, time)
    prior_level: xr.DataArray  # (storage,) — NaN if not set
    cyclic: xr.DataArray  # (storage,)
    charge_flow: xr.DataArray  # (storage,) — str
    discharge_flow: xr.DataArray  # (storage,) — str
    sizing_min: xr.DataArray | None = None  # (sizing_storage,)
    sizing_max: xr.DataArray | None = None  # (sizing_storage,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_storage,)
    sizing_effects_per_size: xr.DataArray | None = None  # (sizing_storage, effect)
    sizing_effects_fixed: xr.DataArray | None = None  # (sizing_storage, effect)

    def __post_init__(self) -> None:
        """Validate capacity, efficiencies, and loss rates."""
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
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with matching variable names.
        """
        kwargs: dict[str, Any] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def build(
        cls,
        storages: list[Storage],
        time: pd.Index,
        dt: xr.DataArray,
        effects: list[Effect] | None = None,
    ) -> Self | None:
        """Build StoragesData from element objects.

        Args:
            storages: Storage definitions.
            time: Time index.
            dt: Timestep durations.
            effects: Effect definitions for sizing cost validation.
        """
        from fluxopt.elements import Sizing

        if not storages:
            return None

        effect_ids = [e.id for e in effects] if effects else []
        stor_ids = [s.id for s in storages]
        n = len(storages)

        capacity_vals = np.full(n, np.nan)
        eta_cs: list[xr.DataArray] = []
        eta_ds: list[xr.DataArray] = []
        losses: list[xr.DataArray] = []
        level_lbs: list[xr.DataArray] = []
        level_ubs: list[xr.DataArray] = []
        prior_level_vals = np.full(n, np.nan)
        cyclic_vals = np.zeros(n, dtype=bool)
        charge_flow: list[str] = []
        discharge_flow: list[str] = []
        sizing_items: list[tuple[str, Sizing]] = []

        for i, s in enumerate(storages):
            if isinstance(s.capacity, Sizing):
                sizing_items.append((s.id, s.capacity))
            elif s.capacity is not None:
                capacity_vals[i] = s.capacity

            eta_cs.append(as_dataarray(s.eta_charge, {'time': time}))
            eta_ds.append(as_dataarray(s.eta_discharge, {'time': time}))
            losses.append(as_dataarray(s.relative_loss_per_hour, {'time': time}))

            level_lbs.append(as_dataarray(s.relative_minimum_level, {'time': time}))
            level_ubs.append(as_dataarray(s.relative_maximum_level, {'time': time}))

            cyclic_vals[i] = s.cyclic
            if s.prior_level is not None:
                prior_level_vals[i] = s.prior_level

            charge_flow.append(s.charging.id)
            discharge_flow.append(s.discharging.id)

        stor_idx = pd.Index(stor_ids, name='storage')
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_storage')

        return cls(
            capacity=xr.DataArray(capacity_vals, dims=['storage'], coords={'storage': stor_ids}),
            eta_c=xr.concat(eta_cs, dim=stor_idx),
            eta_d=xr.concat(eta_ds, dim=stor_idx),
            loss=xr.concat(losses, dim=stor_idx),
            rel_level_lb=xr.concat(level_lbs, dim=stor_idx),
            rel_level_ub=xr.concat(level_ubs, dim=stor_idx),
            prior_level=xr.DataArray(prior_level_vals, dims=['storage'], coords={'storage': stor_ids}),
            cyclic=xr.DataArray(cyclic_vals, dims=['storage'], coords={'storage': stor_ids}),
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

    def to_netcdf(self, path: str | Path, *, mode: Literal['w', 'a'] = 'a') -> None:
        """Write model data as NetCDF groups under ``/model/``.

        Args:
            path: Output file path.
            mode: Write mode ('w' to overwrite, 'a' to append).
        """
        p = Path(path)
        dataset_fields: dict[str, FlowsData | BusesData | ConvertersData | EffectsData | StoragesData | None] = {
            'flows': self.flows,
            'buses': self.buses,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
        }
        current_mode = mode
        for name, obj in dataset_fields.items():
            if obj is not None:
                obj.to_dataset().to_netcdf(p, mode=current_mode, group=_NC_GROUPS[name], engine='netcdf4')
                current_mode = 'a'
        meta = xr.Dataset({'dt': self.dt, 'weights': self.weights})
        meta.to_netcdf(p, mode=current_mode, group='model/meta', engine='netcdf4')

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ModelData | None:
        """Read model data from NetCDF groups.

        Args:
            path: Input file path.

        Returns:
            ModelData or None if no model groups found.
        """
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

        return cls(
            flows=flows,
            buses=buses,
            converters=converters,
            effects=effects,
            storages=storages,
            dt=dt,
            weights=meta['weights'],
            time=time,
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
        """Build ModelData from element objects.

        Args:
            timesteps: Time index for the optimization horizon.
            buses: Energy buses.
            effects: Effects to track.
            ports: System boundary ports.
            converters: Linear converters.
            storages: Energy storages.
            dt: Timestep duration in hours. Auto-derived if None.
        """
        from fluxopt.types import compute_dt as _compute_dt

        converters = converters or []
        stor_list = storages or []
        time = normalize_timesteps(timesteps)
        dt_da = _compute_dt(time, dt)

        flows, bus_coeff = _collect_flows(ports, converters, stor_list)
        _validate_system(buses, effects, ports, converters, stor_list, flows)

        weights = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time}, name='weight')

        # Scalar dt for prior duration computation (use first timestep)
        dt_scalar = float(dt_da.values[0])
        flows_data = FlowsData.build(flows, time, effects, dt=dt_scalar)
        buses_data = BusesData.build(buses, flows, bus_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time)
        storages_data = StoragesData.build(stor_list, time, dt_da, effects)

        return cls(
            flows=flows_data,
            buses=buses_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dt=dt_da,
            weights=weights,
            time=time,
        )


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> tuple[list[Flow], dict[str, float]]:
    """Gather all flows and assign bus-balance coefficients by direction.

    Args:
        ports: System boundary ports.
        converters: Converter components.
        storages: Storage components.

    Returns:
        Tuple of (flows, bus_coeff) where bus_coeff maps flow id to
        +1 (produces into bus) or -1 (consumes from bus).
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
    """Validate unique ids and bus references across all elements.

    Args:
        buses: Bus definitions.
        effects: Effect definitions.
        ports: Port components.
        converters: Converter components.
        storages: Storage components.
        flows: All collected flows.
    """
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
