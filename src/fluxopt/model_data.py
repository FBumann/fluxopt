from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, get_args

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.constraints.episodes import Episodes
from fluxopt.contract import BoundType, Dim
from fluxopt.types import PiecewiseMethod, as_dataarray, fast_concat, normalize_timesteps
from fluxopt.validation import validate_system

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage, _BoundFlow
    from fluxopt.types import TimeIndex, Timesteps, Variate


@dataclass(frozen=True)
class _EffectTemplate:
    """Pre-computed shape/dims/coords for an effect-dimensioned zero array."""

    shape: tuple[int, ...]
    dims: tuple[str, ...]
    coords: dict[str, Any]
    as_da_coords: dict[str, Any]

    def zeros(self) -> xr.DataArray:
        """Create a zero-filled DataArray with this template's shape."""
        return xr.DataArray(np.zeros(self.shape), dims=list(self.dims), coords=self.coords)


def _effect_template(
    base_dims: dict[str, Any],
    period: pd.Index | None = None,
) -> _EffectTemplate:
    """Build template shape/dims/coords for effect arrays with optional period.

    Args:
        base_dims: Ordered mapping of dim_name -> coord_values.
        period: Period index to append as trailing dimension.
    """
    dims = list(base_dims.keys())
    coords = dict(base_dims)
    shape = [len(v) for v in base_dims.values()]
    as_da_coords: dict[str, Any] = {k: v for k, v in base_dims.items() if k not in ('effect', 'source_effect')}

    if period is not None:
        dims.append('period')
        coords['period'] = period
        shape.append(len(period))
        as_da_coords['period'] = period

    return _EffectTemplate(
        shape=tuple(shape),
        dims=tuple(dims),
        coords=coords,
        as_da_coords=as_da_coords,
    )


_NC_GROUPS = {
    'flows': 'model/flows',
    'carriers': 'model/carriers',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
    'piecewise': 'model/pw',
}


def _raise_netcdf_read_error(path: Path, exc: OSError) -> NoReturn:
    """Re-raise a netCDF read failure, clarifying the Windows non-ASCII path bug.

    netcdf4/libnetcdf (through 4.9.3) fails to open files under non-ASCII
    *directories* on Windows with a misleading ``PermissionError``/``OSError``
    (upstream bug Unidata/netcdf4-python#1482). When the failing path is
    non-ASCII on Windows we surface an actionable message; otherwise the original
    error propagates unchanged. Only read paths are wrapped — the error only
    surfaces if netcdf4 actually raises, so nothing that would work is blocked.

    Args:
        path: The path being read.
        exc: The error raised by the netCDF engine.

    Raises:
        ValueError: On Windows when the failing path contains non-ASCII characters.
        OSError: The original error, on any other platform or path.
    """
    if os.name == 'nt' and not str(path).isascii():
        raise ValueError(
            f'Failed to read netCDF at a path containing non-ASCII characters on Windows: {path}. '
            'netcdf4 cannot open files under non-ASCII directories on Windows '
            '(upstream bug Unidata/netcdf4-python#1482). Use an ASCII-only directory and file name.'
        ) from exc
    raise exc


def _to_dataset(obj: DataclassInstance) -> xr.Dataset:
    """Convert a data dataclass to an xr.Dataset.

    Args:
        obj: Dataclass with DataArray fields and scalar attrs.
    """
    data_vars: dict[str, xr.DataArray] = {}
    attrs: dict[str, object] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is None or is_dataclass(val):
            continue  # nested container fields serialize as their own netCDF sub-group
        if isinstance(val, xr.DataArray):
            data_vars[f.name] = val
        else:
            attrs[f.name] = val
    ds = xr.Dataset(data_vars)
    ds.attrs.update(attrs)
    return ds


# Nested container fields on FlowsData / StoragesData — serialized as netCDF
# sub-groups, not variables in the parent table's Dataset.
_CONTAINER_FIELD_NAMES = frozenset({'sizing', 'status', 'invest', 'cstatus'})


def _container_from_dataset[T: DataclassInstance](cls: type[T], ds: xr.Dataset) -> T:
    """Rebuild a nested container dataclass from its own Dataset node.

    Every field is a plain ``xr.DataArray | None``; required fields are always
    present in *ds*, optional ones fall back to ``None`` when absent.
    """
    return cls(**{f.name: ds.get(f.name) for f in fields(cls)})


@dataclass
class SizingData:
    """Sizing (capacity optimization) arrays for one entity family."""

    min: xr.DataArray  # (dim,)
    max: xr.DataArray  # (dim,)
    mandatory: xr.DataArray  # (dim,)
    effects_per_size: xr.DataArray  # (dim, effect, period?) — dense, zero where absent
    effects_fixed: xr.DataArray  # (dim, effect, period?) — dense, zero where absent

    def __post_init__(self) -> None:
        """Validate min >= 0 and max >= min."""
        mask = self.min < 0
        if mask.any():
            raise ValueError(f'Sizing.size_min < 0 on {list(self.min.coords[self.min.dims[0]][mask].values)}')
        mask = self.max < self.min
        if mask.any():
            dim = self.min.dims[0]
            raise ValueError(f'Sizing.size_max < size_min on {list(self.min.coords[dim][mask].values)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset."""
        return _container_from_dataset(cls, ds)

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Sizing]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Validate Sizing objects and collect into DataArrays, or None if empty.

        Args:
            items: Pairs of (element_id, Sizing).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            period: Period index for period-varying effects.
        """
        if not items:
            return None

        effect_set = set(effect_ids)
        tmpl = _effect_template({'effect': effect_ids}, period)

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        eps_slices: list[xr.DataArray] = []
        ef_slices: list[xr.DataArray] = []

        for item_id, s in items:
            ids.append(item_id)
            mins.append(s.size_min)
            maxs.append(s.size_max)
            mandatories.append(s.mandatory)

            eps = tmpl.zeros()
            ef = tmpl.zeros()
            for ek, ev in s.effects_per_size.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_per_size on {item_id!r}')
                eps.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            for ek, ev in s.effects_fixed.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_fixed on {item_id!r}')
                ef.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            eps_slices.append(eps)
            ef_slices.append(ef)

        coords = {dim: ids}
        sizing_idx = pd.Index(ids, name=dim)
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            effects_per_size=fast_concat(eps_slices, sizing_idx),
            effects_fixed=fast_concat(ef_slices, sizing_idx),
        )


@dataclass
class InvestmentData:
    """Investment (build-timing optimization) arrays for one entity family."""

    min: xr.DataArray  # (invest_dim,)
    max: xr.DataArray  # (invest_dim,)
    mandatory: xr.DataArray  # (invest_dim,)
    lifetime: xr.DataArray  # (invest_dim,) — NaN = forever
    prior_size: xr.DataArray  # (invest_dim,)
    effects_per_size_at_build: xr.DataArray  # (invest_dim, effect, period?) — once
    effects_fixed_at_build: xr.DataArray  # (invest_dim, effect, period?) — once
    effects_per_size_recurring: xr.DataArray  # (invest_dim, effect, period?)
    effects_fixed_recurring: xr.DataArray  # (invest_dim, effect, period?)

    def __post_init__(self) -> None:
        """Validate size bounds, prior size, and lifetime (also on netCDF reload)."""
        dim = self.min.dims[0]
        ids = self.min.coords[dim]
        if (mask := self.min < 0).any():
            raise ValueError(f'Investment.size_min < 0 on {list(ids[mask].values)}')
        if (mask := self.max < self.min).any():
            raise ValueError(f'Investment.size_max < size_min on {list(ids[mask].values)}')
        if (mask := self.prior_size < 0).any():
            raise ValueError(f'Investment.prior_size < 0 on {list(ids[mask].values)}')
        if (mask := self.lifetime <= 0).any():
            raise ValueError(f'Investment.lifetime must be positive on {list(ids[mask].values)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset."""
        return _container_from_dataset(cls, ds)

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Investment]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Validate Investment objects and collect into DataArrays, or None if empty.

        Args:
            items: Pairs of (element_id, Investment).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            period: Period index for period-varying effects.
        """
        if not items:
            return None

        effect_set = set(effect_ids)
        tmpl = _effect_template({'effect': effect_ids}, period)

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        lifetimes: list[float] = []
        prior_sizes: list[float] = []
        all_slices: dict[str, list[xr.DataArray]] = {
            'eps': [],
            'ef': [],
            'eps_p': [],
            'ef_p': [],
        }

        for item_id, inv in items:
            ids.append(item_id)
            mins.append(inv.size_min)
            maxs.append(inv.size_max)
            mandatories.append(inv.mandatory)
            lifetimes.append(float(inv.lifetime) if inv.lifetime is not None else np.nan)
            prior_sizes.append(inv.prior_size)

            for label, src_dict, dest_key in [
                ('Investment.effects_per_size_at_build', inv.effects_per_size_at_build, 'eps'),
                ('Investment.effects_fixed_at_build', inv.effects_fixed_at_build, 'ef'),
                ('Investment.effects_per_size_recurring', inv.effects_per_size_recurring, 'eps_p'),
                ('Investment.effects_fixed_recurring', inv.effects_fixed_recurring, 'ef_p'),
            ]:
                arr = tmpl.zeros()
                for ek, ev in src_dict.items():
                    if ek not in effect_set:
                        raise ValueError(f'Unknown effect {ek!r} in {label} on {item_id!r}')
                    arr.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
                all_slices[dest_key].append(arr)

        coords = {dim: ids}
        invest_idx = pd.Index(ids, name=dim)
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            lifetime=xr.DataArray(np.array(lifetimes), dims=[dim], coords=coords),
            prior_size=xr.DataArray(np.array(prior_sizes), dims=[dim], coords=coords),
            effects_per_size_at_build=fast_concat(all_slices['eps'], invest_idx),
            effects_fixed_at_build=fast_concat(all_slices['ef'], invest_idx),
            effects_per_size_recurring=fast_concat(all_slices['eps_p'], invest_idx),
            effects_fixed_recurring=fast_concat(all_slices['ef_p'], invest_idx),
        )


@dataclass
class StatusData:
    """Binary on/off behavior arrays for one entity family (flow or component)."""

    uptime_min: xr.DataArray  # (dim,)
    uptime_max: xr.DataArray  # (dim,)
    downtime_min: xr.DataArray  # (dim,)
    downtime_max: xr.DataArray  # (dim,)
    initial: xr.DataArray  # (dim,) — NaN = free
    effects_running: xr.DataArray  # (dim, effect, time) — dense, zero where absent
    effects_startup: xr.DataArray  # (dim, effect, time) — dense, zero where absent
    previous_uptime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior
    previous_downtime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior
    governed_flows: xr.DataArray | None = None  # (dim, governed_idx) — only for component status

    def __post_init__(self) -> None:
        """Validate durations >= 0 and max >= min where both given."""
        for name in ('uptime_min', 'uptime_max', 'downtime_min', 'downtime_max'):
            arr: xr.DataArray = getattr(self, name)
            mask = (~np.isnan(arr)) & (arr < 0)
            if mask.any():
                dim = arr.dims[0]
                raise ValueError(f'Status.{name} < 0 on {list(arr.coords[dim][mask].values)}')

        for lo, hi, what in (
            (self.uptime_min, self.uptime_max, 'uptime'),
            (self.downtime_min, self.downtime_max, 'downtime'),
        ):
            both = ~np.isnan(lo) & ~np.isnan(hi)
            bad = both & (hi < lo)
            if bad.any():
                dim = lo.dims[0]
                raise ValueError(f'Status.{what}_max < {what}_min on {list(lo.coords[dim][bad].values)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset."""
        return _container_from_dataset(cls, ds)

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Status]],
        effect_ids: list[str],
        mapper: _TimeMapper,
        dim: str,
        prior_rates_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
        governed_flows_map: dict[str, list[str]] | None = None,
    ) -> Self | None:
        """Validate Status objects and collect into DataArrays, or None if empty.

        Args:
            items: Pairs of (id, Status).
            effect_ids: Known effect ids for validation.
            mapper: Converter of operational inputs onto the flat time axis.
            dim: Dimension name for the resulting arrays.
            prior_rates_map: Item id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
            governed_flows_map: Item id to ids of flows the status governs.
                Only populated for component-level status; emits a 2D
                ``(dim, governed_idx)`` string array.
        """
        from fluxopt.constraints.status import compute_previous_duration

        if not items:
            return None

        prior_rates_map = prior_rates_map or {}
        effect_set = set(effect_ids)
        tmpl = _effect_template({'effect': effect_ids, 'time': mapper.time})

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
            min_ups.append(s.uptime_min if s.uptime_min is not None else np.nan)
            max_ups.append(s.uptime_max if s.uptime_max is not None else np.nan)
            min_downs.append(s.downtime_min if s.downtime_min is not None else np.nan)
            max_downs.append(s.downtime_max if s.downtime_max is not None else np.nan)

            prior = prior_rates_map.get(item_id)
            if prior is not None:
                initials.append(1.0 if prior[-1] > 0 else 0.0)
                prior_da = xr.DataArray(prior, dims=['_prior_t'])
                prev_ups.append(compute_previous_duration(prior_da, target_state=1, dt=dt))
                prev_downs.append(compute_previous_duration(prior_da, target_state=0, dt=dt))
            else:
                initials.append(np.nan)
                prev_ups.append(np.nan)
                prev_downs.append(np.nan)

            er = tmpl.zeros()
            for ek, ev in s.effects_per_running_hour.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_running_hour on {item_id!r}')
                er.loc[ek] = mapper.to_flat(ev, name=ek)
            er_slices.append(er)

            es = tmpl.zeros()
            for ek, ev in s.effects_per_startup.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_startup on {item_id!r}')
                es.loc[ek] = mapper.to_flat(ev, name=ek)
            es_slices.append(es)

        coords = {dim: ids}
        status_idx = pd.Index(ids, name=dim)

        prev_up_arr = np.array(prev_ups)
        prev_down_arr = np.array(prev_downs)

        governed: xr.DataArray | None = None
        if governed_flows_map:
            max_n = max(len(governed_flows_map.get(i, [])) for i in ids)
            if max_n > 0:
                rows = [
                    governed_flows_map.get(i, []) + [''] * (max_n - len(governed_flows_map.get(i, []))) for i in ids
                ]
                governed = xr.DataArray(
                    np.array(rows, dtype=object),
                    dims=[dim, 'governed_idx'],
                    coords={dim: ids},
                )

        return cls(
            uptime_min=xr.DataArray(np.array(min_ups), dims=[dim], coords=coords),
            uptime_max=xr.DataArray(np.array(max_ups), dims=[dim], coords=coords),
            downtime_min=xr.DataArray(np.array(min_downs), dims=[dim], coords=coords),
            downtime_max=xr.DataArray(np.array(max_downs), dims=[dim], coords=coords),
            initial=xr.DataArray(np.array(initials), dims=[dim], coords=coords),
            effects_running=fast_concat(er_slices, status_idx),
            effects_startup=fast_concat(es_slices, status_idx),
            previous_uptime=xr.DataArray(prev_up_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_up_arr))
            else None,
            previous_downtime=xr.DataArray(prev_down_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_down_arr))
            else None,
            governed_flows=governed,
        )


@dataclass
class FlowsData:
    bound_type: xr.DataArray  # (flow,) — BoundType.UNSIZED | BoundType.BOUNDED | BoundType.PROFILE
    rel_lb: xr.DataArray  # (flow, time)
    rel_ub: xr.DataArray  # (flow, time)
    fixed_profile: xr.DataArray  # (flow, time) — NaN where not fixed
    size: xr.DataArray  # (flow,) — NaN for unsized
    effect_coeff: xr.DataArray  # (flow, effect, time)
    flow_hours_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    flow_hours_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    ramp_up: xr.DataArray | None = None  # (flow, time) — NaN = no limit [1/h]
    ramp_down: xr.DataArray | None = None  # (flow, time) — NaN = no limit [1/h]
    sizing: SizingData | None = None  # dim Dim.SIZING_FLOW
    status: StatusData | None = None  # dim Dim.STATUS_FLOW
    invest: InvestmentData | None = None  # dim Dim.INVEST_FLOW
    cstatus: StatusData | None = None  # dim Dim.CSTATUS_COMPONENT, entity coord 'component'

    def __post_init__(self) -> None:
        """Validate relative bounds, status non-degeneracy, and sized-feature requirements."""
        reduce_dims = [d for d in self.rel_lb.dims if d != 'flow']
        bad_neg = (self.rel_lb < -1e-12).any(reduce_dims)
        if bad_neg.any():
            raise ValueError(f'Negative lower bounds on flows: {list(self.rel_lb.coords["flow"][bad_neg].values)}')
        bad_order = (self.rel_lb > self.rel_ub + 1e-12).any(reduce_dims)
        if bad_order.any():
            raise ValueError(
                f'Lower bound > upper bound on flows: {list(self.rel_lb.coords["flow"][bad_order].values)}'
            )
        self._check_status_not_degenerate()
        self._check_sized_features()

    def _check_status_not_degenerate(self) -> None:
        """Status flows need rel_lb > 0 everywhere, else on/off is indistinguishable.

        Enforced here (not only on the ``Flow`` element) because ModelData can
        be edited directly or reloaded from file — a zero lower bound would
        make the model solve with silently meaningless status results.
        """
        if self.status is None:
            return
        status_ids = list(self.status.uptime_min.coords[self.status.uptime_min.dims[0]].values)
        lb = self.rel_lb.sel(flow=status_ids)
        degenerate = (lb <= 0).any([d for d in lb.dims if d != 'flow'])
        if degenerate.any():
            raise ValueError(
                f'Status flows must have rel_lb > 0 (else on/off is indistinguishable); '
                f'violated on {list(lb.coords["flow"][degenerate].values)}'
            )

    def _check_sized_features(self) -> None:
        """Ramp limits and load-factor bounds need a sized flow (fixed, Sizing, or Investment).

        Without a size these features would feed NaN into constraint
        coefficients; the element layer already rejects this at authoring
        time, this is the guard for direct data edits and reloads.
        """
        flow_ids = self.size.coords['flow'].values
        extra: set[str] = set()
        if self.sizing is not None:
            extra |= set(map(str, self.sizing.min.coords[self.sizing.min.dims[0]].values))
        if self.invest is not None:
            extra |= set(map(str, self.invest.min.coords[self.invest.min.dims[0]].values))
        sized = self.size.notnull().values | np.isin(flow_ids, list(extra))

        for da, label in (
            (self.ramp_up, 'ramp_up'),
            (self.ramp_down, 'ramp_down'),
            (self.load_factor_min, 'load_factor_min'),
            (self.load_factor_max, 'load_factor_max'),
        ):
            if da is None:
                continue
            has = da.notnull()
            if extra_dims := [d for d in has.dims if d != 'flow']:
                has = has.any(extra_dims)
            if (bad := has.values & ~sized).any():
                raise ValueError(
                    f'{label} requires a sized flow (fixed, Sizing, or Investment) on {list(flow_ids[bad])}'
                )

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, containers: dict[str, Any] | None = None) -> Self:
        """Deserialize from xr.Dataset plus reconstructed nested containers.

        Args:
            ds: Dataset with the table's plain-DataArray variables.
            containers: Nested container objects (``sizing``/``status``/
                ``invest``/``cstatus``) parsed from netCDF sub-groups.
        """
        containers = containers or {}
        kwargs: dict[str, Any] = {
            f.name: containers.get(f.name) if f.name in _CONTAINER_FIELD_NAMES else ds.get(f.name) for f in fields(cls)
        }
        return cls(**kwargs)

    @classmethod
    def build(
        cls,
        flows: list[_BoundFlow],
        mapper: _TimeMapper,
        effects: list[Effect],
        dt: float = 1.0,
        period: pd.Index | None = None,
        component_status_items: list[tuple[str, Status, list[str]]] | None = None,
    ) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            mapper: Converter of operational inputs onto the flat time axis.
                Operational profiles (``relative_rate_min/max``,
                ``fixed_relative_profile``, ``effects_per_flow_hour``, ramps)
                may vary per period via per-period mappings, ``(period,)``
                arrays, or ``(time, period)`` frames.
            effects: Effect definitions for cost coefficients.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for multi-period models (period-scoped
                sizing/investment effects).
            component_status_items: Component-level status entries as
                ``(component_id, Status, [governed flow ids])``. Each entry
                produces an on/startup/shutdown binary keyed by the
                component, gating all listed flows together.
        """
        from fluxopt.elements import Investment, Sizing

        time = mapper.time
        flow_ids = [bf.id for bf in flows]
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)
        n_time = len(time)
        n_effects = len(effect_ids)

        bound_type: list[str] = []
        rel_lbs: list[xr.DataArray] = []
        rel_ubs: list[xr.DataArray] = []
        profiles: list[xr.DataArray] = []
        size_vals = np.full(len(flows), np.nan)
        fh_min_vals = np.full(len(flows), np.nan)
        fh_max_vals = np.full(len(flows), np.nan)
        lf_min_vals = np.full(len(flows), np.nan)
        lf_max_vals = np.full(len(flows), np.nan)
        ramp_ups: list[xr.DataArray] = []
        ramp_downs: list[xr.DataArray] = []
        has_ramp_up = False
        has_ramp_down = False
        effect_coeffs: list[xr.DataArray] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        status_items: list[tuple[str, Status]] = []
        prior_rates_map: dict[str, list[float]] = {}

        nan_envelope = xr.DataArray(
            np.full(n_time, np.nan),
            dims=['time'],
            coords={'time': time},
        )

        for i, (fid, f, _sign) in enumerate(flows):
            rel_lbs.append(mapper.to_flat(f.relative_rate_min, name='relative_rate_min'))
            rel_ubs.append(mapper.to_flat(f.relative_rate_max, name='relative_rate_max'))

            if isinstance(f.size, Sizing):
                sizing_items.append((fid, f.size))
            elif isinstance(f.size, Investment):
                invest_items.append((fid, f.size))
            elif f.size is not None:
                size_vals[i] = f.size

            if f.flow_hours_min is not None:
                fh_min_vals[i] = f.flow_hours_min
            if f.flow_hours_max is not None:
                fh_max_vals[i] = f.flow_hours_max
            if f.load_factor_min is not None:
                lf_min_vals[i] = f.load_factor_min
            if f.load_factor_max is not None:
                lf_max_vals[i] = f.load_factor_max

            has_ramp_up = has_ramp_up or f.ramp_up_per_hour is not None
            has_ramp_down = has_ramp_down or f.ramp_down_per_hour is not None
            ramp_ups.append(
                mapper.to_flat(f.ramp_up_per_hour, name='ramp_up') if f.ramp_up_per_hour is not None else nan_envelope
            )
            ramp_downs.append(
                mapper.to_flat(f.ramp_down_per_hour, name='ramp_down')
                if f.ramp_down_per_hour is not None
                else nan_envelope
            )

            if f.fixed_relative_profile is not None:
                profiles.append(mapper.to_flat(f.fixed_relative_profile, name='fixed_relative_profile'))
                bound_type.append(BoundType.PROFILE)
            elif f.size is None:
                profiles.append(nan_envelope)
                bound_type.append(BoundType.UNSIZED)
            else:
                profiles.append(nan_envelope)
                bound_type.append(BoundType.BOUNDED)

            # Effect coefficients for this flow
            ec = xr.DataArray(
                np.zeros((n_effects, n_time)),
                dims=['effect', 'time'],
                coords={'effect': effect_ids, 'time': time},
            )
            for effect_label, factor in f.effects_per_flow_hour.items():
                if effect_label not in effect_set:
                    raise ValueError(f'Unknown effect {effect_label!r} in Flow.effects_per_flow_hour on {fid!r}')
                ec.loc[effect_label] = mapper.to_flat(factor, name=effect_label)
            effect_coeffs.append(ec)

            if f.status is not None:
                status_items.append((fid, f.status))

            if f.prior_rates is not None:
                prior_rates_map[fid] = f.prior_rates

        flow_idx = pd.Index(flow_ids, name='flow')
        return cls(
            bound_type=xr.DataArray(bound_type, dims=['flow'], coords={'flow': flow_ids}),
            rel_lb=fast_concat(rel_lbs, flow_idx),
            rel_ub=fast_concat(rel_ubs, flow_idx),
            fixed_profile=fast_concat(profiles, flow_idx),
            size=xr.DataArray(size_vals, dims=['flow'], coords={'flow': flow_ids}),
            effect_coeff=fast_concat(effect_coeffs, flow_idx),
            flow_hours_min=_flow_bound_or_none(fh_min_vals, flow_ids),
            flow_hours_max=_flow_bound_or_none(fh_max_vals, flow_ids),
            load_factor_min=_flow_bound_or_none(lf_min_vals, flow_ids),
            load_factor_max=_flow_bound_or_none(lf_max_vals, flow_ids),
            ramp_up=fast_concat(ramp_ups, flow_idx) if has_ramp_up else None,
            ramp_down=fast_concat(ramp_downs, flow_idx) if has_ramp_down else None,
            sizing=SizingData.build(sizing_items, effect_ids, dim=Dim.SIZING_FLOW, period=period),
            invest=InvestmentData.build(invest_items, effect_ids, dim=Dim.INVEST_FLOW, period=period),
            status=StatusData.build(
                status_items,
                effect_ids,
                mapper,
                dim=Dim.STATUS_FLOW,
                prior_rates_map=prior_rates_map,
                dt=dt,
            ),
            cstatus=StatusData.build(
                [(cid, s) for cid, s, _ in (component_status_items or [])],
                effect_ids,
                mapper,
                dim=Dim.CSTATUS_COMPONENT,
                governed_flows_map={cid: gov for cid, _, gov in (component_status_items or [])} or None,
            ),
        )


def _flow_bound_or_none(vals: np.ndarray, flow_ids: list[str]) -> xr.DataArray | None:
    """Wrap per-flow bound values as a (flow,) DataArray, or None if all NaN.

    Args:
        vals: Bound value per flow; NaN = unbounded.
        flow_ids: Flow coordinate labels.
    """
    if np.all(np.isnan(vals)):
        return None
    return xr.DataArray(vals, dims=['flow'], coords={'flow': flow_ids})


def _carrier_dim_id(flow: Flow) -> str:
    """Return the carrier dimension coordinate value for a flow.

    Single-node carriers use the carrier id directly.
    Multi-node carriers use ``carrier_id:node``.

    Args:
        flow: Flow with carrier (and optional node).
    """
    from fluxopt.elements import node_id

    if flow.node is not None:
        return node_id(flow.carrier, flow.node)
    return flow.carrier


@dataclass
class CarriersData:
    flow_coeff: xr.DataArray  # (carrier, flow) — +1/-1/NaN
    unit: xr.DataArray  # (carrier,) — energy unit label
    color: xr.DataArray  # (carrier,) — plot color ('' if unset)
    description: xr.DataArray  # (carrier,) — human-readable description

    def __post_init__(self) -> None:
        """Validate balance coefficients are +1 / -1 / NaN (also on netCDF reload)."""
        coeff = self.flow_coeff.values
        ok = np.isnan(coeff) | (coeff == 1.0) | (coeff == -1.0)
        if not ok.all():
            bad = sorted({float(v) for v in coeff[~ok].ravel()})
            raise ValueError(f'CarriersData.flow_coeff must be +1, -1, or NaN; got {bad}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with ``flow_coeff``, ``unit``, ``color``, ``description``.
        """
        return cls(
            flow_coeff=ds['flow_coeff'],
            unit=ds['unit'],
            color=ds['color'],
            description=ds['description'],
        )

    @classmethod
    def build(cls, carriers: list[Carrier], flows: list[_BoundFlow], carrier_coeff: dict[str, float]) -> Self:
        """Build CarriersData from explicit carrier declarations.

        Args:
            carriers: Declared carriers.
            flows: All collected flows.
            carrier_coeff: Mapping of qualified flow id to +1 (produces) or -1 (consumes).
        """
        from fluxopt.elements import node_id

        flow_ids = [bf.id for bf in flows]
        # Build carrier dim ids from explicit declarations
        carrier_ids: list[str] = []
        for c in carriers:
            if c.nodes:
                carrier_ids.extend(node_id(c.id, node) for node in c.nodes)
            else:
                carrier_ids.append(c.id)

        coeff = np.full((len(carrier_ids), len(flow_ids)), np.nan)
        for fi, (fid, f, _sign) in enumerate(flows):
            ci = carrier_ids.index(_carrier_dim_id(f))
            coeff[ci, fi] = carrier_coeff[fid]

        # Expand carrier metadata to match carrier dim (one entry per node)
        units: list[str] = []
        colors: list[str] = []
        descriptions: list[str] = []
        for c in carriers:
            n = max(len(c.nodes), 1)
            units.extend([c.unit] * n)
            colors.extend([c.color or ''] * n)
            descriptions.extend([c.description] * n)

        return cls(
            flow_coeff=xr.DataArray(coeff, dims=['carrier', 'flow'], coords={'carrier': carrier_ids, 'flow': flow_ids}),
            unit=xr.DataArray(units, dims=['carrier'], coords={'carrier': carrier_ids}),
            color=xr.DataArray(colors, dims=['carrier'], coords={'carrier': carrier_ids}),
            description=xr.DataArray(descriptions, dims=['carrier'], coords={'carrier': carrier_ids}),
        )


@dataclass
class ConvertersData:
    pair_coeff: xr.DataArray  # (pair, eq_idx, time) — non-zero coefficients only
    pair_converter: xr.DataArray  # (pair,) — converter id per pair
    pair_flow: xr.DataArray  # (pair,) — flow id per pair
    eq_mask: xr.DataArray  # (converter, eq_idx)

    def __post_init__(self) -> None:
        """Validate pair/mask consistency (also on netCDF reload)."""
        if self.eq_mask.dtype != bool:
            raise ValueError(f'ConvertersData.eq_mask must be boolean, got dtype {self.eq_mask.dtype}')
        known = set(self.eq_mask.coords['converter'].values)
        if unknown := sorted(set(self.pair_converter.values) - known):
            raise ValueError(f'ConvertersData.pair_converter references unknown converter(s) {unknown}')

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
    def build(cls, converters: list[Converter], mapper: _TimeMapper) -> Self | None:
        """Build ConvertersData with sparse pair-based conversion coefficients.

        Only linear converters are included; piecewise converters
        (``conversion is not None``) live in :class:`PiecewiseData`.

        Args:
            converters: Converter definitions.
            mapper: Converter of operational inputs onto the flat time axis.
        """
        converters = [c for c in converters if c.conversion is None]
        if not converters:
            return None

        time = mapper.time
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

            for fid, flow, _sign in conv._qualified_flows():
                short = flow.short_id
                eq_coeffs = np.zeros((max_eq, n_time))
                for eq_i, equation in enumerate(conv.conversion_factors):
                    if short in equation:
                        eq_coeffs[eq_i] = mapper.to_flat(equation[short], name=short).values
                pairs_conv.append(conv.id)
                pairs_flow.append(fid)
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


@dataclass
class PiecewiseData:
    """Piecewise-linear conversion data for converters with ``PiecewiseConversion``.

    Stored sparsely as one row per (converter, flow) pair; the ``method``
    and ``availability`` arrays index by ``pw_converter``.
    """

    breakpoints: xr.DataArray  # (pw_pair, breakpoint, time)
    pair_converter: xr.DataArray  # (pw_pair,) — converter id
    pair_flow: xr.DataArray  # (pw_pair,) — qualified flow id
    pair_bound: xr.DataArray  # (pw_pair,) — '==' / '<=' / '>='
    method: xr.DataArray  # (pw_converter,) — 'auto' / 'sos2' / 'incremental' / 'lp'
    availability: xr.DataArray  # (pw_converter, time)
    has_status: xr.DataArray  # (pw_converter,) — bool

    def __post_init__(self) -> None:
        """Validate method and bound values (also on netCDF reload)."""
        valid_methods = set(get_args(PiecewiseMethod.__value__))
        if bad := sorted(set(map(str, self.method.values)) - valid_methods):
            raise ValueError(f'PiecewiseData.method must be one of {sorted(valid_methods)}; got {bad}')
        if bad := sorted(set(map(str, self.pair_bound.values)) - {'==', '<=', '>='}):
            raise ValueError(f"PiecewiseData.pair_bound must be '==', '<=', or '>='; got {bad}")

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with piecewise variables.
        """
        return cls(
            breakpoints=ds['breakpoints'],
            pair_converter=ds['pair_converter'],
            pair_flow=ds['pair_flow'],
            pair_bound=ds['pair_bound'],
            method=ds['method'],
            availability=ds['availability'],
            has_status=ds['has_status'],
        )

    def converter_ids(self) -> list[str]:
        """Return list of piecewise converter ids in original order."""
        return list(self.method.coords[Dim.PW_CONVERTER].values)

    @classmethod
    def build(cls, converters: list[Converter], mapper: _TimeMapper) -> Self | None:
        """Build PiecewiseData from converters with ``PiecewiseConversion``.

        Args:
            converters: Converter definitions; only those with
                ``conversion is not None`` are processed.
            mapper: Converter of operational inputs onto the flat time axis.
        """
        converters = [c for c in converters if c.conversion is not None]
        if not converters:
            return None

        conv_ids: list[str] = []
        methods: list[str] = []
        avail_slices: list[xr.DataArray] = []
        has_statuses: list[bool] = []

        pair_conv_ids: list[str] = []
        pair_flow_ids: list[str] = []
        pair_bounds: list[str] = []
        bp_slices: list[xr.DataArray] = []

        for conv in converters:
            assert conv.conversion is not None
            curve = conv.conversion
            conv_ids.append(conv.id)
            methods.append(curve.method)
            avail_slices.append(mapper.to_flat(curve.availability, name='availability'))
            has_statuses.append(curve.status is not None)

            short_to_qid = {bf.flow.short_id: bf.id for bf in conv._qualified_flows()}
            for short, pts, bound in curve._iter_normalized():
                qid = short_to_qid[short]
                bp_arrays = [mapper.to_flat(bp, name='breakpoint') for bp in pts]
                bp_idx = pd.Index(range(len(bp_arrays)), name='breakpoint')
                bp_da = fast_concat(bp_arrays, bp_idx)
                pair_conv_ids.append(conv.id)
                pair_flow_ids.append(qid)
                pair_bounds.append(bound)
                bp_slices.append(bp_da)

        pair_idx = pd.Index(range(len(bp_slices)), name=Dim.PW_PAIR)
        breakpoints_da = fast_concat(bp_slices, pair_idx)

        conv_idx = pd.Index(conv_ids, name=Dim.PW_CONVERTER)
        availability = fast_concat(avail_slices, conv_idx)

        data = cls(
            breakpoints=breakpoints_da,
            pair_converter=xr.DataArray(pair_conv_ids, dims=[Dim.PW_PAIR]),
            pair_flow=xr.DataArray(pair_flow_ids, dims=[Dim.PW_PAIR]),
            pair_bound=xr.DataArray(pair_bounds, dims=[Dim.PW_PAIR]),
            method=xr.DataArray(methods, dims=[Dim.PW_CONVERTER], coords={Dim.PW_CONVERTER: conv_ids}),
            availability=availability,
            has_status=xr.DataArray(has_statuses, dims=[Dim.PW_CONVERTER], coords={Dim.PW_CONVERTER: conv_ids}),
        )
        data._warn_redundant_status()
        return data

    def _warn_redundant_status(self) -> None:
        """Warn for converters where Status is set but the curve includes a
        (0, ..., 0) breakpoint at any (breakpoint, timestep) position.

        When that's the case, the optimizer can sit at zero with ``active=1``,
        so the on/off binary is decoupled from the actual operating state and
        Status features will not behave as expected.
        """
        atol = 1e-9
        is_zero = abs(self.breakpoints) <= atol  # (pw_pair, breakpoint, time)
        for conv_id in self.converter_ids():
            if not bool(self.has_status.sel(pw_converter=conv_id).item()):
                continue
            mask = self.pair_converter.values == conv_id
            all_flows_zero = is_zero.isel(pw_pair=mask).all(Dim.PW_PAIR)  # (breakpoint, time)
            if bool(all_flows_zero.any().item()):
                warnings.warn(
                    f'PiecewiseConversion on converter {conv_id!r} has Status, '
                    'but the curve includes a (0, ..., 0) breakpoint. The '
                    'optimizer can sit at zero with status=on, decoupling the '
                    'binary from the actual operating state — Status features '
                    'will not behave as expected. If you want Status to work '
                    'as expected, drop the zero breakpoint so the only way to '
                    'produce zero is status=off.',
                    UserWarning,
                    stacklevel=4,
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
    total_min: xr.DataArray  # (effect,) — weighted total bound
    total_max: xr.DataArray  # (effect,) — weighted total bound
    periodic_min: xr.DataArray  # (effect[, period]) — per-period bound
    periodic_max: xr.DataArray  # (effect[, period]) — per-period bound
    cf_temporal: xr.DataArray | None = None  # (effect, source_effect, time)
    period_weights: xr.DataArray | None = None  # (effect, period)

    def __post_init__(self) -> None:
        """Reject self-references and cycles in the cross-effect matrix (also on netCDF reload)."""
        if self.cf_temporal is None:
            return
        contributes = self.cf_temporal != 0
        extra_dims = [d for d in contributes.dims if d not in ('effect', 'source_effect')]
        contributes = contributes.any(extra_dims)  # (effect, source_effect)
        for eid in contributes.coords['effect'].values:
            if bool(contributes.sel(effect=eid, source_effect=eid)):
                raise ValueError(f'Effect {eid!r} cannot reference itself in contribution_from')
        adjacency = {
            str(eid): [
                str(s)
                for s in contributes.coords['source_effect'].values
                if bool(contributes.sel(effect=eid, source_effect=s))
            ]
            for eid in contributes.coords['effect'].values
        }
        cycle = _detect_contribution_cycle(adjacency)
        if cycle is not None:
            raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with effect variables.
        """
        return cls(**{f.name: ds[f.name] for f in fields(cls) if f.name in ds.data_vars})

    @classmethod
    def build(
        cls,
        effects: list[Effect],
        mapper: _TimeMapper,
        period: pd.Index | None = None,
    ) -> Self:
        """Build EffectsData from element objects.

        Args:
            effects: Effect definitions.
            mapper: Converter of operational inputs onto the flat time axis.
            period: Period index (multi-period only).
        """
        time = mapper.time
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)
        n = len(effects)
        total_min = np.full(n, np.nan)
        total_max = np.full(n, np.nan)
        periodic_mins: list[xr.DataArray] = []
        periodic_maxs: list[xr.DataArray] = []

        # Periodic bounds are scalar in single-period models, (period,) in multi-period
        period_coords: dict[str, Any] = {'period': period} if period is not None else {}
        nan_periodic = (
            xr.DataArray(np.full(len(period), np.nan), dims=['period'], coords={'period': period})
            if period is not None
            else xr.DataArray(np.nan)
        )

        has_contributions = False
        for i, e in enumerate(effects):
            if e.total_min is not None:
                total_min[i] = e.total_min
            if e.total_max is not None:
                total_max[i] = e.total_max
            periodic_mins.append(
                as_dataarray(e.periodic_min, period_coords) if e.periodic_min is not None else nan_periodic
            )
            periodic_maxs.append(
                as_dataarray(e.periodic_max, period_coords) if e.periodic_max is not None else nan_periodic
            )
            if e.contribution_from:
                has_contributions = True

        # Build cross-effect contribution arrays; self-references and cycles
        # are rejected by __post_init__ on the dense matrix.
        cf_temporal: xr.DataArray | None = None
        if has_contributions:
            tmpl_t = _effect_template({'effect': effect_ids, 'source_effect': effect_ids, 'time': time})
            temporal_mat = tmpl_t.zeros()
            for e in effects:
                for src_id, factor in e.contribution_from.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from on {e.id!r}')
                    temporal_mat.loc[e.id, src_id] = mapper.to_flat(factor, name=src_id)
            cf_temporal = temporal_mat

        effect_idx = pd.Index(effect_ids, name='effect')

        # Per-effect period weights
        pw: xr.DataArray | None = None
        if period is not None:
            has_pw = any(e.period_weights is not None for e in effects)
            n_periods = len(period)
            if has_pw:
                mat = np.full((n, n_periods), np.nan)
                for i, e in enumerate(effects):
                    if e.period_weights is not None:
                        if len(e.period_weights) != n_periods:
                            msg = f'Effect {e.id!r}: period_weights has {len(e.period_weights)} entries, expected {n_periods}'
                            raise ValueError(msg)
                        vals = np.asarray(e.period_weights, dtype=float)
                        if not np.all(np.isfinite(vals)) or not np.all(vals > 0):
                            msg = f'Effect {e.id!r}: period_weights must be positive and finite, got {vals}'
                            raise ValueError(msg)
                        mat[i] = vals
                pw = xr.DataArray(mat, dims=['effect', 'period'], coords={'effect': effect_ids, 'period': period})

        return cls(
            total_min=xr.DataArray(total_min, dims=['effect'], coords={'effect': effect_ids}),
            total_max=xr.DataArray(total_max, dims=['effect'], coords={'effect': effect_ids}),
            periodic_min=fast_concat(periodic_mins, effect_idx),
            periodic_max=fast_concat(periodic_maxs, effect_idx),
            cf_temporal=cf_temporal,
            period_weights=pw,
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
    final_level_min: xr.DataArray | None = None  # (storage,) — NaN = unbounded [MWh]
    final_level_max: xr.DataArray | None = None  # (storage,) — NaN = unbounded [MWh]
    prevent_simultaneous: xr.DataArray | None = None  # (storage,) — bool
    sizing: SizingData | None = None  # dim Dim.SIZING_STORAGE
    invest: InvestmentData | None = None  # dim Dim.INVEST_STORAGE

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
        bad_loss = ((self.loss < 0) | (self.loss > 1)).any('time')
        if bad_loss.any():
            raise ValueError(f'relative_loss_per_hour must be in [0, 1] on storages: {list(s[bad_loss].values)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, containers: dict[str, Any] | None = None) -> Self:
        """Deserialize from xr.Dataset plus reconstructed nested containers.

        Args:
            ds: Dataset with the table's plain-DataArray variables.
            containers: Nested container objects (``sizing``/``status``/
                ``invest``/``cstatus``) parsed from netCDF sub-groups.
        """
        containers = containers or {}
        kwargs: dict[str, Any] = {
            f.name: containers.get(f.name) if f.name in _CONTAINER_FIELD_NAMES else ds.get(f.name) for f in fields(cls)
        }
        return cls(**kwargs)

    @classmethod
    def build(
        cls,
        storages: list[Storage],
        mapper: _TimeMapper,
        effects: list[Effect] | None = None,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Build StoragesData from element objects.

        Args:
            storages: Storage definitions.
            mapper: Converter of operational inputs onto the flat time axis.
            effects: Effect definitions for sizing cost validation.
            period: Period index for period-varying effects.
        """
        from fluxopt.elements import Investment, Sizing

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
        final_min_vals = np.full(n, np.nan)
        final_max_vals = np.full(n, np.nan)
        prevent_vals = np.zeros(n, dtype=bool)
        charge_flow: list[str] = []
        discharge_flow: list[str] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []

        for i, s in enumerate(storages):
            if isinstance(s.capacity, Sizing):
                sizing_items.append((s.id, s.capacity))
            elif isinstance(s.capacity, Investment):
                invest_items.append((s.id, s.capacity))
            elif s.capacity is not None:
                capacity_vals[i] = s.capacity

            eta_cs.append(mapper.to_flat(s.eta_charge, name='eta_charge'))
            eta_ds.append(mapper.to_flat(s.eta_discharge, name='eta_discharge'))
            losses.append(mapper.to_flat(s.relative_loss_per_hour, name='relative_loss_per_hour'))

            level_lbs.append(mapper.to_flat(s.relative_level_min, name='relative_level_min'))
            level_ubs.append(mapper.to_flat(s.relative_level_max, name='relative_level_max'))

            cyclic_vals[i] = s.cyclic
            if s.prior_level is not None:
                prior_level_vals[i] = s.prior_level
            if s.final_level_min is not None:
                final_min_vals[i] = s.final_level_min
            if s.final_level_max is not None:
                final_max_vals[i] = s.final_level_max
            prevent_vals[i] = s.prevent_simultaneous

            charge_flow.append(s._charging_id)
            discharge_flow.append(s._discharging_id)

        stor_idx = pd.Index(stor_ids, name='storage')
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
            final_level_min=(
                xr.DataArray(final_min_vals, dims=['storage'], coords={'storage': stor_ids})
                if not np.all(np.isnan(final_min_vals))
                else None
            ),
            final_level_max=(
                xr.DataArray(final_max_vals, dims=['storage'], coords={'storage': stor_ids})
                if not np.all(np.isnan(final_max_vals))
                else None
            ),
            prevent_simultaneous=(
                xr.DataArray(prevent_vals, dims=['storage'], coords={'storage': stor_ids})
                if prevent_vals.any()
                else None
            ),
            sizing=SizingData.build(sizing_items, effect_ids, dim=Dim.SIZING_STORAGE, period=period),
            invest=InvestmentData.build(invest_items, effect_ids, dim=Dim.INVEST_STORAGE, period=period),
        )


def _compute_period_weights(
    periods: list[int] | pd.Index,
    period_weights: list[float] | None = None,
) -> tuple[pd.Index, xr.DataArray]:
    """Compute period weights from a period index.

    Args:
        periods: Integer period labels (e.g. [2020, 2025, 2030]).
        period_weights: Explicit weights per period. If None, inferred from
            ``np.diff(periods)`` with the last gap repeated.

    Returns:
        Tuple of (period_index, period_weights DataArray).
    """
    idx = pd.Index(periods, name='period')
    if not np.issubdtype(idx.dtype, np.integer):  # pyrefly: ignore[bad-argument-type]
        raise TypeError(f'periods must be integer, got {idx.dtype}')
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError('periods must be monotonically increasing and unique')

    if period_weights is not None:
        if len(period_weights) != len(idx):
            msg = f'period_weights has {len(period_weights)} entries, expected {len(idx)}'
            raise ValueError(msg)
        w = np.asarray(period_weights, dtype=float)
    elif len(idx) < 2:
        raise ValueError('period_weights is required when only one period is given')
    else:
        gaps = np.diff(idx.to_numpy().astype(int))
        w = np.append(gaps, gaps[-1])

    if not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise ValueError(f'period_weights must be positive and finite, got {w}')

    return idx, xr.DataArray(w, dims=['period'], coords={'period': idx}, name='period_weight')


def _replicate_time_index(base: TimeIndex, period_idx: pd.Index) -> list[TimeIndex]:
    """Replicate a within-period time index once per period with shifted labels.

    Datetime labels shift into each period's calendar by a whole-day offset
    anchored at Jan 1 of the base year (the day count between the Jan 1s of
    the base and target years). A constant offset preserves time-of-day,
    strict ordering, and dt exactly — leap-day grids replicate safely; the
    trade is that dates after Feb 28 drift one calendar day when base and
    target year differ in leap parity. Integer labels shift by the index
    span, giving a global running index.

    Args:
        base: Normalized within-period time index.
        period_idx: Integer period labels (ascending).
    """
    if isinstance(base, pd.DatetimeIndex):
        p0 = int(period_idx[0])
        base_year = int(base[0].year)
        anchor = pd.Timestamp(year=base_year, month=1, day=1)
        segments: list[TimeIndex] = []
        for p in period_idx:
            offset = pd.Timestamp(year=base_year + int(p) - p0, month=1, day=1) - anchor
            segments.append(base + offset)
        return segments
    step = int(base[1] - base[0]) if len(base) > 1 else 1
    span = int(base[-1] - base[0]) + step
    return [base + k * span for k in range(len(period_idx))]


def _concat_time_indexes(segments: list[TimeIndex]) -> TimeIndex:
    """Concatenate per-period time indexes into the flat time index.

    Args:
        segments: One index per period, in period order.

    Raises:
        ValueError: If the flat index is not strictly increasing.
    """
    flat: TimeIndex = segments[0]
    for seg in segments[1:]:
        flat = flat.append(seg)
    if len(flat) > 1 and (not flat.is_monotonic_increasing or not flat.is_unique):
        raise ValueError(
            'Flat time index must be strictly increasing across periods. '
            'Per-period timesteps overlap — pass timesteps as a {period: index} '
            'mapping with non-overlapping calendar ranges.'
        )
    return flat


@dataclass
class Dims:
    """Shared model coordinates and temporal metadata.

    Owns the flat time dimension (spanning all periods), timestep durations,
    weights, and the period dimension. In multi-period models ``time_period``
    maps each timestep to its investment period; operational arrays and
    variables carry only the flat ``time`` dim, while investment-scoped data
    lives on ``period``. Treated as immutable after construction (derived
    views are cached). See: docs/design/time-index.md
    """

    time: xr.DataArray  # (time,) — flat coordinate labels across all periods
    dt: xr.DataArray  # (time,) — timestep durations [h]
    weights: xr.DataArray  # (time,) — timestep weights (occurrence counts)
    period: xr.DataArray | None = None  # (period,) — coordinate labels
    period_weights: xr.DataArray | None = None  # (period,) — duration weights
    time_period: xr.DataArray | None = None  # (time,) — period label per timestep

    def __post_init__(self) -> None:
        arrays = [('dt', self.dt), ('weights', self.weights)]
        if self.time_period is not None:
            arrays.append(('time_period', self.time_period))
        for name, arr in arrays:
            if arr.dims != ('time',):
                raise ValueError(f"Dims.{name} must be 1D with dims=('time',), got {arr.dims}")
            if not arr.coords['time'].equals(self.time):
                raise ValueError(f'Dims.{name} time coordinate does not match Dims.time')
        if (self.period is not None) != (self.time_period is not None):
            raise ValueError('Dims.period and Dims.time_period must be set together')
        if self.period is not None:
            assert self.time_period is not None
            labels = self.period.values
            tp = self.time_period.values
            if np.any(np.diff(tp) < 0):
                raise ValueError('Dims.time_period must be non-decreasing along time')
            present = pd.unique(tp)
            if list(present) != list(labels):
                raise ValueError(f'Dims.time_period labels {list(present)} do not match Dims.period {list(labels)}')

    def coords(self, *, time: bool = False, period: bool = False) -> dict[str, xr.DataArray]:
        """Return shared coordinates for variable/DataArray creation.

        The single point of truth for the model's variate dims: operational
        variables use ``coords(time=True)`` (the flat time axis), investment
        variables ``coords(period=True)``. When a new variate dim (e.g.
        ``scenario``) is added, extend this method once and every call site
        picks it up.

        Args:
            time: Include the flat time coordinate.
            period: Include the period coordinate (no-op in single-period mode).
        """
        result: dict[str, xr.DataArray] = {}
        if time:
            result['time'] = self.time
        if period and self.period is not None:
            result['period'] = self.period
        return result

    # -- Episode boundaries (flat time axis) -----------------------------

    @cached_property
    def episodes(self) -> Episodes:
        """Episode partition of the flat time axis.

        An episode is a maximal run of consecutive-in-modeled-time timesteps;
        temporal-coupling constraints (SOC recursion, status windows, ramps)
        never chain across its boundaries. Boundaries fall where
        ``time_period`` changes — one episode per period; single-period
        models are one episode. See docs/design/time-index.md.
        """
        if self.time_period is None:
            return Episodes.single(self.time)
        return Episodes.from_changes(self.time_period)

    @cached_property
    def _time_period_indexer(self) -> xr.DataArray:
        """(time,) period label per timestep, without ride-along coords."""
        assert self.time_period is not None
        return xr.DataArray(self.time_period.values, dims=['time'], coords={'time': self.time})

    def period_grouper(self, name: str = 'period') -> xr.DataArray:
        """Grouper mapping each timestep to its period label, for groupby.

        Args:
            name: Name of the resulting group dimension.

        Raises:
            ValueError: In single-period mode (nothing to group by).
        """
        if self.time_period is None:
            raise ValueError('period_grouper requires a multi-period model')
        return self._time_period_indexer.rename(name)

    def map_to_time(self, obj: Any) -> Any:
        """Expand a period-dimensioned object onto the flat time axis.

        ``obj[..., period]`` becomes ``obj[..., time]`` with each timestep
        carrying its period's value (vectorized indexing). Objects without a
        ``period`` dim pass through unchanged, as does everything in
        single-period mode.

        Args:
            obj: xr.DataArray or linopy Variable/LinearExpression.
        """
        if self.time_period is None or 'period' not in obj.dims:
            return obj
        result = obj.sel(period=self._time_period_indexer)
        if not hasattr(result, 'drop_vars'):
            # linopy Variable lacks drop_vars; lift to a LinearExpression so
            # the stray coord cannot trigger outer-join warnings downstream.
            result = 1.0 * result
        return result.drop_vars('period', errors='ignore')

    def _reduce_time(self, obj: Any, how: str) -> Any:
        """Reduce over time within each period.

        Multi-period: groupby ``time_period`` → result on the ``period`` dim.
        Single-period: plain ``.{how}('time')``.

        Args:
            obj: xr.DataArray or linopy Variable/LinearExpression with a time dim.
            how: Reduction method name (``'sum'``, ``'mean'``).
        """
        if self.time_period is None:
            return getattr(obj, how)('time')
        assert self.period is not None
        result = getattr(obj.groupby(self.period_grouper()), how)()
        return result.sel(period=self.period.values)

    def sum_time(self, obj: Any) -> Any:
        """Sum over time within each period (see :meth:`_reduce_time`)."""
        return self._reduce_time(obj, 'sum')

    def mean_time(self, obj: xr.DataArray) -> xr.DataArray:
        """Mean over time within each period (see :meth:`_reduce_time`)."""
        return self._reduce_time(obj, 'mean')

    def _boundary_time(self, obj: Any, positions: np.ndarray, single: int) -> Any:
        """Value at one boundary timestep of each period, on the period dim.

        Multi-period: ``(…, time)`` → ``(…, period)``; single-period: plain
        ``isel(time=single)`` (period-less).
        """
        if self.time_period is None:
            return obj.isel(time=single)
        assert self.period is not None
        return obj.isel(time=positions.tolist()).rename({'time': 'period'}).assign_coords(period=self.period.values)

    def first_time(self, obj: Any) -> Any:
        """Value at each period's first timestep (see :meth:`_boundary_time`)."""
        return self._boundary_time(obj, self.episodes.start_positions, 0)

    def last_time(self, obj: Any) -> Any:
        """Value at each period's last timestep (see :meth:`_boundary_time`)."""
        return self._boundary_time(obj, self.episodes.last_positions, -1)

    def varies_within_period(self, obj: xr.DataArray) -> xr.DataArray:
        """Mask of elements whose values change within some period.

        Compares against each period's first value exactly rather than the
        per-period mean, so a constant factor cannot register as varying
        through float error. Values that differ only across periods do not
        count as varying.
        """
        first = self.map_to_time(self.first_time(obj))
        return (obj != first).any('time')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        data_vars: dict[str, xr.DataArray] = {'dt': self.dt, 'weights': self.weights}
        if self.period is not None:
            data_vars['period'] = self.period
        if self.period_weights is not None:
            data_vars['period_weights'] = self.period_weights
        if self.time_period is not None:
            data_vars['time_period'] = self.time_period
        return xr.Dataset(data_vars)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with dt, weights, and optional period fields.
        """
        dt = ds['dt']
        time_idx = dt.coords['time']
        return cls(
            time=time_idx,
            dt=dt,
            weights=ds['weights'],
            period=ds.get('period', None),
            period_weights=ds.get('period_weights', None),
            time_period=ds.get('time_period', None),
        )

    @classmethod
    def build(
        cls,
        timesteps: Timesteps | Mapping[int, Timesteps],
        dt: float | list[float] | None = None,
        periods: list[int] | pd.Index | None = None,
        period_weights: list[float] | None = None,
    ) -> Self:
        """Build Dims from user timesteps and optional periods.

        Three input modes:

        - Plain index, no periods: single-period, unchanged semantics.
        - Plain index + ``periods``: uniform multi-period — the index is
          replicated per period; datetime labels shift by each period's
          year gap to the first period (the first period keeps the base
          labels), integer labels offset by the span.
        - ``{period: index}`` mapping: ragged multi-period — each period has
          its own (datetime) grid; resolutions and lengths may differ.

        Args:
            timesteps: Time index, or mapping of period label to per-period index.
            dt: Timestep duration in hours. Auto-derived if None. For plain
                indexes a list must match the input index length; for mappings
                it must match the flat length.
            periods: Integer period labels for uniform multi-period mode.
                Must be None when *timesteps* is a mapping.
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        from fluxopt.types import compute_dt as _compute_dt

        period_idx: pd.Index | None = None
        period_weights_da: xr.DataArray | None = None
        time_period_vals: np.ndarray | None = None

        if isinstance(timesteps, Mapping):
            if periods is not None:
                raise ValueError('periods must not be given when timesteps is a {period: index} mapping')
            period_idx, period_weights_da = _compute_period_weights(list(timesteps.keys()), period_weights)
            segments: list[TimeIndex] = []
            for p in period_idx:
                seg = normalize_timesteps(timesteps[int(p)])
                if not isinstance(seg, pd.DatetimeIndex):
                    raise TypeError(
                        f'Ragged multi-period timesteps require datetime indexes, got {seg.dtype} for period {int(p)}'
                    )
                segments.append(seg)
            flat = _concat_time_indexes(segments)
            if dt is None:
                dt_vals = np.concatenate([_compute_dt(seg, None).values for seg in segments])
            elif isinstance(dt, (int, float)):
                dt_vals = np.full(len(flat), float(dt))
            else:
                if len(dt) != len(flat):
                    raise ValueError(f'dt length {len(dt)} does not match flat timesteps length {len(flat)}')
                dt_vals = np.array(dt, dtype=float)
            time_period_vals = np.concatenate(
                [np.full(len(seg), int(p)) for p, seg in zip(period_idx, segments, strict=True)]
            )
        elif periods is not None:
            base = normalize_timesteps(timesteps)
            period_idx, period_weights_da = _compute_period_weights(periods, period_weights)
            base_dt = _compute_dt(base, dt)
            segments = _replicate_time_index(base, period_idx)
            flat = _concat_time_indexes(segments)
            dt_vals = np.tile(base_dt.values, len(period_idx))
            time_period_vals = np.repeat(period_idx.to_numpy(), len(base))
        else:
            flat = normalize_timesteps(timesteps)
            dt_vals = _compute_dt(flat, dt).values

        time_coord = xr.DataArray(flat, dims=['time'], coords={'time': flat})
        dt_da = xr.DataArray(dt_vals, dims=['time'], coords={'time': flat}, name='dt')
        weights = xr.DataArray(np.ones(len(flat)), dims=['time'], coords={'time': flat}, name='weight')

        period_da: xr.DataArray | None = None
        time_period_da: xr.DataArray | None = None
        if period_idx is not None:
            period_da = xr.DataArray(period_idx.values, dims=['period'], coords={'period': period_idx})
            time_period_da = xr.DataArray(time_period_vals, dims=['time'], coords={'time': flat}, name='time_period')

        return cls(
            time=time_coord,
            dt=dt_da,
            weights=weights,
            period=period_da,
            period_weights=period_weights_da,
            time_period=time_period_da,
        )


_CONTAINER_TYPES: dict[str, type] = {
    'sizing': SizingData,
    'status': StatusData,
    'invest': InvestmentData,
    'cstatus': StatusData,
}


def _table_containers(obj: DataclassInstance) -> dict[str, Any]:
    """Nested container fields of a table object that are present (not None)."""
    return {
        f.name: getattr(obj, f.name)
        for f in fields(obj)
        if f.name in _CONTAINER_FIELD_NAMES and getattr(obj, f.name) is not None
    }


def _nc_group_paths(p: Path) -> set[str]:
    """All group paths present in a netCDF file (e.g. ``{'model', 'model/flows', ...}``).

    Group *absence* is decided from this listing, so real I/O errors while
    reading a present group propagate instead of being mistaken for absence.
    """
    import netCDF4

    def walk(grp: Any, prefix: str) -> set[str]:
        out: set[str] = set()
        for name, sub in grp.groups.items():
            path = f'{prefix}{name}'
            out.add(path)
            out |= walk(sub, path + '/')
        return out

    with netCDF4.Dataset(p) as nc:
        return walk(nc, '')


def _load_containers(p: Path, group: str, cls: type[DataclassInstance], present: set[str]) -> dict[str, Any]:
    """Load a table's nested container sub-groups from netCDF, keyed by field name.

    Args:
        p: File path.
        group: The table's group path (e.g. ``'model/flows'``).
        cls: Table dataclass whose container fields to look for.
        present: Group paths that exist in the file (see :func:`_nc_group_paths`).
    """
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in _CONTAINER_FIELD_NAMES and f'{group}/{f.name}' in present:
            ds = xr.load_dataset(p, group=f'{group}/{f.name}', engine='netcdf4')
            out[f.name] = _CONTAINER_TYPES[f.name].from_dataset(ds)
    return out


@dataclass
class _TimeMapper:
    """Converts user operational inputs onto the flat time axis.

    Accepted input shapes (multi-period):

    - scalar — broadcast to all timesteps
    - ``{period: Variate}`` mapping — each entry aligned to that period's grid
    - ``(period,)`` DataArray / period-named Series — each period's value
      repeated over its timesteps (bare lists never mean per-period)
    - ``(time,)`` of flat length — used as-is
    - ``(time,)`` matching the within-period grid — tiled per period (uniform mode)
    - ``(time, period)`` DataArray/DataFrame with within-period time labels —
      flattened in period order (uniform mode)

    No resampling ever happens; mismatched grids raise. Single-period mode
    delegates to :func:`fluxopt.types.as_dataarray` unchanged.
    """

    dims: Dims
    base_time: pd.Index | None = None  # within-period labels (uniform mode only)

    @cached_property
    def time(self) -> pd.Index:
        """Flat time index."""
        return pd.Index(self.dims.time.values, name='time')

    @cached_property
    def _period_labels(self) -> list[int]:
        assert self.dims.period is not None
        return [int(p) for p in self.dims.period.values]

    @cached_property
    def _segments(self) -> list[tuple[int, slice]]:
        """Per-period (label, positional slice) pairs in period order."""
        starts = self.dims.episodes.start_positions
        lasts = self.dims.episodes.last_positions
        return [(p, slice(int(s), int(e) + 1)) for p, s, e in zip(self._period_labels, starts, lasts, strict=True)]

    def to_flat(self, value: Variate | Mapping[int, Variate], name: str = 'value') -> xr.DataArray:
        """Convert a user operational input to a (time,) DataArray on the flat axis.

        Args:
            value: Operational input (see class docstring for accepted shapes).
            name: Name for the resulting DataArray.
        """
        flat_idx = self.time
        if self.dims.period is None:
            if isinstance(value, Mapping):
                raise TypeError(f'{name}: per-period mapping given, but the model has no periods')
            return as_dataarray(value, {'time': flat_idx}, name=name)

        if isinstance(value, Mapping):
            return self._from_mapping(value, name)
        if isinstance(value, (int, float)):
            return as_dataarray(float(value), {'time': flat_idx}, name=name)
        if isinstance(value, pd.DataFrame):
            named = {a.name for a in value.axes}
            if named != {'time', 'period'}:
                raise ValueError(
                    f'{name}: DataFrame axes must be named time/period (got {[a.name for a in value.axes]!r})'
                )
            return self._from_dataarray(xr.DataArray(value), name)
        if isinstance(value, pd.Series):
            if value.index.name in ('time', 'period'):
                return self._from_dataarray(xr.DataArray(value), name)
            if value.index.name is not None:
                raise ValueError(
                    f'{name}: Series index name {value.index.name!r} is not a model dim; '
                    f"use 'time' or 'period', or an unnamed index for positional matching."
                )
            return self._from_unnamed(np.asarray(value.values, dtype=float), name)
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError(f'{name}: np.ndarray must be 1-D (got ndim={value.ndim})')
            return self._from_unnamed(value.astype(float), name)
        if isinstance(value, list):
            return self._from_unnamed(np.asarray(value, dtype=float), name)
        if isinstance(value, xr.DataArray):
            return self._from_dataarray(value, name)
        raise TypeError(f'{name}: unsupported input type {type(value)}')

    def _from_mapping(self, value: Mapping[int, Variate], name: str) -> xr.DataArray:
        """Align a ``{period: entry}`` mapping segment by segment.

        Each entry may be labeled by its period's segment of the flat index
        or — in uniform mode, where segment labels are shifted into each
        period's calendar internally — by the base grid the user actually
        knows.
        """
        labels = self._period_labels
        keys = {int(k) for k in value}
        if keys != set(labels):
            raise ValueError(f'{name}: mapping keys {sorted(keys)} do not match periods {labels}')
        flat_idx = self.time
        parts: list[np.ndarray] = []
        for p, slc in self._segments:
            seg_idx = flat_idx[slc]
            try:
                arr = as_dataarray(value[p], {'time': seg_idx}, name=name).values
            except ValueError:
                if self.base_time is None or len(self.base_time) != len(seg_idx):
                    raise
                arr = as_dataarray(value[p], {'time': self.base_time}, name=name).values
            parts.append(arr)
        return xr.DataArray(np.concatenate(parts), dims=['time'], coords={'time': flat_idx}, name=name)

    def _from_dataarray(self, da: xr.DataArray, name: str) -> xr.DataArray:
        flat_idx = self.time
        dset = {str(d) for d in da.dims}
        if not dset <= {'time', 'period'}:
            raise ValueError(f'{name}: dims {sorted(dset - {"time", "period"})} not in (time, period)')

        if dset == set():
            return as_dataarray(float(da.values), {'time': flat_idx}, name=name)

        if dset == {'period'}:
            self._check_period_coord(da, name)
            if 'period' not in da.coords:
                da = da.assign_coords(period=self._period_labels)
            expanded = self.dims.map_to_time(da.astype(float))
            return expanded.rename(name)

        if dset == {'time'}:
            if 'time' not in da.coords:
                return self._from_unnamed(da.values.astype(float), name)
            in_idx = pd.Index(da.coords['time'].values)
            if in_idx.equals(flat_idx):
                return da.astype(float).rename(name)
            if self.base_time is not None and in_idx.equals(self.base_time):
                return self._tile(da.values.astype(float), name)
            raise ValueError(
                f'{name}: time coord matches neither the flat time index nor the '
                f'within-period grid. Align to the flat index or pass a '
                f'{{period: series}} mapping.'
            )

        # (time, period)
        self._check_period_coord(da, name)
        if self.base_time is None:
            raise ValueError(
                f'{name}: (time, period) input requires a uniform grid; '
                f'pass a {{period: series}} mapping for ragged periods.'
            )
        in_idx = pd.Index(da.coords['time'].values)
        if not in_idx.equals(self.base_time):
            raise ValueError(f'{name}: time coord does not match the within-period grid')
        if 'period' not in da.coords:
            da = da.assign_coords(period=self._period_labels)
        ordered = da.astype(float).transpose('period', 'time').sel(period=self._period_labels)
        return xr.DataArray(ordered.values.reshape(-1), dims=['time'], coords={'time': flat_idx}, name=name)

    def _from_unnamed(self, arr: np.ndarray, name: str) -> xr.DataArray:
        """Match a bare 1-D array to a time grid.

        Unnamed 1-D input is always a *time profile* — flat length or the
        within-period grid (tiled). It never means per-period values: those
        require a named form ({period: value} mapping, (period,) DataArray,
        or a pd.Series with index.name='period'), so no length collision can
        change the meaning of a plain list.
        """
        flat_idx = self.time
        n = len(arr)
        if n == len(flat_idx):
            return xr.DataArray(np.asarray(arr, dtype=float), dims=['time'], coords={'time': flat_idx}, name=name)
        if self.base_time is not None and n == len(self.base_time):
            return self._tile(np.asarray(arr, dtype=float), name)
        options = [f'flat time({len(flat_idx)})']
        if self.base_time is not None:
            options.append(f'within-period time({len(self.base_time)})')
        raise ValueError(
            f'{name}: length {n} matches no time grid: {", ".join(options)}. '
            f'For per-period values pass a {{period: value}} mapping, a (period,) '
            f"xr.DataArray, or a pd.Series with index.name='period'."
        )

    def _tile(self, arr: np.ndarray, name: str) -> xr.DataArray:
        assert self.dims.period is not None
        flat_idx = self.time
        tiled = np.tile(arr, len(self.dims.period))
        return xr.DataArray(tiled, dims=['time'], coords={'time': flat_idx}, name=name)

    def _check_period_coord(self, da: xr.DataArray, name: str) -> None:
        if 'period' in da.coords:
            given = pd.Index([int(p) for p in da.coords['period'].values])
            if not given.equals(pd.Index(self._period_labels)):
                raise ValueError(
                    f'{name}: period coord {list(given)} does not match model periods {self._period_labels}'
                )
        elif da.sizes['period'] != len(self._period_labels):
            raise ValueError(f'{name}: period dim length {da.sizes["period"]} != {len(self._period_labels)} periods')


@dataclass
class ModelData:
    flows: FlowsData
    carriers: CarriersData
    converters: ConvertersData | None  # None when no linear converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dims: Dims
    piecewise: PiecewiseData | None = None  # None when no piecewise converters

    def __post_init__(self) -> None:
        """Validate cross-table id consistency (also on netCDF reload).

        Each table validates itself in its own ``__post_init__``; this checks
        that ids referenced *between* tables resolve, so a tampered or
        hand-edited file fails here instead of as a ``KeyError`` deep in
        model building.
        """
        flow_ids = set(map(str, self.flows.size.coords['flow'].values))

        def check_flows(ids: list[str], what: str) -> None:
            if unknown := sorted(set(ids) - flow_ids):
                raise ValueError(f'{what} references unknown flow id(s) {unknown}')

        def coord_ids(da: xr.DataArray) -> list[str]:
            return [str(v) for v in da.coords[da.dims[0]].values]

        check_flows([str(v) for v in self.carriers.flow_coeff.coords['flow'].values], 'carriers.flow_coeff')
        if self.flows.sizing is not None:
            check_flows(coord_ids(self.flows.sizing.min), 'flows.sizing')
        if self.flows.invest is not None:
            check_flows(coord_ids(self.flows.invest.min), 'flows.invest')
        if self.flows.status is not None:
            check_flows(coord_ids(self.flows.status.uptime_min), 'flows.status')
        if self.flows.cstatus is not None and self.flows.cstatus.governed_flows is not None:
            governed = [str(v) for v in self.flows.cstatus.governed_flows.values.ravel() if str(v)]
            check_flows(governed, 'flows.cstatus.governed_flows')
        if self.converters is not None:
            check_flows([str(v) for v in self.converters.pair_flow.values], 'converters.pair_flow')
        if self.piecewise is not None:
            check_flows([str(v) for v in self.piecewise.pair_flow.values], 'piecewise.pair_flow')
        if self.storages is not None:
            check_flows([str(v) for v in self.storages.charge_flow.values], 'storages.charge_flow')
            check_flows([str(v) for v in self.storages.discharge_flow.values], 'storages.discharge_flow')
            storage_ids = set(map(str, self.storages.capacity.coords['storage'].values))
            for container, what in (
                (self.storages.sizing, 'storages.sizing'),
                (self.storages.invest, 'storages.invest'),
            ):
                if container is not None and (unknown := sorted(set(coord_ids(container.min)) - storage_ids)):
                    raise ValueError(f'{what} references unknown storage id(s) {unknown}')

        effect_ids = set(map(str, self.effects.total_min.coords['effect'].values))
        coeff_effects = set(map(str, self.flows.effect_coeff.coords['effect'].values))
        if coeff_effects != effect_ids:
            raise ValueError(
                f'flows.effect_coeff effect coordinate {sorted(coeff_effects)} does not match '
                f'the effects table {sorted(effect_ids)}'
            )

    def to_netcdf(self, path: str | Path, *, mode: Literal['w', 'a'] = 'a') -> None:
        """Write model data as NetCDF groups under ``/model/``.

        Args:
            path: Output file path.
            mode: Write mode ('w' to overwrite, 'a' to append).
        """
        p = Path(path)
        dataset_fields: dict[
            str,
            FlowsData | CarriersData | ConvertersData | EffectsData | StoragesData | PiecewiseData | None,
        ] = {
            'flows': self.flows,
            'carriers': self.carriers,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
            'piecewise': self.piecewise,
        }
        current_mode = mode
        for name, obj in dataset_fields.items():
            if obj is not None:
                obj.to_dataset().to_netcdf(p, mode=current_mode, group=_NC_GROUPS[name], engine='netcdf4')
                current_mode = 'a'
                for cname, container in _table_containers(obj).items():
                    container.to_dataset().to_netcdf(p, mode='a', group=f'{_NC_GROUPS[name]}/{cname}', engine='netcdf4')
        self.dims.to_dataset().to_netcdf(p, mode=current_mode, group='model/meta', engine='netcdf4')

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ModelData:
        """Read model data from NetCDF groups.

        Args:
            path: Input file path.

        Raises:
            OSError: If no model data groups found in the file.
            ValueError: On Windows when reading a non-ASCII path (netcdf4 limitation).
        """
        p = Path(path)
        try:
            present = _nc_group_paths(p)
        except OSError as e:
            _raise_netcdf_read_error(p, e)
        if 'model/meta' not in present:
            raise OSError(f'No fluxopt model data found in {p} (missing model/meta group)')
        meta = xr.load_dataset(p, group='model/meta', engine='netcdf4')

        datasets: dict[str, xr.Dataset] = {
            name: xr.load_dataset(p, group=group, engine='netcdf4') if group in present else xr.Dataset()
            for name, group in _NC_GROUPS.items()
        }

        flows = FlowsData.from_dataset(datasets['flows'], _load_containers(p, _NC_GROUPS['flows'], FlowsData, present))
        carriers = CarriersData.from_dataset(datasets['carriers'])
        converters = ConvertersData.from_dataset(datasets['converters']) if datasets['converters'].data_vars else None
        effects = EffectsData.from_dataset(datasets['effects'])
        storages = (
            StoragesData.from_dataset(
                datasets['storages'], _load_containers(p, _NC_GROUPS['storages'], StoragesData, present)
            )
            if datasets['storages'].data_vars
            else None
        )
        piecewise = PiecewiseData.from_dataset(datasets['piecewise']) if datasets['piecewise'].data_vars else None

        return cls(
            flows=flows,
            carriers=carriers,
            converters=converters,
            effects=effects,
            storages=storages,
            dims=Dims.from_dataset(meta),
            piecewise=piecewise,
        )

    @classmethod
    def build(
        cls,
        timesteps: Timesteps | Mapping[int, Timesteps],
        carriers: list[Carrier],
        effects: list[Effect],
        ports: list[Port],
        converters: list[Converter] | None = None,
        storages: list[Storage] | None = None,
        dt: float | list[float] | None = None,
        periods: list[int] | pd.Index | None = None,
        period_weights: list[float] | None = None,
    ) -> Self:
        """Build ModelData from element objects.

        Args:
            timesteps: Time index for the optimization horizon, or a
                ``{period: index}`` mapping for ragged multi-period grids
                (each period may have its own resolution and length).
            carriers: Carrier declarations.
            effects: Effects to track.
            ports: System boundary ports.
            converters: Linear converters.
            storages: Energy storages.
            dt: Timestep duration in hours. Auto-derived if None.
            periods: Integer period labels for multi-period optimization
                (uniform grid mode; forbidden with a timesteps mapping).
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        from fluxopt.elements import PENALTY_EFFECT_ID, Effect

        converters = converters or []
        stor_list = storages or []

        if not any(e.id == PENALTY_EFFECT_ID for e in effects):
            effects = [*effects, Effect(id=PENALTY_EFFECT_ID)]

        flows, carrier_coeff = _collect_flows(ports, converters, stor_list)
        validate_system(carriers=carriers, effects=effects, ports=ports, converters=converters, storages=stor_list)

        dims = Dims.build(timesteps, dt=dt, periods=periods, period_weights=period_weights)
        base_time = (
            normalize_timesteps(timesteps) if dims.period is not None and not isinstance(timesteps, Mapping) else None
        )
        mapper = _TimeMapper(dims, base_time=base_time)

        # Scalar dt for prior duration computation. Pre-horizon steps have no
        # dt of their own, so the first timestep's duration stands in; on a
        # non-uniform grid that is an assumption the user should know about.
        dt_scalar = float(dims.dt.values[0])
        if any(bf.flow.prior_rates is not None for bf in flows) and not np.allclose(dims.dt.values, dt_scalar):
            warnings.warn(
                f'prior_rates with non-uniform dt: pre-horizon status durations assume the first '
                f'timestep duration ({dt_scalar} h) for every prior step. If your prior steps had '
                f'different durations, adjust prior_rates to compensate.',
                UserWarning,
                stacklevel=2,
            )
        period_idx = pd.Index(dims.period.values) if dims.period is not None else None

        comp_status_items: list[tuple[str, Status, list[str]]] = [
            (s.id, s.status, [s._charging_id, s._discharging_id]) for s in stor_list if s.status is not None
        ]
        comp_status_items.extend(
            (c.id, c.conversion.status, [bf.id for bf in c._qualified_flows()])
            for c in converters
            if c.conversion is not None and c.conversion.status is not None
        )

        flows_data = FlowsData.build(
            flows,
            mapper,
            effects,
            dt=dt_scalar,
            period=period_idx,
            component_status_items=comp_status_items,
        )
        carriers_data = CarriersData.build(carriers, flows, carrier_coeff)
        converters_data = ConvertersData.build(converters, mapper)
        effects_data = EffectsData.build(effects, mapper, period=period_idx)
        storages_data = StoragesData.build(stor_list, mapper, effects, period=period_idx)
        piecewise_data = PiecewiseData.build(converters, mapper)

        return cls(
            flows=flows_data,
            carriers=carriers_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dims=dims,
            piecewise=piecewise_data,
        )


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> tuple[list[_BoundFlow], dict[str, float]]:
    """Gather qualified flows from every component with carrier-balance signs.

    Args:
        ports: System boundary ports.
        converters: Converter components.
        storages: Storage components.

    Returns:
        Tuple of (flows, carrier_coeff) where carrier_coeff maps qualified
        flow id to +1 (produces into carrier) or -1 (consumes from carrier).
    """
    flows: list[_BoundFlow] = []
    for comp in (*ports, *converters, *(storages or [])):
        flows.extend(comp._qualified_flows())
    return flows, {bf.id: float(bf.sign) for bf in flows}
