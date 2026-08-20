from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Self, get_args

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from fluxopt.contract import BoundType, Dim
from fluxopt.types import PiecewiseMethod, as_dataarray, fast_concat, normalize_timesteps
from fluxopt.validation import validate_system

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage, _BoundFlow
    from fluxopt.types import TimeIndex, Timesteps


def compute_previous_duration(
    previous_status: xr.DataArray,
    target_state: int,
    dt: xr.DataArray | float,
) -> float:
    """Compute consecutive duration of target_state at end of previous_status.

    Walks backward through previous_status counting timesteps that match
    the target state, then multiplies by timestep duration.

    Args:
        previous_status: Previous status values (time dimension).
        target_state: 1 for active (uptime), 0 for inactive (downtime).
        dt: Duration per timestep (scalar or DataArray).

    Returns:
        Total duration in target state at end of previous period.
    """
    values = previous_status.values
    count = 0
    for v in reversed(values):
        if (target_state == 1 and v > 0) or (target_state == 0 and v == 0):
            count += 1
        else:
            break

    if isinstance(dt, xr.DataArray):
        return float(dt.values[-count:].sum()) if count > 0 else 0.0
    return dt * count


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
    """Sizing (capacity optimization) parameters for one entity family.

    Two frames rather than five arrays. `bounds` is one row per entity — the
    scalars a `Sizing` declares — and `effects` is one row per effect an
    entity actually charges, which is the shape #306 and #311 established for
    coefficients: the pairs that exist, not the product they sit in.
    """

    #: (entity, size_min, size_max, mandatory) — one row per sized entity
    bounds: pl.DataFrame
    #: (entity, effect[, period], per_size, fixed) — declared pairs only
    effects: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The entities this table sizes, in declaration order."""
        return self.bounds['entity'].to_list()

    def __post_init__(self) -> None:
        """Re-check the bounds `Sizing` already refuses, for a reloaded file.

        The element layer is where this rule is enforced — there it fires on
        the value the user wrote, naming the field. A hand-edited netCDF never
        passed through `Sizing` at all, which is the only reason to say it
        twice. See docs/design/validation-layers.md.
        """
        bad = self.bounds.filter((pl.col('size_min') < 0) | (pl.col('size_max') < pl.col('size_min')))
        if len(bad):
            raise ValueError(f'Sizing bounds are not orderable on {bad["entity"].to_list()}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Sizing]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Collect Sizing objects into frames, or None if there are none.

        Args:
            items: Pairs of (element_id, Sizing).
            effect_ids: Declared effects; unused now that the pairs name their
                own, kept so every container builds the same way.
            dim: Historic dimension name; the frame keys on ``entity``.
            period: Period index for period-varying effects.
        """
        del effect_ids, dim
        if not items:
            return None

        periods = list(range(len(period))) if period is not None else [0]
        n_periods = len(periods)

        entities: list[str] = []
        effects: list[str] = []
        per_size: list[float] = []
        fixed: list[float] = []
        for item_id, s in items:
            for ek in sorted(set(s.effects_per_size) | set(s.effects_fixed)):
                entities.extend([item_id] * n_periods)
                effects.extend([ek] * n_periods)
                per_size.extend(_per_period(s.effects_per_size.get(ek, 0.0), n_periods))
                fixed.extend(_per_period(s.effects_fixed.get(ek, 0.0), n_periods))

        return cls(
            bounds=pl.DataFrame(
                {
                    'entity': [i for i, _ in items],
                    'size_min': [float(z.size_min) for _, z in items],
                    'size_max': [float(z.size_max) for _, z in items],
                    'mandatory': [bool(z.mandatory) for _, z in items],
                }
            ),
            effects=pl.DataFrame(
                {
                    'entity': entities,
                    'effect': effects,
                    'period': periods * (len(entities) // n_periods),
                    'per_size': per_size,
                    'fixed': fixed,
                },
                schema={
                    'entity': pl.String,
                    'effect': pl.String,
                    'period': pl.Int64,
                    'per_size': pl.Float64,
                    'fixed': pl.Float64,
                },
            ),
        )


def _per_period(value: Any, n_periods: int) -> list[float]:
    """A lump coefficient's value in each period.

    A scalar applies to every period; a per-period sequence gives one value
    each. Built as a list rather than a row per period because the frame is
    assembled column-wise — a dict per row costs more than the data.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    return [float(arr[0])] * n_periods if arr.size == 1 else [float(v) for v in arr[:n_periods]]


@dataclass
class InvestmentData:
    """Investment (build-timing optimization) parameters for one entity family.

    Absence is a missing row, not a sentinel. An entity with no `lifetime`
    has no row in `lifetime` and is alive forever; a coefficient nobody
    declared has no row in `effects` and charges nothing. That is the
    convention frames make available and the one lpspec's language already
    uses, so the two now agree.
    """

    #: (entity, size_min, size_max, mandatory, prior_size) — one row each
    bounds: pl.DataFrame
    #: (entity, periods) — only those that expire. No row means forever.
    lifetime: pl.DataFrame
    #: (entity, effect, period, per_size_at_build, fixed_at_build,
    #: per_size_recurring, fixed_recurring) — declared pairs only
    effects: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The entities this table invests in, in declaration order."""
        return self.bounds['entity'].to_list()

    def __post_init__(self) -> None:
        """Re-check the bounds `Investment` already refuses, for a reloaded file.

        A reload guard — see docs/design/validation-layers.md.
        """
        bad = self.bounds.filter(
            (pl.col('size_min') < 0) | (pl.col('size_max') < pl.col('size_min')) | (pl.col('prior_size') < 0)
        )
        if len(bad):
            raise ValueError(f'Investment bounds are not orderable on {bad["entity"].to_list()}')
        if len(short := self.lifetime.filter(pl.col('periods') <= 0)):
            raise ValueError(f'Investment.lifetime must be positive on {short["entity"].to_list()}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Investment]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Collect Investment objects into frames, or None if there are none.

        Args:
            items: Pairs of (element_id, Investment).
            effect_ids: Declared effects; the pairs name their own.
            dim: Historic dimension name; the frames key on ``entity``.
            period: Period index for period-varying effects.
        """
        del effect_ids, dim
        if not items:
            return None

        periods = list(range(len(period))) if period is not None else [0]
        n_periods = len(periods)
        kinds = (
            ('per_size_at_build', 'effects_per_size_at_build'),
            ('fixed_at_build', 'effects_fixed_at_build'),
            ('per_size_recurring', 'effects_per_size_recurring'),
            ('fixed_recurring', 'effects_fixed_recurring'),
        )

        entities: list[str] = []
        effects: list[str] = []
        values: dict[str, list[float]] = {column: [] for column, _ in kinds}
        for item_id, inv in items:
            declared: set[str] = set()
            for _, field in kinds:
                declared |= set(getattr(inv, field))
            for ek in sorted(declared):
                entities.extend([item_id] * n_periods)
                effects.extend([ek] * n_periods)
                for column, field in kinds:
                    values[column].extend(_per_period(getattr(inv, field).get(ek, 0.0), n_periods))

        with_lifetime = [(i, inv) for i, inv in items if inv.lifetime is not None]
        return cls(
            bounds=pl.DataFrame(
                {
                    'entity': [i for i, _ in items],
                    'size_min': [float(v.size_min) for _, v in items],
                    'size_max': [float(v.size_max) for _, v in items],
                    'mandatory': [bool(v.mandatory) for _, v in items],
                    'prior_size': [float(v.prior_size) for _, v in items],
                }
            ),
            lifetime=pl.DataFrame(
                {
                    'entity': [i for i, _ in with_lifetime],
                    'periods': [int(v.lifetime) for _, v in with_lifetime],  # type: ignore[arg-type]
                },
                schema={'entity': pl.String, 'periods': pl.Int64},
            ),
            effects=pl.DataFrame(
                {
                    'entity': entities,
                    'effect': effects,
                    'period': periods * (len(entities) // n_periods),
                    **values,
                },
                schema={
                    'entity': pl.String,
                    'effect': pl.String,
                    'period': pl.Int64,
                    **{column: pl.Float64 for column, _ in kinds},
                },
            ),
        )


@dataclass
class StatusData:
    """Binary on/off behavior arrays for one entity family (flow or component)."""

    uptime_min: xr.DataArray  # (dim,)
    uptime_max: xr.DataArray  # (dim,)
    downtime_min: xr.DataArray  # (dim,)
    downtime_max: xr.DataArray  # (dim,)
    initial: xr.DataArray  # (dim,) — NaN = free
    effects_running: xr.DataArray  # (dim, effect, time, period?) — dense, zero where absent
    effects_startup: xr.DataArray  # (dim, effect, time, period?) — dense, zero where absent
    previous_uptime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior
    previous_downtime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior

    def __post_init__(self) -> None:
        """Re-check the durations `Status` already refuses, for a reloaded file.

        A reload guard — see docs/design/validation-layers.md.
        """
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
        time: TimeIndex,
        dim: str,
        prior_rates_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Validate Status objects and collect into DataArrays, or None if empty.

        Args:
            items: Pairs of (id, Status).
            effect_ids: Known effect ids for validation.
            time: Time index for effect arrays.
            dim: Dimension name for the resulting arrays.
            prior_rates_map: Item id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for period-varying effects.
        """
        if not items:
            return None

        prior_rates_map = prior_rates_map or {}
        tmpl = _effect_template({'effect': effect_ids, 'time': time}, period)

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
                er.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            er_slices.append(er)

            es = tmpl.zeros()
            for ek, ev in s.effects_per_startup.items():
                es.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            es_slices.append(es)

        coords = {dim: ids}
        status_idx = pd.Index(ids, name=dim)

        prev_up_arr = np.array(prev_ups)
        prev_down_arr = np.array(prev_downs)

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
        )


@dataclass
class FlowsData:
    bound_type: xr.DataArray  # (flow,) — BoundType.UNSIZED | BoundType.BOUNDED | BoundType.PROFILE
    rel_lb: xr.DataArray  # (flow, time[, period])
    rel_ub: xr.DataArray  # (flow, time[, period])
    fixed_profile: xr.DataArray  # (flow, time[, period]) — NaN where not fixed
    size: xr.DataArray  # (flow,) — NaN for unsized
    #: One row per (flow, effect) a flow actually charges, each carrying its
    #: own ``(time[, period])`` series. Dense over the pair product it is not:
    #: on the stress reference system that product is 443 MB at 4% live.
    effect_pair_flow: xr.DataArray  # (pair,) — flow id
    effect_pair_effect: xr.DataArray  # (pair,) — effect id
    effect_pair_coeff: xr.DataArray  # (pair, time[, period])
    #: (flow,) — every flow, in order. The pair table names only the flows
    #: that charge something, so the roster cannot be read off it.
    flow_id: xr.DataArray
    flow_hours_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    flow_hours_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    ramp_up: xr.DataArray | None = None  # (flow, time[, period]) — NaN = no limit [1/h]
    ramp_down: xr.DataArray | None = None  # (flow, time[, period]) — NaN = no limit [1/h]
    sizing: SizingData | None = None  # dim Dim.SIZING_FLOW
    status: StatusData | None = None  # dim Dim.STATUS_FLOW
    invest: InvestmentData | None = None  # dim Dim.INVEST_FLOW
    cstatus: StatusData | None = None  # dim Dim.CSTATUS_COMPONENT, entity coord 'component'
    #: (flow,) — the component whose Status governs this flow, '' where none.
    #: A fact about the flow, so it sits here rather than as a ragged padded
    #: matrix on the component's Status, which is what it used to be.
    governed_by: xr.DataArray | None = None

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
            extra |= set(self.sizing.ids)
        if self.invest is not None:
            extra |= set(self.invest.ids)
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
        time: TimeIndex,
        effects: list[Effect],
        dt: float = 1.0,
        period: pd.Index | None = None,
        component_status_items: list[tuple[str, Status, list[str]]] | None = None,
    ) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            time: Time index.
            effects: Effect definitions for cost coefficients.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for multi-period models. When provided,
                ``effect_pair_coeff``, ``rel_lb``, ``rel_ub`` and ``fixed_profile``
                gain a ``period`` dimension so that ``effects_per_flow_hour``,
                ``relative_rate_min``, ``relative_rate_max`` and
                ``fixed_relative_profile`` can vary across periods.
            component_status_items: Component-level status entries as
                ``(component_id, Status, [governed flow ids])``. Each entry
                produces an on/startup/shutdown binary keyed by the
                component, gating all listed flows together.
        """
        from fluxopt.elements import Investment, Sizing

        flow_ids = [bf.id for bf in flows]
        effect_ids = [e.id for e in effects]
        n_time = len(time)

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
        pair_flows: list[str] = []
        pair_effects: list[str] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        status_items: list[tuple[str, Status]] = []
        prior_rates_map: dict[str, list[float]] = {}

        envelope_coords: dict[str, Any] = {'time': time}
        if period is not None:
            envelope_coords['period'] = period
        nan_envelope = xr.DataArray(
            np.full([len(v) for v in envelope_coords.values()], np.nan),
            dims=list(envelope_coords),
            coords=envelope_coords,
        )

        for i, (fid, f, _sign) in enumerate(flows):
            rel_lbs.append(as_dataarray(f.relative_rate_min, envelope_coords))
            rel_ubs.append(as_dataarray(f.relative_rate_max, envelope_coords))

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
                as_dataarray(f.ramp_up_per_hour, envelope_coords) if f.ramp_up_per_hour is not None else nan_envelope
            )
            ramp_downs.append(
                as_dataarray(f.ramp_down_per_hour, envelope_coords)
                if f.ramp_down_per_hour is not None
                else nan_envelope
            )

            if f.fixed_relative_profile is not None:
                profiles.append(as_dataarray(f.fixed_relative_profile, envelope_coords))
                bound_type.append(BoundType.PROFILE)
            elif f.size is None:
                profiles.append(nan_envelope)
                bound_type.append(BoundType.UNSIZED)
            else:
                profiles.append(nan_envelope)
                bound_type.append(BoundType.BOUNDED)

            # Effect coefficients for this flow — one row per effect charged
            as_da_coords: dict[str, Any] = {'time': time}
            if period is not None:
                as_da_coords['period'] = period
            for effect_label, factor in f.effects_per_flow_hour.items():
                pair_flows.append(fid)
                pair_effects.append(effect_label)
                effect_coeffs.append(as_dataarray(factor, as_da_coords))

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
            effect_pair_flow=xr.DataArray(pair_flows, dims=['effect_pair']),
            effect_pair_effect=xr.DataArray(pair_effects, dims=['effect_pair']),
            effect_pair_coeff=(
                fast_concat(effect_coeffs, pd.Index(range(len(effect_coeffs)), name='effect_pair'))
                if effect_coeffs
                else xr.DataArray(np.zeros((0, n_time)), dims=['effect_pair', 'time'], coords={'time': time})
            ),
            flow_id=xr.DataArray(flow_ids, dims=['flow'], coords={'flow': flow_ids}),
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
                time,
                dim=Dim.STATUS_FLOW,
                prior_rates_map=prior_rates_map,
                dt=dt,
                period=period,
            ),
            cstatus=StatusData.build(
                [(cid, s) for cid, s, _ in (component_status_items or [])],
                effect_ids,
                time,
                dim=Dim.CSTATUS_COMPONENT,
                period=period,
            ),
            governed_by=_governed_by(flow_ids, component_status_items or []),
        )


def _governed_by(flow_ids: list[str], items: list[tuple[str, Any, list[str]]]) -> xr.DataArray | None:
    """Which component's Status governs each flow, '' where none governs it.

    The inverse of the map the caller supplies, and the direction every reader
    wanted: a flow is governed by at most one component, so this is a column.
    """
    if not items:
        return None
    owner = {fid: cid for cid, _status, governed in items for fid in governed}
    return xr.DataArray([owner.get(fid, '') for fid in flow_ids], dims=['flow'], coords={'flow': flow_ids})


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
    """Which carrier each flow balances on, and what a carrier is called.

    Two frames because they are two things: `membership` is per flow,
    `carriers` is per carrier. A flow is on exactly one carrier, so
    membership is a column rather than the `(carrier, flow)` matrix it used
    to be.
    """

    #: (flow, carrier, sign) — +1 produces into it, -1 consumes from it
    membership: pl.DataFrame
    #: (carrier, unit, color, description) — one row per declared carrier
    carriers: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The declared carriers, in declaration order."""
        return self.carriers['carrier'].to_list()

    def __post_init__(self) -> None:
        """Check the signs, and that every carrier named is declared.

        Layer 3 — see docs/design/validation-layers.md.
        """
        if len(bad := self.membership.filter(~pl.col('sign').is_in(pl.Series([1.0, -1.0]).implode()))):
            raise ValueError(f'CarriersData.sign must be +1 or -1; got {bad["sign"].to_list()}')
        if len(stray := self.membership.filter(~pl.col('carrier').is_in(self.carriers['carrier'].implode()))):
            raise ValueError(
                f'CarriersData.membership names carriers that are not declared: '
                f'{sorted(set(stray["carrier"].to_list()))}'
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

        rows = [
            (node_id(c.id, node) if node else c.id, c.unit, c.color or '', c.description)
            for c in carriers
            for node in c.nodes or [None]
        ]

        return cls(
            membership=pl.DataFrame(
                {
                    'flow': [bf.id for bf in flows],
                    'carrier': [_carrier_dim_id(f) for _fid, f, _sign in flows],
                    'sign': [float(carrier_coeff[bf.id]) for bf in flows],
                },
                schema={'flow': pl.String, 'carrier': pl.String, 'sign': pl.Float64},
            ),
            carriers=pl.DataFrame(
                {
                    'carrier': [r[0] for r in rows],
                    'unit': [r[1] for r in rows],
                    'color': [r[2] for r in rows],
                    'description': [r[3] for r in rows],
                },
                schema={'carrier': pl.String, 'unit': pl.String, 'color': pl.String, 'description': pl.String},
            ),
        )


@dataclass
class ConvertersData:
    """Linear conversion equations, as the coefficients they actually have.

    A converter states some equations; each names some of its flows with a
    coefficient. That is a table, and only the non-zero entries are in it —
    a flow a given equation does not mention has no row rather than a zero.
    """

    #: (converter, flow, eq_idx, time, value) — non-zero coefficients only
    coefficients: pl.DataFrame
    #: (converter, n_equations) — equations are numbered 0..n-1 per converter
    equations: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The converters stating linear equations, in declaration order."""
        return self.equations['converter'].to_list()

    @property
    def width(self) -> int:
        """The most equations any one converter states."""
        return int(self.equations['n_equations'].max() or 0)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        """Check the counts are positive and the rows name converters we carry.

        Layer 3 — see docs/design/validation-layers.md.
        """
        if len(short := self.equations.filter(pl.col('n_equations') < 1)):
            raise ValueError(f'ConvertersData.n_equations must be positive; got {short["n_equations"].to_list()}')
        stray = self.coefficients.filter(~pl.col('converter').is_in(self.equations['converter'].implode()))
        if len(stray):
            raise ValueError(
                f'ConvertersData.coefficients references unknown converter(s) '
                f'{sorted(set(stray["converter"].to_list()))}'
            )

    @classmethod
    def build(cls, converters: list[Converter], time: TimeIndex) -> Self | None:
        """Build ConvertersData from the equations each converter states.

        Only linear converters are included; piecewise converters
        (``conversion is not None``) live in :class:`PiecewiseData`.

        Args:
            converters: Converter definitions.
            time: Time index.
        """
        converters = [c for c in converters if c.conversion is None]
        if not converters:
            return None

        labels = list(time)
        n_time = len(labels)
        conv_col: list[str] = []
        flow_col: list[str] = []
        eq_col: list[int] = []
        time_col: list[Any] = []
        value_col: list[float] = []
        for conv in converters:
            for fid, flow, _sign in conv._qualified_flows():
                short = flow.short_id
                for eq_i, equation in enumerate(conv.conversion_factors):
                    if short not in equation:
                        continue
                    values = np.broadcast_to(as_dataarray(equation[short], {'time': time}).values, (n_time,))
                    conv_col.extend([conv.id] * n_time)
                    flow_col.extend([fid] * n_time)
                    eq_col.extend([eq_i] * n_time)
                    time_col.extend(labels)
                    value_col.extend(float(v) for v in values)

        return cls(
            coefficients=pl.DataFrame(
                {
                    'converter': conv_col,
                    'flow': flow_col,
                    'eq_idx': eq_col,
                    'time': time_col,
                    'value': value_col,
                }
            ),
            equations=pl.DataFrame(
                {
                    'converter': [c.id for c in converters],
                    'n_equations': [len(c.conversion_factors) for c in converters],
                },
                schema={'converter': pl.String, 'n_equations': pl.Int64},
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
        """Re-check what `PiecewiseConversion` already refuses, for a reloaded file.

        A reload guard, not the enforcement: `method` is a `Literal` on the
        element and `bound` is checked when the curve is constructed. See
        docs/design/validation-layers.md.
        """
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
    def build(cls, converters: list[Converter], time: TimeIndex) -> Self | None:
        """Build PiecewiseData from converters with ``PiecewiseConversion``.

        Args:
            converters: Converter definitions; only those with
                ``conversion is not None`` are processed.
            time: Time index for breakpoint and availability arrays.
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
            avail_slices.append(as_dataarray(curve.availability, {'time': time}))
            has_statuses.append(curve.status is not None)

            short_to_qid = {bf.flow.short_id: bf.id for bf in conv._qualified_flows()}
            for short, pts, bound in curve._iter_normalized():
                qid = short_to_qid[short]
                bp_arrays = [as_dataarray(bp, {'time': time}) for bp in pts]
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
    #: One row per declared ``contribution_from`` factor, each carrying its
    #: own ``(time[, period])`` series. The dense ``(effect, source_effect,
    #: time, period)`` matrix this replaces was 67 MB at 2% live on the stress
    #: reference system — and most of that was not sparsity but repetition,
    #: a scalar factor broadcast across every timestep and period.
    cf_pair_effect: xr.DataArray | None = None  # (cf_pair,) — the effect charged
    cf_pair_source: xr.DataArray | None = None  # (cf_pair,) — the effect it comes from
    cf_pair_factor: xr.DataArray | None = None  # (cf_pair, time[, period])
    period_weights: xr.DataArray | None = None  # (effect, period)

    def cf_matrix(self) -> xr.DataArray | None:
        """The cross-effect factors as a square matrix, or None if there are none.

        Built here rather than stored, because inverting it is the only thing
        that wants it square — and a stored one costs a full
        ``(effect, source_effect, time, period)`` grid to hold a handful of
        declared factors. Transient by design: nothing keeps the result.
        """
        if self.cf_pair_effect is None:
            return None
        assert self.cf_pair_source is not None
        assert self.cf_pair_factor is not None
        effects = list(self.total_min.coords['effect'].values)
        factor = self.cf_pair_factor
        rest = {d: factor.coords[d] for d in factor.dims if d != 'cf_pair'}
        matrix = xr.DataArray(
            np.zeros((len(effects), len(effects), *[len(v) for v in rest.values()])),
            dims=['effect', 'source_effect', *rest],
            coords={'effect': effects, 'source_effect': effects, **rest},
        )
        for i, (e, src) in enumerate(zip(self.cf_pair_effect.values, self.cf_pair_source.values, strict=True)):
            matrix.loc[{'effect': str(e), 'source_effect': str(src)}] = factor.isel(cf_pair=i)
        return matrix

    def __post_init__(self) -> None:
        """Reject self-references and cycles in the cross-effect matrix (also on netCDF reload)."""
        if self.cf_pair_effect is None:
            return
        assert self.cf_pair_source is not None
        assert self.cf_pair_factor is not None
        live = (self.cf_pair_factor != 0).any([d for d in self.cf_pair_factor.dims if d != 'cf_pair'])
        edges = [
            (str(e), str(src))
            for e, src, keep in zip(self.cf_pair_effect.values, self.cf_pair_source.values, live.values, strict=True)
            if keep
        ]
        for eid, src in edges:
            if eid == src:
                raise ValueError(f'Effect {eid!r} cannot reference itself in contribution_from')
        # Every effect is a node, edges or not: the cycle walk looks up each
        # neighbour's state, so a node reachable but unlisted is a KeyError.
        adjacency: dict[str, list[str]] = {str(e): [] for e in self.total_min.coords['effect'].values}
        for eid, src in edges:
            adjacency[eid].append(src)
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
        time: TimeIndex,
        period: pd.Index | None = None,
    ) -> Self:
        """Build EffectsData from element objects.

        Args:
            effects: Effect definitions.
            time: Time index.
            period: Period index (multi-period only).
        """
        effect_ids = [e.id for e in effects]
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
        cf_effects: list[str] = []
        cf_sources: list[str] = []
        cf_factors: list[xr.DataArray] = []
        if has_contributions:
            tmpl_t = _effect_template({'effect': effect_ids, 'time': time}, period)
            for e in effects:
                for src_id, factor in e.contribution_from.items():
                    cf_effects.append(e.id)
                    cf_sources.append(src_id)
                    cf_factors.append(as_dataarray(factor, tmpl_t.as_da_coords))

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
            cf_pair_effect=xr.DataArray(cf_effects, dims=['cf_pair']) if cf_factors else None,
            cf_pair_source=xr.DataArray(cf_sources, dims=['cf_pair']) if cf_factors else None,
            cf_pair_factor=(
                fast_concat(cf_factors, pd.Index(range(len(cf_factors)), name='cf_pair')) if cf_factors else None
            ),
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
        """Re-check the ranges `Storage` already refuses, on the resolved values.

        Two things reach here that the element could not see: a reloaded
        netCDF, which never passed through `Storage` at all, and a
        `ProfileRef`, whose numbers arrive when profiles are resolved and so
        are not there to check when the storage is written. Everything else
        was refused at construction — see docs/design/validation-layers.md.
        """
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
        time: TimeIndex,
        dt: xr.DataArray,
        effects: list[Effect] | None = None,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Build StoragesData from element objects.

        Args:
            storages: Storage definitions.
            time: Time index.
            dt: Timestep durations.
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

            eta_cs.append(as_dataarray(s.eta_charge, {'time': time}))
            eta_ds.append(as_dataarray(s.eta_discharge, {'time': time}))
            losses.append(as_dataarray(s.relative_loss_per_hour, {'time': time}))

            level_lbs.append(as_dataarray(s.relative_level_min, {'time': time}))
            level_ubs.append(as_dataarray(s.relative_level_max, {'time': time}))

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


@dataclass
class Dims:
    """Shared model coordinates and temporal metadata.

    Owns the time and period dimensions, timestep durations, and weights.
    """

    time: xr.DataArray  # (time,) — coordinate labels
    dt: xr.DataArray  # (time,) — timestep durations [h]
    weights: xr.DataArray  # (time,) — timestep weights
    period: xr.DataArray | None = None  # (period,) — coordinate labels
    period_weights: xr.DataArray | None = None  # (period,) — duration weights

    def __post_init__(self) -> None:
        for name, arr in [('dt', self.dt), ('weights', self.weights)]:
            if arr.dims != ('time',):
                raise ValueError(f"Dims.{name} must be 1D with dims=('time',), got {arr.dims}")
            if not arr.coords['time'].equals(self.time):
                raise ValueError(f'Dims.{name} time coordinate does not match Dims.time')

    def coords(self, *, time: bool = False, period: bool = False) -> dict[str, xr.DataArray]:
        """Return shared coordinates for variable/DataArray creation.

        Also the single point of truth for the model's variate dims used by
        :func:`fluxopt.types.as_dataarray`: pick the reach a field supports
        (e.g. ``coords(time=True, period=True)`` for operational profiles,
        ``coords(period=True)`` for investment-time fields). When a new
        variate dim (e.g. ``scenario``) is added, extend this method once
        and every call site picks it up.

        Args:
            time: Include the time coordinate.
            period: Include the period coordinate (no-op in single-period mode).
        """
        result: dict[str, xr.DataArray] = {}
        if time:
            result['time'] = self.time
        if period and self.period is not None:
            result['period'] = self.period
        return result

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        data_vars: dict[str, xr.DataArray] = {'dt': self.dt, 'weights': self.weights}
        if self.period is not None:
            data_vars['period'] = self.period
        if self.period_weights is not None:
            data_vars['period_weights'] = self.period_weights
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
        )

    @classmethod
    def build(
        cls,
        time: TimeIndex,
        dt: xr.DataArray,
        periods: list[int] | pd.Index | None = None,
        period_weights: list[float] | None = None,
    ) -> Self:
        """Build Dims from a time index and optional periods.

        Args:
            time: Normalized time index.
            dt: Timestep durations.
            periods: Integer period labels for multi-period optimization.
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        time_coord = xr.DataArray(time, dims=['time'], coords={'time': time})
        weights = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time}, name='weight')

        period_da: xr.DataArray | None = None
        period_weights_da: xr.DataArray | None = None
        if periods is not None:
            period_idx, period_weights_da = _compute_period_weights(periods, period_weights)
            period_da = xr.DataArray(period_idx.values, dims=['period'], coords={'period': period_idx})

        return cls(
            time=time_coord,
            dt=dt,
            weights=weights,
            period=period_da,
            period_weights=period_weights_da,
        )


_CONTAINER_TYPES: dict[str, type] = {
    'sizing': SizingData,
    'status': StatusData,
    'invest': InvestmentData,
    'cstatus': StatusData,
}


#: Which sub-container each top-level one carries, and of what class. Read by
#: :meth:`ModelData.load` to rebuild them from their own directories.
_SUB_CONTAINERS: dict[str, dict[str, Any]] = {
    'flows': {'sizing': SizingData, 'invest': InvestmentData, 'status': StatusData, 'cstatus': StatusData},
    'storages': {'sizing': SizingData, 'invest': InvestmentData},
}


def _frames_of(obj: Any) -> dict[str, pl.DataFrame]:
    """Every polars frame a container holds, by field name.

    Empty for a container still holding arrays, which is how
    :meth:`ModelData.save` tells the two apart without asking either to
    declare which it is.
    """
    return {f.name: value for f in dataclasses.fields(obj) if isinstance(value := getattr(obj, f.name), pl.DataFrame)}


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
class ModelData:
    flows: FlowsData
    carriers: CarriersData
    converters: ConvertersData | None  # None when no linear converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dims: Dims
    piecewise: PiecewiseData | None = None  # None when no piecewise converters

    def __post_init__(self) -> None:
        """Check ids referenced *between* tables resolve.

        Layer 3, and the clearest case for it: no single table can answer
        this, and a reloaded file never passed the system layer. See
        docs/design/validation-layers.md.

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

        check_flows(self.carriers.membership['flow'].to_list(), 'carriers.membership')
        if self.flows.sizing is not None:
            check_flows(self.flows.sizing.ids, 'flows.sizing')
        if self.flows.invest is not None:
            check_flows(self.flows.invest.ids, 'flows.invest')
        if self.flows.status is not None:
            check_flows(coord_ids(self.flows.status.uptime_min), 'flows.status')
        if self.flows.governed_by is not None:
            components = (
                {str(c) for c in self.flows.cstatus.uptime_min.coords[Dim.CSTATUS_COMPONENT].values}
                if self.flows.cstatus is not None
                else set()
            )
            named = {str(v) for v in self.flows.governed_by.values if str(v)}
            if unknown := sorted(named - components):
                raise ValueError(f'flows.governed_by names components without a Status: {unknown}')
        if self.converters is not None:
            check_flows(self.converters.coefficients['flow'].to_list(), 'converters.coefficients')
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
                if container is None:
                    continue
                ids = container.ids
                if unknown := sorted(set(ids) - storage_ids):
                    raise ValueError(f'{what} references unknown storage id(s) {unknown}')

        effect_ids = set(map(str, self.effects.total_min.coords['effect'].values))
        coeff_effects = {str(e) for e in self.flows.effect_pair_effect.values}
        if not coeff_effects <= effect_ids:
            raise ValueError(
                f'flows.effect_pair_effect names effects {sorted(coeff_effects - effect_ids)} that are not in '
                f'the effects table {sorted(effect_ids)}'
            )

    def save(self, path: str | Path) -> None:
        """Write the model data as a directory of tables.

        One parquet file per frame, one netCDF file per container still
        holding arrays. Parquet because these *are* tables: it carries the
        schema, so a column with no rows still knows it holds strings, and a
        timestamp survives without anyone deciding what unit its integers
        meant.

        Args:
            path: Directory to write into. Created if absent.
        """
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        for name, obj in self._containers().items():
            if obj is None:
                continue
            group = root / name
            group.mkdir(exist_ok=True)
            frames = _frames_of(obj)
            for frame_name, frame in frames.items():
                frame.write_parquet(group / f'{frame_name}.parquet')
            if not frames:
                obj.to_dataset().to_netcdf(group / 'arrays.nc', engine='netcdf4')
            for cname, sub in _table_containers(obj).items():
                sub_group = group / cname
                sub_group.mkdir(exist_ok=True)
                sub_frames = _frames_of(sub)
                for frame_name, frame in sub_frames.items():
                    frame.write_parquet(sub_group / f'{frame_name}.parquet')
                if not sub_frames:
                    sub.to_dataset().to_netcdf(sub_group / 'arrays.nc', engine='netcdf4')

    @classmethod
    def load(cls, path: str | Path) -> ModelData:
        """Read model data written by :meth:`save`.

        Args:
            path: The directory :meth:`save` wrote.

        Raises:
            OSError: If the directory holds no fluxopt model data.
        """
        root = Path(path)
        if not (root / 'dims').is_dir():
            raise OSError(f'No fluxopt model data found in {root} (missing dims/)')

        def read(name: str, klass: Any, subs: dict[str, Any] | None = None) -> Any:
            group = root / name
            if not group.is_dir():
                return None
            frames = {f.stem: pl.read_parquet(f) for f in sorted(group.glob('*.parquet'))}
            if frames:
                return klass(**frames, **(subs or {}))
            ds = xr.load_dataset(group / 'arrays.nc', engine='netcdf4')
            return klass.from_dataset(ds, subs) if subs is not None else klass.from_dataset(ds)

        def read_subs(name: str, klass: Any) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for field, sub_class in _SUB_CONTAINERS.get(name, {}).items():
                sub = read(f'{name}/{field}', sub_class)
                if sub is not None:
                    out[field] = sub
            return out

        return cls(
            flows=read('flows', FlowsData, read_subs('flows', FlowsData)),
            carriers=read('carriers', CarriersData),
            converters=read('converters', ConvertersData),
            effects=read('effects', EffectsData),
            storages=read('storages', StoragesData, read_subs('storages', StoragesData)),
            dims=read('dims', Dims),
            piecewise=read('piecewise', PiecewiseData),
        )

    def _containers(self) -> dict[str, Any]:
        """Every top-level container, by the name its directory takes."""
        return {
            'flows': self.flows,
            'carriers': self.carriers,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
            'piecewise': self.piecewise,
            'dims': self.dims,
        }

    @classmethod
    def build(
        cls,
        timesteps: Timesteps,
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
            timesteps: Time index for the optimization horizon.
            carriers: Carrier declarations.
            effects: Effects to track.
            ports: System boundary ports.
            converters: Linear converters.
            storages: Energy storages.
            dt: Timestep duration in hours. Auto-derived if None.
            periods: Integer period labels for multi-period optimization.
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        from fluxopt.elements import PENALTY_EFFECT_ID, Effect
        from fluxopt.types import compute_dt as _compute_dt

        converters = converters or []
        stor_list = storages or []
        time = normalize_timesteps(timesteps)
        dt_da = _compute_dt(time, dt)

        if not any(e.id == PENALTY_EFFECT_ID for e in effects):
            effects = [*effects, Effect(id=PENALTY_EFFECT_ID)]

        flows, carrier_coeff = _collect_flows(ports, converters, stor_list)
        validate_system(carriers=carriers, effects=effects, ports=ports, converters=converters, storages=stor_list)

        dims = Dims.build(time, dt_da, periods=periods, period_weights=period_weights)

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
            time,
            effects,
            dt=dt_scalar,
            period=period_idx,
            component_status_items=comp_status_items,
        )
        carriers_data = CarriersData.build(carriers, flows, carrier_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time, period=period_idx)
        storages_data = StoragesData.build(stor_list, time, dims.dt, effects, period=period_idx)
        piecewise_data = PiecewiseData.build(converters, time)

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
