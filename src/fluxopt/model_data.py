from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, get_args

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from fluxopt.types import PiecewiseMethod, as_dataarray, normalize_timesteps
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


# Nested container fields on FlowsData / StoragesData — written to a
# sub-directory of their own, not as frames of the parent container.
_CONTAINER_FIELD_NAMES = frozenset({'sizing', 'invest'})


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
        the value the user wrote, naming the field. A hand-edited file never
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
        period: pd.Index | None = None,
    ) -> Self | None:
        """Collect Sizing objects into frames, or None if there are none.

        Args:
            items: Pairs of (element_id, Sizing).
            period: Period index for period-varying effects.
        """
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
                per_size.extend(
                    _per_period(s.effects_per_size.get(ek, 0.0), n_periods, f'{item_id!r} effects_per_size[{ek!r}]')
                )
                fixed.extend(_per_period(s.effects_fixed.get(ek, 0.0), n_periods, f'{item_id!r} effects_fixed[{ek!r}]'))

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


def _over_periods(value: Any, n_periods: int, what: str) -> np.ndarray:
    """A per-period value as an array, refusing a length the period axis cannot hold.

    A scalar broadcasts to every period, and a sequence has to name each one.
    Any other length was written against a different number of periods than
    the system has; truncating it — which is what indexing quietly does —
    would drop values the user gave and solve a different problem.

    The element layer cannot decide this: how many periods exist is a property
    of the system, not of the element carrying the sequence.
    """
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size not in (1, n_periods):
        raise ValueError(
            f'{what} has {arr.size} values but the system has {n_periods} period(s). '
            'Give one value per period, or a single value for all of them.'
        )
    return arr


def _per_period(value: Any, n_periods: int, what: str) -> list[float]:
    """A lump coefficient's value in each period.

    Built as a list rather than a row per period because the frame is
    assembled column-wise — a dict per row costs more than the data.
    """
    arr = _over_periods(value, n_periods, what)
    return [float(arr[0])] * n_periods if arr.size == 1 else [float(v) for v in arr]


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
        period: pd.Index | None = None,
    ) -> Self | None:
        """Collect Investment objects into frames, or None if there are none.

        Args:
            items: Pairs of (element_id, Investment).
            period: Period index for period-varying effects.
        """
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
                    values[column].extend(
                        _per_period(getattr(inv, field).get(ek, 0.0), n_periods, f'{item_id!r} {field}[{ek!r}]')
                    )

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
    """Binary on/off behaviour for one entity family — a flow's or a component's.

    Durations are optional on the element, so an entity that constrains none
    has no row in `durations` rather than a row of NaN. The same for what
    happened before the horizon: `prior` holds only the entities that ran.
    """

    #: (entity,) — every entity carrying a Status, in declaration order
    entities: pl.DataFrame
    #: (entity, uptime_min, uptime_max, downtime_min, downtime_max) — only
    #: entities constraining at least one. A null in a column is that one
    #: bound being unset, which is different from having none at all.
    durations: pl.DataFrame
    #: (entity, initial, previous_uptime, previous_downtime) — only entities
    #: whose state before the horizon is known
    prior: pl.DataFrame
    #: (entity, effect, time, period, running, startup) — declared pairs only
    effects: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The entities carrying a Status, in declaration order."""
        return self.entities['entity'].to_list()

    def __post_init__(self) -> None:
        """Re-check the durations `Status` already refuses, for a reloaded file.

        A reload guard — see docs/design/validation-layers.md.
        """
        columns = ('uptime_min', 'uptime_max', 'downtime_min', 'downtime_max')
        if len(bad := self.durations.filter(pl.any_horizontal(pl.col(c) < 0 for c in columns))):
            raise ValueError(f'Status durations are negative on {bad["entity"].to_list()}')
        unordered = self.durations.filter(
            (pl.col('uptime_max') < pl.col('uptime_min')) | (pl.col('downtime_max') < pl.col('downtime_min'))
        )
        if len(unordered):
            raise ValueError(f'Status max is below min on {unordered["entity"].to_list()}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Status]],
        time: TimeIndex,
        prior_rates_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Collect Status objects into frames, or None if there are none.

        Args:
            items: Pairs of (id, Status).
            time: Time index for effect series.
            prior_rates_map: Item id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for period-varying effects.
        """
        if not items:
            return None

        prior_rates_map = prior_rates_map or {}
        periods = list(range(len(period))) if period is not None else [0]
        labels = np.asarray(time)
        n_time = len(labels)

        constrained = [
            (i, z)
            for i, z in items
            if any(v is not None for v in (z.uptime_min, z.uptime_max, z.downtime_min, z.downtime_max))
        ]
        ran_before = [i for i, _ in items if prior_rates_map.get(i) is not None]

        def duration(prior: list[float], state: int) -> float:
            return compute_previous_duration(xr.DataArray(prior, dims=['_prior_t']), target_state=state, dt=dt)

        coords: dict[str, Any] = {'time': time}
        if period is not None:
            coords['period'] = period
        cols: dict[str, list[np.ndarray]] = {
            k: [] for k in ('entity', 'effect', 'time', 'period', 'running', 'startup')
        }
        for item_id, z in items:
            for ek in sorted(set(z.effects_per_running_hour) | set(z.effects_per_startup)):
                for p_index in periods:
                    cols['entity'].append(np.full(n_time, item_id))
                    cols['effect'].append(np.full(n_time, ek))
                    cols['time'].append(labels)
                    cols['period'].append(np.full(n_time, p_index))
                    cols['running'].append(_series(z.effects_per_running_hour.get(ek, 0.0), coords, p_index, n_time))
                    cols['startup'].append(_series(z.effects_per_startup.get(ek, 0.0), coords, p_index, n_time))

        def joined(key: str, dtype: Any) -> np.ndarray:
            parts = cols[key]
            return np.concatenate(parts) if parts else np.array([], dtype=dtype)

        return cls(
            entities=pl.DataFrame({'entity': [i for i, _ in items]}, schema={'entity': pl.String}),
            durations=pl.DataFrame(
                {
                    'entity': [i for i, _ in constrained],
                    'uptime_min': [z.uptime_min for _, z in constrained],
                    'uptime_max': [z.uptime_max for _, z in constrained],
                    'downtime_min': [z.downtime_min for _, z in constrained],
                    'downtime_max': [z.downtime_max for _, z in constrained],
                },
                schema={
                    'entity': pl.String,
                    'uptime_min': pl.Float64,
                    'uptime_max': pl.Float64,
                    'downtime_min': pl.Float64,
                    'downtime_max': pl.Float64,
                },
            ),
            prior=pl.DataFrame(
                {
                    'entity': ran_before,
                    'initial': [1.0 if prior_rates_map[i][-1] > 0 else 0.0 for i in ran_before],
                    'previous_uptime': [duration(prior_rates_map[i], 1) for i in ran_before],
                    'previous_downtime': [duration(prior_rates_map[i], 0) for i in ran_before],
                },
                schema={
                    'entity': pl.String,
                    'initial': pl.Float64,
                    'previous_uptime': pl.Float64,
                    'previous_downtime': pl.Float64,
                },
            ),
            effects=pl.DataFrame(
                {
                    'entity': joined('entity', str),
                    'effect': joined('effect', str),
                    'time': pd.to_datetime(joined('time', 'datetime64[ns]')).to_pydatetime().tolist(),
                    'period': joined('period', int),
                    'running': joined('running', float),
                    'startup': joined('startup', float),
                },
                schema={
                    'entity': pl.String,
                    'effect': pl.String,
                    'time': pl.Datetime('us'),
                    'period': pl.Int64,
                    'running': pl.Float64,
                    'startup': pl.Float64,
                },
            ),
        )


def _series(value: Any, coords: dict[str, Any], p_index: int, n_time: int) -> np.ndarray:
    """One period's slice of a possibly period-varying time series.

    Coerced against every axis the model has, not just time: a value may be
    declared per period, and `as_dataarray` refuses a dim its target does not
    name rather than guessing which axis was meant.
    """
    da = as_dataarray(value, coords)
    # By name, not by position: a value declared only per period comes back
    # one-dimensional over `period`, which a shape test reads as a time series.
    if 'period' in da.dims:
        da = da.isel(period=p_index if da.sizes['period'] > p_index else 0)
    return np.broadcast_to(np.asarray(da.values, dtype=float), (n_time,))


@dataclass
class FlowsData:
    """Flow parameters, as the tables a flow declares.

    Absence is a missing row. A flow whose size is being optimized has no row
    in `sizes`; one that limits no ramp has none in `ramps`; a flow charges
    only the effects it has rows for in `effect_pairs`. `envelope` is the
    exception and is dense on purpose: the relative bounds sit at the rate
    variable's own grid, where a missing row reads as a binding zero.
    """

    #: (flow,) — every flow, in declaration order. The roster: the other
    #: tables name only the flows that declare the thing they hold.
    flows: pl.DataFrame
    #: (flow, size) — only flows sized to a fixed value
    sizes: pl.DataFrame
    #: (flow, time, period, relative_rate_min, relative_rate_max) — dense
    envelope: pl.DataFrame
    #: (flow, time, period, value) — only flows whose rate follows a profile
    fixed_profile: pl.DataFrame
    #: (flow, effect, time, period, value) — one row per (flow, effect) a flow
    #: actually charges. Dense over that product it is not: on the stress
    #: reference system the product is 443 MB at 4% live.
    effect_pairs: pl.DataFrame
    #: (flow, time, period, ramp_up, ramp_down) [1/h] — only flows limiting a
    #: ramp, and only the direction they limit
    ramps: pl.DataFrame
    #: (flow, flow_hours_min, flow_hours_max, load_factor_min, load_factor_max)
    #: — only flows bounding an aggregate, per period
    aggregates: pl.DataFrame
    #: (flow, component) — only flows a component's Status governs. A flow is
    #: governed by at most one component, so this is a column rather than the
    #: ragged padded matrix it used to be on the component's Status.
    governed_by: pl.DataFrame
    sizing: SizingData | None = None
    invest: InvestmentData | None = None

    @property
    def ids(self) -> list[str]:
        """Every flow, in declaration order."""
        return self.flows['flow'].to_list()

    @property
    def sized_ids(self) -> set[str]:
        """Flows whose size is known here — fixed, not optimized."""
        return set(self.sizes['flow'].to_list())

    @property
    def has_size(self) -> set[str]:
        """Flows with a size at all — fixed here, or a variable to be solved for."""
        sized = self.sized_ids
        for container in (self.sizing, self.invest):
            if container is not None:
                sized |= set(container.ids)
        return sized

    @property
    def profiled_ids(self) -> list[str]:
        """Flows whose rate is pinned to a profile, in declaration order."""
        profiled = set(self.fixed_profile['flow'].to_list())
        return [f for f in self.ids if f in profiled]

    @property
    def bounded_ids(self) -> list[str]:
        """Flows whose rate envelope is ``[rel_lb, rel_ub] * size``.

        Which is: it has a size, and no profile pinning it. A flow that is
        neither bounded nor profiled is unsized, and the program asks about
        that by asking about neither — so the third case needs no name.
        """
        sized, profiled = self.has_size, set(self.fixed_profile['flow'].to_list())
        return [f for f in self.ids if f in sized and f not in profiled]

    def __post_init__(self) -> None:
        """Validate relative bounds, status non-degeneracy, and sized-feature requirements."""
        negative = self.envelope.filter(pl.col('relative_rate_min') < -1e-12)['flow']
        if bad := negative.unique(maintain_order=True).to_list():
            raise ValueError(f'Negative lower bounds on flows: {bad}')
        crossed = self.envelope.filter(pl.col('relative_rate_min') > pl.col('relative_rate_max') + 1e-12)['flow']
        if bad := crossed.unique(maintain_order=True).to_list():
            raise ValueError(f'Lower bound > upper bound on flows: {bad}')
        self._check_sized_features()

    def _check_sized_features(self) -> None:
        """Ramp limits and load-factor bounds need a sized flow (fixed, Sizing, or Investment).

        Without a size these features would feed a null into constraint
        coefficients; the element layer already rejects this at authoring
        time, this is the guard for direct data edits and reloads.
        """
        sized = self.has_size
        for frame, columns in (
            (self.ramps, ('ramp_up', 'ramp_down')),
            (self.aggregates, ('load_factor_min', 'load_factor_max')),
        ):
            for column in columns:
                declared = frame.filter(pl.col(column).is_not_null())['flow'].unique(maintain_order=True).to_list()
                if bad := [f for f in declared if f not in sized]:
                    raise ValueError(f'{column} requires a sized flow (fixed, Sizing, or Investment) on {bad}')

    @classmethod
    def build(
        cls,
        flows: list[_BoundFlow],
        time: TimeIndex,
        period: pd.Index | None = None,
        component_status_items: list[tuple[str, Status, list[str]]] | None = None,
    ) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            time: Time index.
            period: Period index for multi-period models. `envelope`,
                `fixed_profile`, `effect_pairs` and `ramps` carry a period
                column either way, so the relative bounds, the profile, the
                per-flow-hour coefficients and the ramp limits can all vary
                across periods.
            component_status_items: Component-level status entries as
                ``(component_id, Status, [governed flow ids])``. Only
                ``governed_by`` is read off them here; the Status objects
                themselves become rows of `ModelData.status`.
        """
        from fluxopt.elements import Investment, Sizing

        flow_ids = [bf.id for bf in flows]
        n_time = len(np.asarray(time))
        n_period = len(period) if period is not None else 1
        envelope_coords: dict[str, Any] = {'time': time}
        if period is not None:
            envelope_coords['period'] = period

        #: The (time, period) grid every per-timestep table is laid out on,
        #: time-major so one `np.tile`/`np.repeat` pair keys every one of them.
        #: Both axes are positions, not labels: the program indexes them that
        #: way and `Dims` holds the labels. A timestamp is a poor join key —
        #: integer timesteps and datetime ones would need different dtypes,
        #: and rounding a datetime to a coarser unit silently collides.
        grid_time = np.repeat(np.arange(n_time), n_period)
        grid_period = np.tile(np.arange(n_period), n_time)

        def spread(value: Any) -> np.ndarray:
            """A flow's per-timestep value, flattened onto the (time, period) grid."""
            return np.broadcast_to(
                np.asarray(as_dataarray(value, envelope_coords).values, dtype=float).reshape(n_time, -1),
                (n_time, n_period),
            ).ravel()

        fixed_sizes: list[tuple[str, float]] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        aggregate_rows: list[tuple[str, Any, Any, Any, Any]] = []
        # Each table is built as value columns plus the keys of the blocks that
        # went into it — every block is one whole (time, period) grid, so the
        # key columns are one `repeat`/`tile` at the end rather than an array
        # built per flow.
        env_cols: dict[str, list[np.ndarray]] = {k: [] for k in ('relative_rate_min', 'relative_rate_max')}
        env_keys: list[str] = []
        profile_cols: dict[str, list[np.ndarray]] = {'value': []}
        profile_keys: list[str] = []
        pair_cols: dict[str, list[np.ndarray]] = {'value': []}
        pair_keys: list[tuple[str, str]] = []
        ramp_cols: dict[str, list[np.ndarray]] = {k: [] for k in ('ramp_up', 'ramp_down')}
        ramp_keys: list[str] = []

        for fid, f, _sign in flows:
            env_keys.append(fid)
            env_cols['relative_rate_min'].append(spread(f.relative_rate_min))
            env_cols['relative_rate_max'].append(spread(f.relative_rate_max))

            if isinstance(f.size, Sizing):
                sizing_items.append((fid, f.size))
            elif isinstance(f.size, Investment):
                invest_items.append((fid, f.size))
            elif f.size is not None:
                fixed_sizes.append((fid, float(f.size)))

            if any(b is not None for b in (f.flow_hours_min, f.flow_hours_max, f.load_factor_min, f.load_factor_max)):
                aggregate_rows.append((fid, f.flow_hours_min, f.flow_hours_max, f.load_factor_min, f.load_factor_max))

            if f.ramp_up_per_hour is not None or f.ramp_down_per_hour is not None:
                ramp_keys.append(fid)
                for column, value in (('ramp_up', f.ramp_up_per_hour), ('ramp_down', f.ramp_down_per_hour)):
                    ramp_cols[column].append(spread(value) if value is not None else np.full(n_time * n_period, np.nan))

            if f.fixed_relative_profile is not None:
                profile_keys.append(fid)
                profile_cols['value'].append(spread(f.fixed_relative_profile))

            for effect_label, factor in f.effects_per_flow_hour.items():
                pair_keys.append((fid, effect_label))
                pair_cols['value'].append(spread(factor))

        def frame(
            keys: dict[str, list[Any]], cols: dict[str, list[np.ndarray]], schema: dict[str, Any]
        ) -> pl.DataFrame:
            """One typed table from its block keys and its value columns.

            Every block spans the whole (time, period) grid, so the key columns
            repeat once per block and the grid tiles under them.
            """
            blocks = len(next(iter(cols.values())))
            data: dict[str, Any] = {name: np.repeat(values, n_time * n_period) for name, values in keys.items()}
            data['time'] = np.tile(grid_time, blocks)
            data['period'] = np.tile(grid_period, blocks)
            for key, parts in cols.items():
                data[key] = np.concatenate(parts) if parts else np.array([], dtype=float)
            nan_free = [pl.col(c).fill_nan(None) for c, t in schema.items() if t == pl.Float64]
            return pl.DataFrame(data, schema=schema).with_columns(nan_free)

        grid_schema = {'flow': pl.String, 'time': pl.Int64, 'period': pl.Int64}
        owner = {fid: cid for cid, _status, governed in (component_status_items or []) for fid in governed}
        return cls(
            flows=pl.DataFrame(
                {'flow': flow_ids},
                schema={'flow': pl.String},
            ),
            sizes=pl.DataFrame(
                {'flow': [i for i, _ in fixed_sizes], 'size': [s for _, s in fixed_sizes]},
                schema={'flow': pl.String, 'size': pl.Float64},
            ),
            envelope=frame(
                {'flow': env_keys},
                env_cols,
                {**grid_schema, 'relative_rate_min': pl.Float64, 'relative_rate_max': pl.Float64},
            ),
            fixed_profile=frame({'flow': profile_keys}, profile_cols, {**grid_schema, 'value': pl.Float64}),
            effect_pairs=frame(
                {'flow': [f for f, _ in pair_keys], 'effect': [e for _, e in pair_keys]},
                pair_cols,
                {'flow': pl.String, 'effect': pl.String, 'time': pl.Int64, 'period': pl.Int64, 'value': pl.Float64},
            ),
            ramps=frame(
                {'flow': ramp_keys}, ramp_cols, {**grid_schema, 'ramp_up': pl.Float64, 'ramp_down': pl.Float64}
            ),
            aggregates=pl.DataFrame(
                {
                    'flow': [r[0] for r in aggregate_rows],
                    'flow_hours_min': [r[1] for r in aggregate_rows],
                    'flow_hours_max': [r[2] for r in aggregate_rows],
                    'load_factor_min': [r[3] for r in aggregate_rows],
                    'load_factor_max': [r[4] for r in aggregate_rows],
                },
                schema={
                    'flow': pl.String,
                    'flow_hours_min': pl.Float64,
                    'flow_hours_max': pl.Float64,
                    'load_factor_min': pl.Float64,
                    'load_factor_max': pl.Float64,
                },
            ),
            governed_by=pl.DataFrame(
                {'flow': list(owner), 'component': list(owner.values())},
                schema={'flow': pl.String, 'component': pl.String},
            ),
            sizing=SizingData.build(sizing_items, period=period),
            invest=InvestmentData.build(invest_items, period=period),
        )


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
    """Piecewise-linear conversion curves, as the breakpoints they name.

    Two frames: `curves` is per converter — the method, whether a Status
    gates it, and the availability series — and `links` is per (flow,
    breakpoint), which is what a curve *is*.
    """

    #: (converter, method, has_status, time, availability) — one row per
    #: converter per timestep
    curves: pl.DataFrame
    #: (converter, flow, bound, bp, time, value) — the breakpoint each link
    #: passes through
    links: pl.DataFrame

    def converter_ids(self) -> list[str]:
        """Piecewise converter ids, in declaration order."""
        return self.curves['converter'].unique(maintain_order=True).to_list()

    def __post_init__(self) -> None:
        """Re-check what `PiecewiseConversion` already refuses, for a reloaded file.

        A reload guard, not the enforcement: `method` is a `Literal` on the
        element and `bound` is checked when the curve is constructed. See
        docs/design/validation-layers.md.
        """
        valid = set(get_args(PiecewiseMethod.__value__))
        if bad := sorted(set(self.curves['method'].to_list()) - valid):
            raise ValueError(f'PiecewiseData.method must be one of {sorted(valid)}; got {bad}')
        if bad := sorted(set(self.links['bound'].to_list()) - {'==', '<=', '>='}):
            raise ValueError(f"PiecewiseData.bound must be '==', '<=', or '>='; got {bad}")

    @classmethod
    def build(cls, converters: list[Converter], time: TimeIndex) -> Self | None:
        """Build PiecewiseData from converters with ``PiecewiseConversion``.

        Args:
            converters: Converter definitions; only those with
                ``conversion is not None`` are processed.
            time: Time index for breakpoint and availability series.
        """
        converters = [c for c in converters if c.conversion is not None]
        if not converters:
            return None

        labels = np.asarray(time)
        n_time = len(labels)
        curve_cols: dict[str, list[np.ndarray]] = {
            k: [] for k in ('converter', 'method', 'has_status', 'time', 'availability')
        }
        link_cols: dict[str, list[np.ndarray]] = {k: [] for k in ('converter', 'flow', 'bound', 'bp', 'time', 'value')}

        for conv in converters:
            assert conv.conversion is not None
            curve = conv.conversion
            curve_cols['converter'].append(np.full(n_time, conv.id))
            curve_cols['method'].append(np.full(n_time, curve.method))
            curve_cols['has_status'].append(np.full(n_time, curve.status is not None))
            curve_cols['time'].append(labels)
            curve_cols['availability'].append(
                np.broadcast_to(as_dataarray(curve.availability, {'time': time}).values, (n_time,))
            )

            short_to_qid = {bf.flow.short_id: bf.id for bf in conv._qualified_flows()}
            for short, points, bound in curve._iter_normalized():
                for index, point in enumerate(points):
                    link_cols['converter'].append(np.full(n_time, conv.id))
                    link_cols['flow'].append(np.full(n_time, short_to_qid[short]))
                    link_cols['bound'].append(np.full(n_time, bound))
                    link_cols['bp'].append(np.full(n_time, index))
                    link_cols['time'].append(labels)
                    link_cols['value'].append(np.broadcast_to(as_dataarray(point, {'time': time}).values, (n_time,)))

        def frame(cols: dict[str, list[np.ndarray]], schema: dict[str, Any]) -> pl.DataFrame:
            data: dict[str, Any] = {}
            for key, parts in cols.items():
                joined = np.concatenate(parts) if parts else np.array([])
                data[key] = pd.to_datetime(joined).to_pydatetime().tolist() if key == 'time' else joined
            return pl.DataFrame(data, schema=schema)

        data = cls(
            curves=frame(
                curve_cols,
                {
                    'converter': pl.String,
                    'method': pl.String,
                    'has_status': pl.Boolean,
                    'time': pl.Datetime('us'),
                    'availability': pl.Float64,
                },
            ),
            links=frame(
                link_cols,
                {
                    'converter': pl.String,
                    'flow': pl.String,
                    'bound': pl.String,
                    'bp': pl.Int64,
                    'time': pl.Datetime('us'),
                    'value': pl.Float64,
                },
            ),
        )
        data._warn_redundant_status()
        return data

    def _warn_redundant_status(self) -> None:
        """Warn where a gated curve can sit at zero with the binary on.

        A ``(0, ..., 0)`` breakpoint lets the optimizer produce nothing while
        `status=on`, which decouples the binary from the operating state.
        """
        atol = 1e-9
        gated = set(self.curves.filter(pl.col('has_status'))['converter'].to_list())
        for conv_id in gated:
            rows = self.links.filter(pl.col('converter') == conv_id)
            all_zero = (
                rows.group_by(['bp', 'time'])
                .agg(pl.col('value').abs().max().alias('largest'))
                .filter(pl.col('largest') <= atol)
            )
            if len(all_zero):
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
    """What each effect is bounded by, and how effects charge one another.

    Bounds are optional, so an effect that names none has no row rather than
    a row of NaN. Cross-effect factors are the ones declared: the square
    matrix they imply is 67 MB at 2% live on the stress reference system, and
    most of that is not sparsity but a scalar repeated across every timestep.
    """

    #: (effect,) — every declared effect, in declaration order
    effects: pl.DataFrame
    #: (effect, total_min, total_max) — only effects bounding a weighted total
    totals: pl.DataFrame
    #: (effect, period, periodic_min, periodic_max) — only effects bounding a
    #: period, and only the periods they bound
    periodic: pl.DataFrame
    #: (effect, source_effect, time, period, factor) — declared factors only
    contributions: pl.DataFrame
    #: (effect, period, weight) — only effects overriding the global weights
    period_weights: pl.DataFrame

    @property
    def ids(self) -> list[str]:
        """The declared effects, in declaration order."""
        return self.effects['effect'].to_list()

    def cf_matrix(self, periods: list[Any] | None = None) -> xr.DataArray | None:
        """The cross-effect factors as a square matrix, or None if there are none.

        Built here rather than stored, because inverting it is the only thing
        that wants it square, and a stored one costs a full
        ``(effect, source_effect, time, period)`` grid to hold a handful of
        declared factors. Transient by design: nothing keeps the result.

        Args:
            periods: Labels for the period axis. The frame indexes periods by
                position, which is what the binder wants and what every table
                it is contracted against carries. A caller comparing against a
                solution wants the labels the element layer used instead, and
                xarray aligns on coordinate values rather than reporting a
                mismatch — so passing the wrong one matches nothing, silently.
        """
        if self.contributions.is_empty():
            return None
        ids = self.ids
        rest: dict[str, Any] = {}
        for axis in ('time', 'period'):
            values = self.contributions[axis].unique(maintain_order=True).sort().to_list()
            if len(values) > 1 or axis == 'time':
                rest[axis] = values
        labelled = dict(rest)
        if periods is not None and 'period' in labelled:
            labelled['period'] = [periods[i] for i in rest['period']]
        matrix = xr.DataArray(
            np.zeros((len(ids), len(ids), *[len(v) for v in rest.values()])),
            dims=['effect', 'source_effect', *labelled],
            coords={'effect': ids, 'source_effect': ids, **labelled},
        )
        index = {axis: {v: i for i, v in enumerate(values)} for axis, values in rest.items()}
        for row in self.contributions.iter_rows(named=True):
            key: Any = (ids.index(row['effect']), ids.index(row['source_effect']))
            key += tuple(index[axis][row[axis]] for axis in rest)
            matrix.values[key] = row['factor']
        return matrix

    def __post_init__(self) -> None:
        """Reject self-references and cycles in the cross-effect graph.

        Layer 3 — see docs/design/validation-layers.md.
        """
        live = self.contributions.filter(pl.col('factor') != 0)
        edges = live.select(['effect', 'source_effect']).unique(maintain_order=True)
        for effect, source in zip(edges['effect'], edges['source_effect'], strict=True):
            if effect == source:
                raise ValueError(f'Effect {effect!r} cannot reference itself in contribution_from')
        # Every effect is a node, edges or not: the cycle walk looks up each
        # neighbour's state, so a node reachable but unlisted is a KeyError.
        adjacency: dict[str, list[str]] = {e: [] for e in self.ids}
        for effect, source in zip(edges['effect'], edges['source_effect'], strict=True):
            adjacency[effect].append(source)
        if (cycle := _detect_contribution_cycle(adjacency)) is not None:
            raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')

    @classmethod
    def build(cls, effects: list[Effect], time: TimeIndex, period: pd.Index | None = None) -> Self:
        """Build EffectsData from element objects.

        Args:
            effects: Effect definitions.
            time: Time index.
            period: Period index (multi-period only).
        """
        periods = list(range(len(period))) if period is not None else [0]
        labels = np.asarray(time)
        n_time = len(labels)
        coords: dict[str, Any] = {'time': time}
        if period is not None:
            coords['period'] = period

        bounded = [e for e in effects if e.total_min is not None or e.total_max is not None]
        periodic_rows = [
            (
                e.id,
                p,
                _at_period(e.periodic_min, p, len(periods), f'{e.id!r} periodic_min'),
                _at_period(e.periodic_max, p, len(periods), f'{e.id!r} periodic_max'),
            )
            for e in effects
            if e.periodic_min is not None or e.periodic_max is not None
            for p in periods
        ]
        weight_rows = [
            (e.id, p, _at_period(e.period_weights, p, len(periods), f'{e.id!r} period_weights'))
            for e in effects
            if e.period_weights is not None
            for p in periods
        ]

        cols: dict[str, list[np.ndarray]] = {k: [] for k in ('effect', 'source_effect', 'time', 'period', 'factor')}
        for e in effects:
            for source, factor in e.contribution_from.items():
                for p_index in periods:
                    cols['effect'].append(np.full(n_time, e.id))
                    cols['source_effect'].append(np.full(n_time, source))
                    cols['time'].append(labels)
                    cols['period'].append(np.full(n_time, p_index))
                    cols['factor'].append(_series(factor, coords, p_index, n_time))

        def joined(key: str, dtype: Any) -> np.ndarray:
            parts = cols[key]
            return np.concatenate(parts) if parts else np.array([], dtype=dtype)

        return cls(
            effects=pl.DataFrame({'effect': [e.id for e in effects]}, schema={'effect': pl.String}),
            totals=pl.DataFrame(
                {
                    'effect': [e.id for e in bounded],
                    'total_min': [e.total_min for e in bounded],
                    'total_max': [e.total_max for e in bounded],
                },
                schema={'effect': pl.String, 'total_min': pl.Float64, 'total_max': pl.Float64},
            ),
            periodic=pl.DataFrame(
                {
                    'effect': [r[0] for r in periodic_rows],
                    'period': [r[1] for r in periodic_rows],
                    'periodic_min': [r[2] for r in periodic_rows],
                    'periodic_max': [r[3] for r in periodic_rows],
                },
                schema={
                    'effect': pl.String,
                    'period': pl.Int64,
                    'periodic_min': pl.Float64,
                    'periodic_max': pl.Float64,
                },
            ),
            contributions=pl.DataFrame(
                {
                    'effect': joined('effect', str),
                    'source_effect': joined('source_effect', str),
                    'time': pd.to_datetime(joined('time', 'datetime64[ns]')).to_pydatetime().tolist(),
                    'period': joined('period', int),
                    'factor': joined('factor', float),
                },
                schema={
                    'effect': pl.String,
                    'source_effect': pl.String,
                    'time': pl.Datetime('us'),
                    'period': pl.Int64,
                    'factor': pl.Float64,
                },
            ),
            period_weights=pl.DataFrame(
                {
                    'effect': [r[0] for r in weight_rows],
                    'period': [r[1] for r in weight_rows],
                    'weight': [r[2] for r in weight_rows],
                },
                schema={'effect': pl.String, 'period': pl.Int64, 'weight': pl.Float64},
            ),
        )


def _at_period(value: Any, p_index: int, n_periods: int, what: str) -> float | None:
    """A possibly per-period bound's value in one period, or None if unset."""
    if value is None:
        return None
    arr = _over_periods(value, n_periods, what)
    return float(arr[0] if arr.size == 1 else arr[p_index])


@dataclass
class StoragesData:
    """Storage parameters, as the four tables a storage declares.

    Absence is a missing row. A storage whose capacity is being optimized has
    no row in `capacity`; one that constrains neither its starting nor its
    final level has no row in `levels`. `profiles` is the exception and is
    dense on purpose: a missing coefficient row reads as a zero, and a
    retention of zero empties the store rather than leaving it alone.
    """

    #: (storage, charge_flow, discharge_flow, cyclic, prevent_simultaneous)
    storages: pl.DataFrame
    #: (storage, capacity) — only storages sized to a fixed value
    capacity: pl.DataFrame
    #: (storage, time, eta_charge, eta_discharge, loss, relative_level_min,
    #: relative_level_max) — dense, one row per storage and timestep
    profiles: pl.DataFrame
    #: (storage, prior_level, final_level_min, final_level_max) — only
    #: storages fixing a level at one end of the horizon
    levels: pl.DataFrame
    sizing: SizingData | None = None
    invest: InvestmentData | None = None

    @property
    def ids(self) -> list[str]:
        """The declared storages, in declaration order."""
        return self.storages['storage'].to_list()

    def __post_init__(self) -> None:
        """Re-check the ranges `Storage` already refuses, on the resolved values.

        Two things reach here that the element could not see: a reloaded
        file, which never passed through `Storage` at all, and a
        `ProfileRef`, whose numbers arrive when profiles are resolved and so
        are not there to check when the storage is written. Everything else
        was refused at construction — see docs/design/validation-layers.md.
        """
        bad_cap = self.capacity.filter(pl.col('capacity') < 0)['storage'].to_list()
        if bad_cap:
            raise ValueError(f'Negative capacity on storages: {bad_cap}')
        for outside, told in (
            ((pl.col('eta_charge') <= 0) | (pl.col('eta_charge') > 1), 'eta_charge must be in (0, 1]'),
            ((pl.col('eta_discharge') <= 0) | (pl.col('eta_discharge') > 1), 'eta_discharge must be in (0, 1]'),
            ((pl.col('loss') < 0) | (pl.col('loss') > 1), 'relative_loss_per_hour must be in [0, 1]'),
        ):
            bad = self.profiles.filter(outside)['storage'].unique(maintain_order=True).to_list()
            if bad:
                raise ValueError(f'{told} on storages: {bad}')

    @classmethod
    def build(
        cls,
        storages: list[Storage],
        time: TimeIndex,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Build StoragesData from element objects.

        Args:
            storages: Storage definitions.
            time: Time index.
            period: Period index for period-varying effects.
        """
        from fluxopt.elements import Investment, Sizing

        if not storages:
            return None

        labels = np.asarray(time)
        n_time = len(labels)

        fixed: list[tuple[str, float]] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        level_rows: list[tuple[str, float | None, float | None, float | None]] = []
        profile_cols: dict[str, list[np.ndarray]] = {
            k: []
            for k in (
                'storage',
                'time',
                'eta_charge',
                'eta_discharge',
                'loss',
                'relative_level_min',
                'relative_level_max',
            )
        }

        def series(value: Any) -> np.ndarray:
            return np.broadcast_to(as_dataarray(value, {'time': time}).values, (n_time,))

        for s in storages:
            if isinstance(s.capacity, Sizing):
                sizing_items.append((s.id, s.capacity))
            elif isinstance(s.capacity, Investment):
                invest_items.append((s.id, s.capacity))
            elif s.capacity is not None:
                fixed.append((s.id, float(s.capacity)))

            profile_cols['storage'].append(np.full(n_time, s.id))
            profile_cols['time'].append(labels)
            profile_cols['eta_charge'].append(series(s.eta_charge))
            profile_cols['eta_discharge'].append(series(s.eta_discharge))
            profile_cols['loss'].append(series(s.relative_loss_per_hour))
            profile_cols['relative_level_min'].append(series(s.relative_level_min))
            profile_cols['relative_level_max'].append(series(s.relative_level_max))

            if s.prior_level is not None or s.final_level_min is not None or s.final_level_max is not None:
                level_rows.append((s.id, s.prior_level, s.final_level_min, s.final_level_max))

        return cls(
            storages=pl.DataFrame(
                {
                    'storage': [s.id for s in storages],
                    'charge_flow': [s._charging_id for s in storages],
                    'discharge_flow': [s._discharging_id for s in storages],
                    'cyclic': [bool(s.cyclic) for s in storages],
                    'prevent_simultaneous': [bool(s.prevent_simultaneous) for s in storages],
                },
                schema={
                    'storage': pl.String,
                    'charge_flow': pl.String,
                    'discharge_flow': pl.String,
                    'cyclic': pl.Boolean,
                    'prevent_simultaneous': pl.Boolean,
                },
            ),
            capacity=pl.DataFrame(
                {'storage': [i for i, _ in fixed], 'capacity': [c for _, c in fixed]},
                schema={'storage': pl.String, 'capacity': pl.Float64},
            ),
            profiles=pl.DataFrame(
                {
                    key: pd.to_datetime(np.concatenate(parts)).to_pydatetime().tolist()
                    if key == 'time'
                    else np.concatenate(parts)
                    for key, parts in profile_cols.items()
                },
                schema={
                    'storage': pl.String,
                    'time': pl.Datetime('us'),
                    'eta_charge': pl.Float64,
                    'eta_discharge': pl.Float64,
                    'loss': pl.Float64,
                    'relative_level_min': pl.Float64,
                    'relative_level_max': pl.Float64,
                },
            ),
            levels=pl.DataFrame(
                {
                    'storage': [r[0] for r in level_rows],
                    'prior_level': [r[1] for r in level_rows],
                    'final_level_min': [r[2] for r in level_rows],
                    'final_level_max': [r[3] for r in level_rows],
                },
                schema={
                    'storage': pl.String,
                    'prior_level': pl.Float64,
                    'final_level_min': pl.Float64,
                    'final_level_max': pl.Float64,
                },
            ),
            sizing=SizingData.build(sizing_items, period=period),
            invest=InvestmentData.build(invest_items, period=period),
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
    """The model's temporal axes: what a timestep is, and what a period is.

    Two tables, indexed by position like every other container. The labels
    the user gave ride along as a column, because they are what a result is
    reported against and nothing else keys on them.

    Absence is a missing table: a system with no investment periods has an
    empty `periods`, which is what `has_periods` reads.
    """

    #: (time, label, dt, weight) — one row per timestep, in order
    timesteps: pl.DataFrame
    #: (period, label, weight) — one row per investment period, in order.
    #: Empty when the system declares no period axis.
    periods: pl.DataFrame

    @property
    def has_periods(self) -> bool:
        """Whether the system declares an investment period axis."""
        return not self.periods.is_empty()

    @property
    def n_time(self) -> int:
        """How many timesteps the horizon has."""
        return len(self.timesteps)

    @property
    def n_periods(self) -> int:
        """How many periods the program indexes — one, when none are declared."""
        return len(self.periods) or 1

    @property
    def period_labels(self) -> list[Any]:
        """The period labels, or ``[0]`` when the system declares no periods."""
        return self.periods['label'].to_list() if self.has_periods else [0]

    # The xarray views. A result is an xarray, so the readers that weigh one
    # against a duration want these as coordinates rather than as columns;
    # everything upstream of a solve reads the frames.
    @property
    def time(self) -> xr.DataArray:
        """Timestep labels as a coordinate."""
        labels = self.timesteps['label'].to_numpy()
        return xr.DataArray(labels, dims=['time'], coords={'time': labels}, name='time')

    @property
    def dt(self) -> xr.DataArray:
        """Timestep durations [h] as a time-indexed array."""
        return self._on_time('dt')

    @property
    def weights(self) -> xr.DataArray:
        """Timestep weights as a time-indexed array."""
        return self._on_time('weight')

    @property
    def period(self) -> xr.DataArray | None:
        """Period labels as a coordinate, or None when there is no period axis."""
        return self._on_period('label')

    @property
    def period_weights(self) -> xr.DataArray | None:
        """Period duration weights, or None when there is no period axis."""
        return self._on_period('weight')

    def _on_time(self, column: str) -> xr.DataArray:
        """One timestep column as an array indexed by the time labels."""
        labels = self.timesteps['label'].to_numpy()
        return xr.DataArray(self.timesteps[column].to_numpy(), dims=['time'], coords={'time': labels}, name=column)

    def _on_period(self, column: str) -> xr.DataArray | None:
        """One period column as an array indexed by the period labels, if any."""
        if not self.has_periods:
            return None
        labels = self.periods['label'].to_numpy()
        return xr.DataArray(self.periods[column].to_numpy(), dims=['period'], coords={'period': labels}, name=column)

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
        if period and (periods := self.period) is not None:
            result['period'] = periods
        return result

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
        labels = np.asarray(time)
        timesteps = pl.DataFrame(
            {
                'time': np.arange(len(labels)),
                'label': labels,
                'dt': np.asarray(dt.values, dtype=float),
                'weight': np.ones(len(labels)),
            }
        )
        if periods is None:
            period_table = pl.DataFrame(
                {'period': [], 'label': [], 'weight': []},
                schema={'period': pl.Int64, 'label': pl.Int64, 'weight': pl.Float64},
            )
        else:
            period_idx, weights = _compute_period_weights(periods, period_weights)
            period_table = pl.DataFrame(
                {
                    'period': np.arange(len(period_idx)),
                    'label': period_idx.values,
                    'weight': np.asarray(weights.values, dtype=float),
                },
                schema={'period': pl.Int64, 'label': pl.Int64, 'weight': pl.Float64},
            )
        return cls(timesteps=timesteps, periods=period_table)


#: Which sub-container each top-level one carries, and of what class. Read by
#: :meth:`ModelData.load` to rebuild them from their own directories.
_SUB_CONTAINERS: dict[str, dict[str, Any]] = {
    'flows': {'sizing': SizingData, 'invest': InvestmentData},
    'storages': {'sizing': SizingData, 'invest': InvestmentData},
}


def _frames_of(obj: Any) -> dict[str, pl.DataFrame]:
    """Every polars frame a container holds, by field name."""
    return {f.name: value for f in dataclasses.fields(obj) if isinstance(value := getattr(obj, f.name), pl.DataFrame)}


def _table_containers(obj: DataclassInstance) -> dict[str, Any]:
    """Nested container fields of a table object that are present (not None)."""
    return {
        f.name: getattr(obj, f.name)
        for f in fields(obj)
        if f.name in _CONTAINER_FIELD_NAMES and getattr(obj, f.name) is not None
    }


@dataclass
class ModelData:
    flows: FlowsData
    carriers: CarriersData
    converters: ConvertersData | None  # None when no linear converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dims: Dims
    piecewise: PiecewiseData | None = None  # None when no piecewise converters
    #: Every on/off decision in the system, whether a flow carries it or a
    #: component does. One table because the program has one `status_entity`
    #: dimension: the two differ only in which rows read the binary, and
    #: `flows.governed_by` is what says which.
    status: StatusData | None = None

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
        flow_ids = set(self.flows.ids)

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
        entities: set[str] = set(self.status.ids) if self.status is not None else set()
        if unknown := sorted(set(self.flows.governed_by['component'].to_list()) - entities):
            raise ValueError(f'flows.governed_by names components without a Status: {unknown}')
        self._check_status_not_degenerate(entities & flow_ids)
        if self.converters is not None:
            check_flows(self.converters.coefficients['flow'].to_list(), 'converters.coefficients')
        if self.piecewise is not None:
            check_flows(self.piecewise.links['flow'].to_list(), 'piecewise.links')
        if self.storages is not None:
            check_flows(self.storages.storages['charge_flow'].to_list(), 'storages.charge_flow')
            check_flows(self.storages.storages['discharge_flow'].to_list(), 'storages.discharge_flow')
            storage_ids = set(self.storages.ids)
            for container, what in (
                (self.storages.sizing, 'storages.sizing'),
                (self.storages.invest, 'storages.invest'),
            ):
                if container is None:
                    continue
                ids = container.ids
                if unknown := sorted(set(ids) - storage_ids):
                    raise ValueError(f'{what} references unknown storage id(s) {unknown}')

        effect_ids = set(self.effects.ids)
        coeff_effects = set(self.flows.effect_pairs['effect'].unique().to_list())
        if not coeff_effects <= effect_ids:
            raise ValueError(
                f'flows.effect_pairs names effects {sorted(coeff_effects - effect_ids)} that are not in '
                f'the effects table {sorted(effect_ids)}'
            )

    def save(self, path: str | Path) -> None:
        """Write the model data as a directory of tables.

        One parquet file per frame, in a directory per container. Parquet
        because these *are* tables: it carries the schema, so a column with no
        rows still knows it holds strings, and an empty frame reloads as the
        same empty frame rather than as something that lost its dtypes.

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
            for frame_name, frame in _frames_of(obj).items():
                frame.write_parquet(group / f'{frame_name}.parquet')
            for cname, sub in _table_containers(obj).items():
                sub_group = group / cname
                sub_group.mkdir(exist_ok=True)
                for frame_name, frame in _frames_of(sub).items():
                    frame.write_parquet(sub_group / f'{frame_name}.parquet')

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
            return klass(**frames, **(subs or {})) if frames else None

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
            status=read('status', StatusData),
        )

    def _check_status_not_degenerate(self, gated: set[str]) -> None:
        """A flow carrying its own Status needs rel_lb > 0, else on/off is degenerate.

        A zero lower bound lets the solver sit at zero with the binary on, so
        the status results mean nothing. `Flow` refuses it at construction;
        this is the guard for a direct edit or a reload, and it lives here
        rather than on `FlowsData` because the envelope and the status table
        are two containers now.

        Args:
            gated: Flows carrying a Status of their own.
        """
        if not gated:
            return
        rows = self.flows.envelope.filter(pl.col('flow').is_in(pl.Series(sorted(gated)).implode()))
        degenerate = rows.filter(pl.col('relative_rate_min') <= 0)['flow'].unique(maintain_order=True).to_list()
        if degenerate:
            raise ValueError(
                f'Status flows must have rel_lb > 0 (else on/off is indistinguishable); violated on {degenerate}'
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
            'status': self.status,
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
        durations = dims.timesteps['dt'].to_numpy()
        dt_scalar = float(durations[0])
        if any(bf.flow.prior_rates is not None for bf in flows) and not np.allclose(durations, dt_scalar):
            warnings.warn(
                f'prior_rates with non-uniform dt: pre-horizon status durations assume the first '
                f'timestep duration ({dt_scalar} h) for every prior step. If your prior steps had '
                f'different durations, adjust prior_rates to compensate.',
                UserWarning,
                stacklevel=2,
            )
        period_idx = pd.Index(dims.periods['label'].to_list()) if dims.has_periods else None

        comp_status_items: list[tuple[str, Status, list[str]]] = [
            (s.id, s.status, [s._charging_id, s._discharging_id]) for s in stor_list if s.status is not None
        ]
        comp_status_items.extend(
            (c.id, c.conversion.status, [bf.id for bf in c._qualified_flows()])
            for c in converters
            if c.conversion is not None and c.conversion.status is not None
        )

        flows_data = FlowsData.build(flows, time, period=period_idx, component_status_items=comp_status_items)
        # One table for one dimension: a flow's own Status and a component's
        # differ only in which rows read the binary, and the program says so
        # with a single `status_entity`. Flows first, so the axis reads in
        # declaration order.
        status_data = StatusData.build(
            [(bf.id, bf.flow.status) for bf in flows if bf.flow.status is not None]
            + [(cid, z) for cid, z, _ in comp_status_items],
            time,
            prior_rates_map={bf.id: bf.flow.prior_rates for bf in flows if bf.flow.prior_rates is not None},
            dt=dt_scalar,
            period=period_idx,
        )
        carriers_data = CarriersData.build(carriers, flows, carrier_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time, period=period_idx)
        storages_data = StoragesData.build(stor_list, time, period=period_idx)
        piecewise_data = PiecewiseData.build(converters, time)

        return cls(
            flows=flows_data,
            carriers=carriers_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dims=dims,
            piecewise=piecewise_data,
            status=status_data,
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
