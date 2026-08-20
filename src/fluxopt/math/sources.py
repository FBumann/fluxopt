"""Bind a :class:`~fluxopt.model_data.ModelData` to fluxopt's math program.

The data half of the build: this module emits the parameter tables
:data:`PROGRAM` declares, and lpspec does the rest.

Sparsity is carried by *row absence* — a parameter keeps its declared rank
while its table holds only live entries. Arrays at or below a variable's own
grid (bounds) stay dense; only the ones whose rank exceeds it
(``effects_per_flow_hour``, ``conversion_factor``) are filtered, which is where the size is.

Every relation between entities is a coordinate on the ``flow`` dimension
rather than a matrix, so topology travels as rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from fluxopt.contract import BoundType
from fluxopt.leontief import apply_leontief, leontief
from fluxopt.validation import reject_varying_contribution_into_lump

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: The YAML program holding fluxopt's math. Shipped as package data.
PROGRAM = Path(__file__).with_name('program.yaml')

#: Parameters the YAML declares with a `period` axis *and* emit without one.
#: Anything here is cross-joined onto every period. A frame-backed table
#: carries its own period column and is not in this list.
PERIOD_PARAMS = frozenset(
    {
        'rate_min',
        'rate_max',
        'rate_min_when_on',
        'rate_max_when_on',
        'rate_fixed_when_on',
        'relative_rate_min',
        'relative_rate_max',
        'fixed_relative_profile',
        'uptime_upper',
        'downtime_upper',
        'ramp_up_limit',
        'ramp_down_limit',
        'ramp_up_coeff',
        'ramp_down_coeff',
        'effects_per_flow_hour',
        'prior_level',
        'pw_avail_bound',
        'flow_hours_min',
        'flow_hours_max',
        'load_factor_min_bound',
        'load_factor_max_bound',
        'load_factor_min_coeff',
        'load_factor_max_coeff',
        'rate_max_at_size_max',
        'rate_min_at_size_max',
        'size_upper',
        'lifetime_window',
        'prior_capacity_active',
    }
)

#: Parameters carrying the `build_period` axis; its labels map to ordinals too.
#: The at-build coefficient tables are frame-backed and already carry theirs.
BUILD_PERIOD_PARAMS = frozenset({'lifetime_window'})


def _size_upper(data: ModelData, fid: str) -> float:
    """Static upper bound on a flow's size — fixed value or sizing max."""
    fds = data.flows
    val = float(fds.size.sel(flow=fid).values)
    if not np.isnan(val):
        return val
    if fds.sizing is not None:
        row = fds.sizing.bounds.filter(pl.col('entity') == fid)
        if len(row):
            return float(row['size_max'][0])
    return 0.0


#: Index columns the program declares with an integer dtype; every other index
#: column is a string label.
_INT_DIMS = frozenset({'time', 'period', 'build_period', 'eq_idx', 'bp'})

#: Parameters the program declares ``dtype: bool``.
_BOOL_PARAMS = frozenset(
    {
        'is_first',
        'is_last',
        'conversion_active',
        'is_cyclic',
        'is_gated',
        'is_piecewise',
        'has_piecewise',
        'pw_gated',
        'pw_equal',
        'pw_upper',
        'pw_lower',
        'pw_bp_present',
        'pw_seg_present',
        'is_bounded',
        'is_profile',
        'has_uptime',
        'has_downtime',
        'forced_on_at_start',
        'forced_off_at_start',
        'has_sizing',
        'size_optional',
        'has_invest',
        'invest_mandatory',
        'has_prior_capacity',
        'has_ramp_up',
        'has_ramp_down',
        'prevent_simultaneous',
        'has_capacity_sizing',
        'capacity_optional',
    }
)


def _stamp_empty_dtypes(sources: dict[str, Any]) -> None:
    """Give every zero-row table the dtypes its parameter declares.

    Pandas cannot type an empty column and picks ``float64``, which the engine
    reads as a numeric label space and refuses against a string dimension. A
    frame with rows carries its own types and is left alone; a frame without
    any has nothing to carry, so the declared types are stamped on here rather
    than guarded at each of the two dozen places one can be produced —
    ``_tidy`` over an all-masked array and the period re-indexing both make
    them.
    """
    for name, df in sources.items():
        if not isinstance(df, pd.DataFrame) or not df.empty:
            continue
        typed = {c: pd.Series([], dtype='int64' if c in _INT_DIMS else 'object') for c in df.columns if c != 'value'}
        typed['value'] = pd.Series([], dtype='bool' if name in _BOOL_PARAMS else 'float64')
        sources[name] = pd.DataFrame(typed)


def _flags(name: str, dim: str, ids: list[str]) -> pd.DataFrame:
    """A boolean table marking *ids* true — typed even when none qualify.

    The empty case is the one that bites: a list comprehension that filters
    everything out leaves pandas to type the label column, and it picks
    ``float64``. :func:`_empty` is what an absent feature looks like.
    """
    return _empty(name, dim) if not ids else pd.DataFrame({dim: ids, 'value': True})


def _empty(name: str, *index_cols: str) -> pd.DataFrame:
    """An empty table for *name*, carrying the dtypes the program declares.

    A parameter with no live entries still binds, and the engine checks a
    label column against its dimension's own — so an all-empty frame has to
    say what it would have held. Pandas types an empty column ``float64``,
    which reads as a numeric label space and is refused.
    """
    cols = {c: pd.Series([], dtype='int64' if c in _INT_DIMS else 'object') for c in index_cols}
    cols['value'] = pd.Series([], dtype='bool' if name in _BOOL_PARAMS else 'float64')
    return pd.DataFrame(cols)


class UnsupportedFeatureError(RuntimeError):
    """The ModelData uses a feature the program does not express yet.

    Raised rather than silently dropping the feature: a missing constraint
    would still solve, just to the wrong answer.
    """


def _tidy(da: xr.DataArray, *, drop_zero: bool, time_ord: dict[Any, int] | None = None) -> pd.DataFrame:
    """Tidy `(dims..., value)` frame; live rows only when *drop_zero*."""
    vals = da.values
    # NaN means "absent"; +/-inf is a legitimate bound and must survive
    keep = ~np.isnan(vals) if vals.dtype.kind == 'f' else np.ones(vals.shape, dtype=bool)
    if drop_zero and vals.dtype.kind == 'f':
        keep = keep & (vals != 0)
    idx = np.nonzero(keep)
    cols: dict[str, Any] = {}
    for dim, positions in zip(da.dims, idx, strict=True):
        labels = da.coords[dim].values[positions] if dim in da.coords else positions
        cols[str(dim)] = [time_ord[v] for v in labels] if (dim == 'time' and time_ord) else labels
    cols['value'] = vals[keep]
    return pd.DataFrame(cols)


def _with_time_ordinals(frame: pl.DataFrame, dims: Any) -> pl.DataFrame:
    """Replace a frame's ``time`` labels with the ordinals the program uses.

    The element layer indexes time by whatever the user gave it; the program
    indexes it by position, because a timestamp is a poor join key.

    Both sides are cast to one time unit first. A freshly built frame carries
    microseconds and one read back from netCDF carries nanoseconds, and polars
    refuses to join across the two — which is the good outcome, since the
    alternative is the failure the labels themselves have: numpy datetimes are
    nanoseconds and reading their raw integers as microseconds turns 2024 into
    the year 55969, so the join matches nothing rather than failing.
    """
    unit = pl.Datetime('us')
    labels = pd.to_datetime(dims.time.values).to_pydatetime().tolist()
    ordinals = pl.DataFrame({'time': labels, 'ord': list(range(len(labels)))}).with_columns(pl.col('time').cast(unit))
    return frame.with_columns(pl.col('time').cast(unit)).join(ordinals, on='time').drop('time').rename({'ord': 'time'})


def _fold_effect_rows(
    frame: pl.DataFrame, entity: str, column: str, leo: xr.DataArray | None, tidy: Any
) -> pl.DataFrame:
    """A per-timestep coefficient table, Leontief-folded on the rows.

    `_fold_lump_rows` for the temporal domain: the factor may vary along time
    and period as well, so the join keys follow whatever axes it carries.
    """
    rows = frame.select(
        [pl.col('entity').alias(entity), 'effect', 'time', 'period', pl.col(column).alias('value')]
    ).filter(pl.col('value') != 0)
    if leo is None or rows.is_empty():
        return rows
    factors = pl.from_pandas(tidy(leo, drop_zero=True)).rename({'effect': 'charged', 'source_effect': 'effect'})
    on = ['effect', *[c for c in ('time', 'period') if c in factors.columns and c in rows.columns]]
    return (
        rows.join(factors, on=on, suffix='_leo')
        .with_columns((pl.col('value') * pl.col('value_leo')).alias('value'))
        .group_by([entity, 'charged', 'time', 'period'])
        .agg(pl.col('value').sum())
        .rename({'charged': 'effect'})
        .select([entity, 'effect', 'time', 'period', 'value'])
    )


def _fold_lump_rows(frame: pl.DataFrame, entity: str, column: str, leo: xr.DataArray | None) -> pl.DataFrame:
    """A lump coefficient table, Leontief-folded on the rows.

    The same contraction `_flow_hour_coefficients` does, on the lump domain:
    every declared (entity, effect) coefficient is joined against the effects
    it feeds and the products summed back. Written once because every lump
    container wants it — flow sizing, storage sizing and investment alike.

    Stays in polars throughout. lpspec takes either, and the arithmetic here
    is a join and a group-by, which is what a dataframe is for.
    """
    # The container keys on `entity` because it serves flows and storages
    # alike; the parameter it feeds names the one it is for.
    rows = frame.select([pl.col('entity').alias(entity), 'effect', 'period', pl.col(column).alias('value')]).filter(
        pl.col('value') != 0
    )
    if leo is None or rows.is_empty():
        return rows

    factors = pl.from_pandas(_tidy(leo, drop_zero=True)).rename({'effect': 'charged', 'source_effect': 'effect'})
    on = ['effect', *[c for c in ('period',) if c in factors.columns and c in rows.columns]]
    return (
        rows.join(factors, on=on, suffix='_leo')
        .with_columns((pl.col('value') * pl.col('value_leo')).alias('value'))
        .group_by([entity, 'charged', 'period'])
        .agg(pl.col('value').sum())
        .rename({'charged': 'effect'})
        .select([entity, 'effect', 'period', 'value'])
    )


def _flow_hour_coefficients(
    fds: Any,
    dims: Any,
    leo: xr.DataArray | None,
    tidy: Any,
) -> pd.DataFrame:
    """The per-flow-hour coefficient table, never materialised dense.

    `effect_pair_coeff` already holds one row per (flow, effect) a flow
    actually charges. What used to happen here was to broadcast that back to
    the full `(flow, effect, time, period)` product so the Leontief contraction
    could be a `xr.dot` over the effect axis — 443 MB at 4% live on the stress
    reference system, thrown away one line later by `drop_zero`.

    So the contraction happens on the rows instead: every live pair is joined
    against the effects it feeds and the products summed back per
    (flow, effect). Same numbers, and the widest thing built is the result.
    """
    pairs = tidy(fds.effect_pair_coeff * dims.dt, drop_zero=True)
    if pairs.empty:
        return _empty('effects_per_flow_hour', 'flow', 'effect', 'time')
    index = pairs.pop('effect_pair').to_numpy().astype(int)
    pairs.insert(0, 'flow', fds.effect_pair_flow.values[index])
    pairs.insert(1, 'effect', fds.effect_pair_effect.values[index])
    if leo is None:
        return pairs

    # (effect, source_effect[, time]) -> rows, dropping the zeros that make a
    # sparse effect graph sparse.
    factors = tidy(leo, drop_zero=True).rename(columns={'effect': 'charged', 'source_effect': 'effect'})
    # Join on every axis the factor itself varies along: `contribution_from`
    # may differ per timestep and per period, and joining on fewer keys than
    # it carries would silently cross-multiply them.
    on = ['effect', *[c for c in ('time', 'period') if c in factors.columns and c in pairs.columns]]
    merged = pairs.merge(factors, on=on, suffixes=('', '_leo'))
    merged['value'] = merged['value'] * merged['value_leo']
    keys = ['flow', 'charged', *[c for c in ('time', 'period') if c in pairs.columns]]
    out = merged.groupby(keys, as_index=False)['value'].sum().rename(columns={'charged': 'effect'})
    return out[[c for c in pairs.columns if c != 'value'] + ['value']]


def _reject_unsupported(data: ModelData) -> None:
    fds = data.flows
    if data.piecewise is not None:
        bad = sorted(set(data.piecewise.curves['method'].to_list()) & {'lp'})
        if bad:
            raise UnsupportedFeatureError(
                "piecewise method 'lp' is linopy's tangent-line relaxation, which this lane has no "
                'formulation for — the adjacency formulation it does have is exact, so it would '
                'answer a different question. Use the default method, or lpspec #695.'
            )
    if fds.invest is not None and data.dims.period is None:
        raise UnsupportedFeatureError('investment requires multi-period optimization (periods must be specified)')
    if fds.sizing is not None and fds.status is not None:
        sz = set(fds.sizing.ids)
        stt = set(fds.status.ids)
        both = sz & stt
        profile = {f for f in both if str(fds.bound_type.sel(flow=f).values) == BoundType.PROFILE}
        if profile:
            # fluxopt rejects this combination too — no formulation exists.
            raise UnsupportedFeatureError(f'fixed profile with status+sizing has no formulation: {sorted(profile)}')


def build_sources(data: ModelData, objective: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Emit the parameter tables and coordinates for :data:`PROGRAM`.

    Args:
        data: The model data to bind. Both backends read the same object.
        objective: Effect ids mapped to their objective weight, as
            :func:`~fluxopt.math.solve.solve` takes it.

    Returns:
        ``(sources, coords)`` ready to pass to ``lpspec.solve``.

    Raises:
        UnsupportedFeatureError: If *data* uses a feature the program does
            not express yet, rather than dropping it silently.
    """
    _reject_unsupported(data)
    reject_varying_contribution_into_lump(data)
    fds, dims = data.flows, data.dims

    time_labels = list(dims.time.values)
    time_ord = {v: i for i, v in enumerate(time_labels)}
    ordinals = list(range(len(time_labels)))

    def tidy(da: xr.DataArray, *, drop_zero: bool) -> pd.DataFrame:
        """`_tidy` with this model's time-ordinal mapping bound in."""
        return _tidy(da, drop_zero=drop_zero, time_ord=time_ord)

    flow_ids = [str(f) for f in fds.size.coords['flow'].values]
    sources: dict[str, Any] = {}
    # Bound up front: the feature blocks below fill these only when the
    # corresponding container is present, but later blocks read them.
    sizing_ids: list[str] = []
    status_ids: list[str] = []
    has_sizing = xr.zeros_like(fds.size, dtype=bool)
    # Lump-domain accumulators, filled by the flow- and storage-sizing blocks.
    # (parameter name, entity dim, coefficients) — the entity dim is carried so
    # an absent term still emits a correctly keyed empty table.
    lump_terms: list[tuple[str, str, xr.DataArray | None]] = []
    #: (parameter, entity dim, frame, value column) — the frame-backed ones
    lump_frames: list[tuple[str, str, pl.DataFrame, str]] = []
    #: (parameter, frame, value column) — charged where the build happened
    at_build_frames: list[tuple[str, pl.DataFrame, str]] = []

    # --- flow rate bounds: dense, they sit at the variable's own grid -----
    size, bt = fds.size, fds.bound_type
    zero = xr.zeros_like(fds.rel_lb)
    lb = xr.where(bt == BoundType.BOUNDED, size * fds.rel_lb, zero)
    ub = xr.where(bt == BoundType.BOUNDED, size * fds.rel_ub, xr.full_like(zero, np.inf))
    is_profile = (bt == BoundType.PROFILE) & fds.fixed_profile.notnull()
    lb = xr.where(is_profile, size * fds.fixed_profile, lb)
    ub = xr.where(is_profile, size * fds.fixed_profile, ub)
    sz = fds.sizing
    if sz is not None:
        sizing_ids = sz.ids
        has_sizing = xr.DataArray(
            [f in set(sizing_ids) for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']}
        )
        lb = xr.where(has_sizing, 0.0, lb)
        ub = xr.where(has_sizing, np.inf, ub)

    inv = fds.invest
    invest_ids: list[str] = []
    has_invest = xr.zeros_like(fds.size, dtype=bool)
    if inv is not None:
        invest_ids = inv.ids
        has_invest = xr.DataArray(
            [f in set(invest_ids) for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']}
        )
        lb = xr.where(has_invest, 0.0, lb)
        ub = xr.where(has_invest, np.inf, ub)

    st = fds.status
    if st is not None:
        status_ids = st.ids
        has_status = xr.DataArray(
            [f in set(status_ids) for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']}
        )
        # `on` carries the envelope for status flows; the variable itself is free above 0
        lb = xr.where(has_status, 0.0, lb)
        ub = xr.where(
            has_status,
            xr.where(bt == BoundType.PROFILE, size * fds.fixed_profile, size * fds.rel_ub),
            ub,
        )
        if sz is not None:
            ub = xr.where(has_status & has_sizing, np.inf, ub)
    sources['rate_min'] = tidy(lb, drop_zero=False)
    sources['rate_max'] = tidy(ub, drop_zero=False)

    # --- carrier balance --------------------------------------------------
    membership = data.carriers.membership
    flow_index = pd.DataFrame({'flow': flow_ids, 'carrier_of': membership['carrier'].to_list()})
    sources['carrier_sign'] = pd.DataFrame({'flow': flow_ids, 'value': membership['sign'].to_numpy()})

    # --- converters -------------------------------------------------------
    if data.converters is not None:
        cds = data.converters
        coeffs = cds.coefficients
        conv_of = dict(zip(coeffs['flow'], coeffs['converter'], strict=True))
        flow_index['converter_of'] = [conv_of.get(f) for f in flow_ids]
        # Already the table the parameter wants — only the time labels are
        # the element layer's and the program indexes them by ordinal.
        sources['conversion_factor'] = _with_time_ordinals(coeffs.filter(pl.col('value') != 0), dims).select(
            ['flow', 'eq_idx', 'time', 'value']
        )
        # One row per equation each converter states — the counts, expanded.
        sources['conversion_active'] = pl.DataFrame(
            {
                'converter': [c for c, n in zip(cds.ids, cds.equations['n_equations'], strict=True) for _ in range(n)],
                'eq_idx': [i for n in cds.equations['n_equations'] for i in range(n)],
            }
        ).with_columns(pl.lit(True).alias('value'))
    else:
        flow_index['converter_of'] = None
        sources['conversion_factor'] = _empty('conversion_factor', 'flow', 'eq_idx', 'time')
        sources['conversion_active'] = _empty('conversion_active', 'converter', 'eq_idx')

    # --- storage ----------------------------------------------------------
    storage_ids: list[str] = []
    if data.storages is not None:
        sds = data.storages
        storage_ids = sds.ids
        charge = sds.storages['charge_flow'].to_list()
        discharge = sds.storages['discharge_flow'].to_list()
        chg_of = dict(zip(charge, storage_ids, strict=True))
        dis_of = dict(zip(discharge, storage_ids, strict=True))
        flow_index['charge_storage'] = [chg_of.get(f) for f in flow_ids]
        flow_index['discharge_storage'] = [dis_of.get(f) for f in flow_ids]

        dt_by_time = pl.DataFrame({'time': ordinals, 'dt': dims.dt.values})
        # One join carries every per-timestep storage parameter, since they
        # all live on the same (storage, time) rows.
        profiles = _with_time_ordinals(sds.profiles, dims).join(dt_by_time, on='time')

        def on_flow(frame: pl.DataFrame, column: str, of_storage: dict[str, str]) -> pl.DataFrame:
            """A per-storage coefficient read on the flow that carries it."""
            renamed = {s: f for f, s in of_storage.items()}
            return (
                frame.select([pl.col('storage').replace_strict(renamed).alias('flow'), 'time', pl.col(column)])
                .rename({column: 'value'})
                .filter(pl.col('value') != 0)
            )

        gains = profiles.with_columns(
            (pl.col('eta_charge') * pl.col('dt')).alias('charge_gain'),
            (pl.col('dt') / pl.col('eta_discharge')).alias('discharge_draw'),
            ((1 - pl.col('loss')) ** pl.col('dt')).alias('retention'),
        )
        sources['charge_gain'] = on_flow(gains, 'charge_gain', chg_of)
        sources['discharge_draw'] = on_flow(gains, 'discharge_draw', dis_of)
        sources['retention'] = gains.select(['storage', 'time', pl.col('retention').alias('value')])

        # An absent capacity row is a storage whose capacity is a variable, so
        # its absolute level bounds are not knowable here: 0 and infinity are
        # what the program reads while the relative pair does the bounding.
        absolute = profiles.join(sds.capacity, on='storage', how='left')
        sources['level_min'] = absolute.select(
            ['storage', 'time', (pl.col('relative_level_min') * pl.col('capacity')).fill_null(0.0).alias('value')]
        )
        sources['level_max'] = absolute.select(
            ['storage', 'time', (pl.col('relative_level_max') * pl.col('capacity')).fill_null(np.inf).alias('value')]
        )
        csz = sds.sizing
        if csz is not None:
            cap_ids = csz.ids
            copt = csz.bounds.filter(~pl.col('mandatory'))['entity'].to_list()
            sources['has_capacity_sizing'] = _flags('has_capacity_sizing', 'storage', cap_ids)
            sources['capacity_optional'] = _flags('capacity_optional', 'storage', copt)
            sources['capacity_min'] = pd.DataFrame({'storage': cap_ids, 'value': csz.bounds['size_min'].to_numpy()})
            sources['capacity_max'] = pd.DataFrame({'storage': cap_ids, 'value': csz.bounds['size_max'].to_numpy()})
            sized = profiles.filter(pl.col('storage').is_in(pl.Series(cap_ids).implode()))
            for key, column in (
                ('relative_level_min', 'relative_level_min'),
                ('relative_level_max', 'relative_level_max'),
            ):
                sources[key] = sized.select(['storage', 'time', pl.col(column).alias('value')]).filter(
                    pl.col('value') != 0
                )
            lump_frames += [
                ('effects_per_capacity', 'storage', csz.effects, 'per_size'),
                ('effects_fixed_capacity', 'storage', csz.effects, 'fixed'),
            ]
        for key, column in (('is_cyclic', 'cyclic'), ('prevent_simultaneous', 'prevent_simultaneous')):
            sources[key] = _flags(key, 'storage', sds.storages.filter(pl.col(column))['storage'].to_list())
        # `prior_level` is dense: it sits on the constant side of the first
        # step's balance, where an absent row is a binding zero — which is
        # also what an unset prior level means.
        sources['prior_level'] = (
            sds.storages.select('storage')
            .join(sds.levels.select(['storage', pl.col('prior_level').alias('value')]), on='storage', how='left')
            .with_columns(pl.col('value').fill_null(0.0))
        )
        for key in ('final_level_min', 'final_level_max'):
            sources[key] = sds.levels.select(['storage', pl.col(key).alias('value')]).drop_nulls()
        for key, flows in (('charge_size_bound', charge), ('discharge_size_bound', discharge)):
            sources[key] = pd.DataFrame({'storage': storage_ids, 'value': [_size_upper(data, f) for f in flows]})
    else:
        flow_index['charge_storage'] = None
        flow_index['discharge_storage'] = None
        for name, dcols in (
            ('is_cyclic', ['storage']),
            ('prior_level', ['storage']),
            ('final_level_min', ['storage']),
            ('final_level_max', ['storage']),
            ('prevent_simultaneous', ['storage']),
            ('charge_size_bound', ['storage']),
            ('discharge_size_bound', ['storage']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})
        for name, dcols in (
            ('charge_gain', ['flow', 'time']),
            ('discharge_draw', ['flow', 'time']),
            ('retention', ['storage', 'time']),
            ('level_min', ['storage', 'time']),
            ('level_max', ['storage', 'time']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})

    # --- status -----------------------------------------------------------
    # Two containers, one axis. `flows.status` is a flow's own on/off decision;
    # `flows.cstatus` is a component's, governing several flows at once. They
    # carry the same fields and obey the same math, so they are bound to one
    # `status_entity` dimension and the program states the family once. Which
    # rows read which binary is `status_of`, and nothing else distinguishes
    # them.
    ec_extra: list[tuple[str, xr.DataArray]] = []
    #: (parameter, frame, value column) — status coefficients, per timestep
    status_effect_frames: list[tuple[str, pl.DataFrame, str]] = []
    cst = fds.cstatus
    blocks: list[tuple[Any, list[str]]] = []
    #: flow id -> the entity whose binary gates it. A self-status flow maps to
    #: itself; a governed flow to its component; an ungated flow to nothing.
    status_of: dict[str, str] = {}
    if st is not None:
        blocks.append((st, status_ids))
        status_of.update({f: f for f in status_ids})
    if cst is not None:
        comp_ids = cst.ids
        blocks.append((cst, comp_ids))
        # A piecewise curve's flows are gated by its convexity row, which
        # already pins every weight to zero when the binary is off. Gating
        # them a second time per flow would be redundant, and wrong for the
        # links the curve only bounds.
        pw_comps = set(data.piecewise.converter_ids()) if data.piecewise is not None else set()
        if fds.governed_by is not None:
            status_of.update(
                {
                    fid: str(owner)
                    for fid, owner in zip(flow_ids, fds.governed_by.values, strict=True)
                    if str(owner) and str(owner) not in pw_comps
                }
            )

    flow_index['status_of'] = [status_of.get(f) for f in flow_ids]
    entity_ids = [e for _, ids in blocks for e in ids]
    gated_ids = [f for f in flow_ids if f in status_of]

    sources['is_gated'] = _flags('is_gated', 'flow', gated_ids)
    sources['is_bounded'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values})
    sources['is_profile'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values})

    if blocks:
        # Every block's frames stacked on the shared entity axis. A vertical
        # concat is what "the same table for another family" means, and the
        # nulls that stood for "this block does not set that field" are now
        # simply rows that are not there.
        durations = pl.concat([b.durations for b, _ in blocks], how='vertical')
        prior = pl.concat([b.prior for b, _ in blocks], how='vertical')
        status_effects = pl.concat([b.effects for b, _ in blocks], how='vertical')
        all_entities = pl.DataFrame({'status_entity': entity_ids}, schema={'status_entity': pl.String})

        def per_entity(frame: pl.DataFrame, column: str) -> Any:
            """One column of a frame, keyed on the entity axis, live rows only."""
            live = frame.filter(pl.col(column).is_not_null()).select(
                [pl.col('entity').alias('status_entity'), pl.col(column).alias('value')]
            )
            return live if len(live) else _empty(column, 'status_entity')

        # Envelopes are per gated flow: a governed flow is sized like any
        # other, and `size * rel` is what the binary scales.
        gsel = {'flow': gated_ids}
        sources['rate_min_when_on'] = tidy((size * fds.rel_lb).sel(gsel), drop_zero=True)
        sources['rate_max_when_on'] = tidy((size * fds.rel_ub).sel(gsel), drop_zero=True)
        sources['rate_fixed_when_on'] = tidy((size * fds.fixed_profile).sel(gsel), drop_zero=True)

        horizon = float(dims.dt.sum())
        bounded_up = durations.filter(pl.col('uptime_min').is_not_null() | pl.col('uptime_max').is_not_null())[
            'entity'
        ].to_list()
        bounded_down = durations.filter(pl.col('downtime_min').is_not_null() | pl.col('downtime_max').is_not_null())[
            'entity'
        ].to_list()
        sources['has_uptime'] = _flags('has_uptime', 'status_entity', bounded_up)
        sources['has_downtime'] = _flags('has_downtime', 'status_entity', bounded_down)
        sources['uptime_min'] = per_entity(durations, 'uptime_min')
        sources['downtime_min'] = per_entity(durations, 'downtime_min')
        sources['initial_status'] = per_entity(prior, 'initial')
        sources['previous_uptime'] = per_entity(prior, 'previous_uptime')
        sources['previous_downtime'] = per_entity(prior, 'previous_downtime')

        # Big-M covers the whole horizon plus whatever ran before it. An
        # entity with no prior has no row in `prior`, and no prior is zero.
        big_m = all_entities.join(
            prior.select([pl.col('entity').alias('status_entity'), 'previous_uptime', 'previous_downtime']),
            on='status_entity',
            how='left',
        ).with_columns(
            (pl.col('previous_uptime').fill_null(0.0) + horizon).alias('up'),
            (pl.col('previous_downtime').fill_null(0.0) + horizon).alias('down'),
        )
        sources['uptime_big_m'] = big_m.select(['status_entity', pl.col('up').alias('value')])
        sources['downtime_big_m'] = big_m.select(['status_entity', pl.col('down').alias('value')])

        # The duration variables' own upper bound, over the entity's timeline:
        # the declared maximum where there is one, the big-M where there is not.
        steps = pl.DataFrame({'time': ordinals}, schema={'time': pl.Int64})
        for kind, ids, declared, fallback in (
            ('uptime', bounded_up, 'uptime_max', 'up'),
            ('downtime', bounded_down, 'downtime_max', 'down'),
        ):
            ceiling = (
                pl.DataFrame({'status_entity': ids}, schema={'status_entity': pl.String})
                .join(durations.select([pl.col('entity').alias('status_entity'), declared]), on='status_entity')
                .join(big_m.select(['status_entity', fallback]), on='status_entity')
                .with_columns(pl.col(declared).fill_null(pl.col(fallback)).alias('value'))
                .select(['status_entity', 'value'])
            )
            sources[f'{kind}_upper'] = ceiling.join(steps, how='cross').select(['status_entity', 'time', 'value'])

        # A prior run shorter than the minimum forces continuation.
        for forced_key, prev_column, min_column in (
            ('forced_on_at_start', 'previous_uptime', 'uptime_min'),
            ('forced_off_at_start', 'previous_downtime', 'downtime_min'),
        ):
            forced = prior.join(durations.select(['entity', min_column]), on='entity', how='inner').filter(
                (pl.col(prev_column) > 0) & (pl.col(prev_column) < pl.col(min_column))
            )
            sources[forced_key] = _flags(forced_key, 'status_entity', forced['entity'].to_list())

        # `dt` turns a per-running-hour rate into the step's cost; a startup
        # happens once at the step, so it is not scaled.
        dt_by_time = pl.DataFrame({'time': ordinals, 'dt': dims.dt.values})
        scaled = (
            _with_time_ordinals(status_effects, dims)
            .join(dt_by_time, on='time')
            .with_columns((pl.col('running') * pl.col('dt')).alias('running'))
        )
        status_effect_frames += [
            ('effects_per_running_hour', scaled, 'running'),
            ('effects_per_startup', scaled, 'startup'),
        ]
    else:
        for n in ('has_uptime', 'has_downtime'):
            sources[n] = _empty(n, 'status_entity')
        for n in (
            'uptime_min',
            'uptime_big_m',
            'downtime_big_m',
            'downtime_min',
            'initial_status',
            'previous_uptime',
            'previous_downtime',
            'forced_on_at_start',
            'forced_off_at_start',
        ):
            sources[n] = _empty(n, 'status_entity')
        for n in ('uptime_upper', 'downtime_upper'):
            sources[n] = _empty(n, 'status_entity', 'time')
        for n in ('rate_min_when_on', 'rate_max_when_on', 'rate_fixed_when_on'):
            sources[n] = _empty(n, 'flow', 'time')

    sources['dt'] = pd.DataFrame({'time': ordinals, 'value': dims.dt.values})
    sources['is_last'] = pd.DataFrame({'time': ordinals, 'value': [i == len(ordinals) - 1 for i in ordinals]})

    # --- sizing -----------------------------------------------------------
    if sz is not None:
        bounds = sz.bounds
        opt_ids = bounds.filter(~pl.col('mandatory'))['entity'].to_list()
        sources['has_sizing'] = _flags('has_sizing', 'flow', sizing_ids)
        sources['size_optional'] = _flags('size_optional', 'flow', opt_ids)
        sources['size_min'] = pd.DataFrame({'flow': sizing_ids, 'value': bounds['size_min'].to_numpy()})
        sources['size_max'] = pd.DataFrame({'flow': sizing_ids, 'value': bounds['size_max'].to_numpy()})
        zsel = {'flow': sizing_ids}
        smax = xr.DataArray(bounds['size_max'].to_numpy(), dims=['flow'], coords={'flow': sizing_ids})
        sources['rate_max_at_size_max'] = tidy((fds.rel_ub.sel(zsel) * smax), drop_zero=True)
        # Dense: this one also stands on the constant side of
        # `status_sizing_rate_min`, where a dropped zero is a bound rather
        # than an absent coefficient. A flow whose lower bound is zero is the
        # ordinary case, so dropping it would break exactly the common one.
        sources['rate_min_at_size_max'] = tidy((fds.rel_lb.sel(zsel) * smax), drop_zero=False)
        lump_frames += [
            ('effects_per_size', 'flow', sz.effects, 'per_size'),
            ('effects_fixed', 'flow', sz.effects, 'fixed'),
        ]
    else:
        for n in ('has_sizing', 'size_optional', 'size_min', 'size_max'):
            sources[n] = _empty(n, 'flow')
        for n in ('rate_max_at_size_max', 'rate_min_at_size_max'):
            sources[n] = _empty(n, 'flow', 'time')

    if 'has_capacity_sizing' not in sources:
        for n in ('has_capacity_sizing', 'capacity_optional', 'capacity_min', 'capacity_max'):
            sources[n] = _empty(n, 'storage')
        for n in ('relative_level_min', 'relative_level_max'):
            sources[n] = _empty(n, 'storage', 'time')

    # --- ramps ------------------------------------------------------------
    sized_set = set(sizing_ids) if sz is not None else set()
    for kind, arr in (('up', fds.ramp_up), ('down', fds.ramp_down)):
        if arr is None:
            sources[f'has_ramp_{kind}'] = _empty(f'has_ramp_{kind}', 'flow')
            sources[f'ramp_{kind}_limit'] = _empty(f'ramp_{kind}_limit', 'flow', 'time')
            sources[f'ramp_{kind}_coeff'] = _empty(f'ramp_{kind}_coeff', 'flow', 'time')
            continue
        nonflow = [d for d in arr.dims if d != 'flow']
        live = arr.notnull().any(nonflow) if nonflow else arr.notnull()
        ids = [str(f) for f in arr.coords['flow'].values[live.values]]
        sources[f'has_ramp_{kind}'] = _flags(f'has_ramp_{kind}', 'flow', ids)
        coeff = (arr * dims.dt).sel(flow=ids)
        sources[f'ramp_{kind}_coeff'] = tidy(coeff.sel(flow=[f for f in ids if f in sized_set]), drop_zero=True)
        fixed = [f for f in ids if f not in sized_set]
        sources[f'ramp_{kind}_limit'] = tidy((coeff * size).sel(flow=fixed), drop_zero=True)
    sources['ramp_bigM'] = pd.DataFrame(
        {'flow': flow_ids, 'value': [_size_upper(data, f) for f in flow_ids]},
    )

    # --- investment -------------------------------------------------------
    if inv is not None:
        period_labels_inv: list[Any] = list(dims.period.values) if dims.period is not None else []
        n_p = len(period_labels_inv)
        # No row means forever, so the lookup's default is the absence.
        expires = dict(zip(inv.lifetime['entity'], inv.lifetime['periods'], strict=True))
        lifetime = [expires.get(f) for f in invest_ids]
        prior = inv.bounds['prior_size'].to_numpy()
        window = np.zeros((len(invest_ids), n_p, n_p))
        prior_active = np.zeros((len(invest_ids), n_p))
        for f_idx in range(len(invest_ids)):
            lt_int = lifetime[f_idx]
            for p_idx in range(n_p):
                for b_idx in range(n_p):
                    alive = b_idx <= p_idx if lt_int is None else b_idx <= p_idx < b_idx + lt_int
                    window[f_idx, p_idx, b_idx] = float(alive)
                if prior[f_idx] > 0 and (lt_int is None or p_idx < lt_int):
                    prior_active[f_idx, p_idx] = 1.0
        coords_w = {'flow': invest_ids, 'period': period_labels_inv, 'build_period': period_labels_inv}
        sources['lifetime_window'] = _tidy(
            xr.DataArray(window, dims=['flow', 'period', 'build_period'], coords=coords_w), drop_zero=True
        )
        sources['prior_capacity_active'] = _tidy(
            xr.DataArray(
                prior_active, dims=['flow', 'period'], coords={'flow': invest_ids, 'period': period_labels_inv}
            ),
            drop_zero=False,
        )
        sources['has_invest'] = _flags('has_invest', 'flow', invest_ids)
        sources['invest_mandatory'] = pd.DataFrame({'flow': invest_ids, 'value': inv.bounds['mandatory'].to_numpy()})
        sources['invest_min'] = pd.DataFrame({'flow': invest_ids, 'value': inv.bounds['size_min'].to_numpy()})
        sources['invest_max'] = pd.DataFrame({'flow': invest_ids, 'value': inv.bounds['size_max'].to_numpy()})
        sources['prior_capacity'] = pd.DataFrame({'flow': invest_ids, 'value': prior})
        sources['has_prior_capacity'] = _flags(
            'has_prior_capacity', 'flow', [f for f, ps in zip(invest_ids, prior, strict=True) if ps > 0]
        )
        lump_frames += [
            ('effects_per_size_recurring', 'flow', inv.effects, 'per_size_recurring'),
            ('effects_fixed_recurring', 'flow', inv.effects, 'fixed_recurring'),
        ]
        # A one-time cost belongs to the period the build happened in, not to
        # every period the unit is alive — which as rows is the diagonal: the
        # same period twice, rather than a square matrix multiplied by an eye.
        at_build_frames += [
            ('effects_per_size_at_build', inv.effects, 'per_size_at_build'),
            ('effects_fixed_at_build', inv.effects, 'fixed_at_build'),
        ]
    else:
        for name in (
            'has_invest',
            'invest_mandatory',
            'invest_min',
            'invest_max',
            'prior_capacity',
            'has_prior_capacity',
        ):
            sources[name] = _empty(name, 'flow')
        sources['lifetime_window'] = _empty('lifetime_window', 'flow', 'period', 'build_period')
        sources['prior_capacity_active'] = _empty('prior_capacity_active', 'flow', 'period')

    # `sizing_rate` covers both mechanisms, so its envelope must span both.
    sized_ids = [*sizing_ids, *invest_ids]
    envelope_sel = {'flow': sized_ids}
    sources['relative_rate_min'] = tidy(fds.rel_lb.sel(envelope_sel), drop_zero=True)
    sources['relative_rate_max'] = tidy(fds.rel_ub.sel(envelope_sel), drop_zero=True)
    sources['fixed_relative_profile'] = tidy(fds.fixed_profile.sel(envelope_sel), drop_zero=True)

    # upper bound of `size`: whichever mechanism sizes the flow
    size_upper = xr.full_like(fds.size, np.inf)
    for ids, maxima in (
        (sizing_ids, sz.bounds['size_max'].to_numpy() if sz is not None else None),
        (invest_ids, inv.bounds['size_max'].to_numpy() if inv is not None else None),
    ):
        if maxima is None or not ids:
            continue
        per_flow = xr.DataArray(maxima, dims=['flow'], coords={'flow': ids}).reindex(flow=flow_ids)
        size_upper = xr.where(per_flow.notnull(), per_flow, size_upper)
    sources['size_upper'] = _tidy(size_upper, drop_zero=False)

    # --- effects: the sparse one -----------------------------------------
    eds = data.effects
    effect_ids = eds.ids
    # dt stays: a per-flow-hour rate times a duration is the step's energy.
    # The aggregation weight does not — the program applies it in the sum,
    # so a named contribution reads as the physical per-step quantity.
    cf = eds.cf_matrix(dims.period.values.tolist() if dims.period is not None else None)
    leo = leontief(cf) if cf is not None else None
    sources['effects_per_flow_hour'] = _flow_hour_coefficients(fds, dims, leo, tidy)
    for name, arr in ec_extra:
        sources[name] = tidy(apply_leontief(leo, arr) if leo is not None else arr, drop_zero=True)
    for name, frame, column in status_effect_frames:
        sources[name] = _fold_effect_rows(frame, 'status_entity', column, leo, tidy)
    for name in ('effects_per_running_hour', 'effects_per_startup'):
        sources.setdefault(name, _empty(name, 'status_entity', 'effect', 'time', 'period'))

    # Lump domain: effect_lump = (I - cf_lump)^-1 . lump_direct, folded into
    # the coefficients so no self-referential effect_lump variable is needed.
    leo_lump = leontief(cf.mean('time')) if cf is not None else None

    def fold(arr: xr.DataArray) -> xr.DataArray:
        """Apply the lump-domain Leontief inverse, if there are cross-effects."""
        return arr if leo_lump is None else apply_leontief(leo_lump, arr)

    for name, entity_dim, arr in lump_terms:
        sources[name] = tidy(fold(arr), drop_zero=True) if arr is not None else _empty(name, entity_dim, 'effect')
    for name, entity_dim, frame, column in lump_frames:
        sources[name] = _fold_lump_rows(frame, entity_dim, column, leo_lump)
    for name, frame, column in at_build_frames:
        folded = _fold_lump_rows(frame, 'flow', column, leo_lump)
        sources[name] = folded.with_columns(pl.col('period').alias('build_period'))
    for name in ('effects_per_size', 'effects_fixed'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period'))
    for name in ('effects_per_capacity', 'effects_fixed_capacity'):
        sources.setdefault(name, _empty(name, 'storage', 'effect', 'period'))
    for name in ('effects_per_size_at_build', 'effects_fixed_at_build'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period', 'build_period'))
    for name in ('effects_per_size_recurring', 'effects_fixed_recurring'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period'))
    # Objective weight x period weight, folded into one parameter. Both are
    # per (effect, period), so the fold is a join and the defaults are what a
    # missing row means: no override, then no global weight, then 1.
    global_weights = (
        pl.DataFrame({'period': list(range(len(dims.period_weights))), 'global_weight': dims.period_weights.values})
        if dims.period_weights is not None
        else pl.DataFrame({'period': [], 'global_weight': []}, schema={'period': pl.Int64, 'global_weight': pl.Float64})
    )
    period_axis = list(range(len(dims.period.values) if dims.period is not None else 1))
    grid = (
        pl.DataFrame({'effect': effect_ids}, schema={'effect': pl.String})
        .join(pl.DataFrame({'period': period_axis}, schema={'period': pl.Int64}), how='cross')
        .join(eds.period_weights, on=['effect', 'period'], how='left')
        .join(global_weights, on='period', how='left')
        .with_columns(pl.col('weight').fill_null(pl.col('global_weight')).fill_null(1.0).alias('weight'))
    )
    objective_by_effect = pl.DataFrame(
        {'effect': effect_ids, 'objective': [float(objective.get(e, 0.0)) for e in effect_ids]},
        schema={'effect': pl.String, 'objective': pl.Float64},
    )
    sources['objective_weight'] = (
        grid.join(objective_by_effect, on='effect')
        .with_columns((pl.col('objective') * pl.col('weight')).alias('value'))
        .filter(pl.col('value') != 0)
        .select(['effect', 'period', 'value'])
    )
    # Weights for the across-period sum: per-effect override, else global, else 1.
    sources['period_weight'] = grid.select(['effect', 'period', pl.col('weight').alias('value')])

    # --- effect limits ----------------------------------------------------
    for key, frame, axes in (
        ('periodic_min', eds.periodic, ['effect', 'period']),
        ('periodic_max', eds.periodic, ['effect', 'period']),
        ('total_min', eds.totals, ['effect']),
        ('total_max', eds.totals, ['effect']),
    ):
        sources[key] = frame.filter(pl.col(key).is_not_null()).select([*axes, pl.col(key).alias('value')])

    # --- temporal boundary mask ------------------------------------------
    sources['is_first'] = pd.DataFrame({'time': ordinals, 'value': [i == 0 for i in ordinals]})
    sources['time_weight'] = pd.DataFrame({'time': ordinals, 'value': dims.weights.values})

    # --- flow aggregates ------------------------------------------------
    # `size` here is the static one; a sized flow's is a variable, so its bound
    # travels as a coefficient instead of a product and the program multiplies.
    weight = dims.dt * dims.weights
    sources['flow_hour_weight'] = pd.DataFrame({'time': ordinals, 'value': weight.values})
    total_duration = float(weight.sum('time'))
    sized = set(sizing_ids) | set(invest_ids)
    static = xr.DataArray([f not in sized for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']})
    for name, arr in (('flow_hours_min', fds.flow_hours_min), ('flow_hours_max', fds.flow_hours_max)):
        sources[name] = tidy(arr.where(arr.notnull()), drop_zero=False) if arr is not None else _empty(name, 'flow')
    for kind, arr in (('min', fds.load_factor_min), ('max', fds.load_factor_max)):
        if arr is None:
            sources[f'load_factor_{kind}_bound'] = _empty(f'load_factor_{kind}_bound', 'flow')
            sources[f'load_factor_{kind}_coeff'] = _empty(f'load_factor_{kind}_coeff', 'flow')
            continue
        live = arr.notnull()
        sources[f'load_factor_{kind}_bound'] = tidy(
            (arr * fds.size * total_duration).where(live & static), drop_zero=False
        )
        sources[f'load_factor_{kind}_coeff'] = tidy((arr * total_duration).where(live & ~static), drop_zero=False)

    # --- piecewise conversion ------------------------------------------
    # The curve tables are already the shape the program wants: a link is a
    # row on `flow`, so nothing has to be reshaped into link slots.
    linear_convs = data.converters.ids if data.converters is not None else []
    bp_width = 0
    pw_status_of: dict[str, str | None] = {}
    pw = data.piecewise
    if pw is not None:
        pw_convs = pw.converter_ids()
        links, curves = pw.links, pw.curves
        # A link is a (converter, flow, bound) — the breakpoints it passes
        # through are its rows, so the identity is what remains after dropping
        # the axes a curve varies along.
        identity = links.select(['converter', 'flow', 'bound']).unique(maintain_order=True)
        pair_flow = identity['flow'].to_list()

        sources['pw_bp_value'] = _with_time_ordinals(links.filter(pl.col('value') != 0), dims).select(
            ['flow', 'bp', 'time', 'value']
        )

        # Which breakpoints a curve has. Curves of different width share one
        # `bp` axis, so the mask is what stops a weight existing past the end
        # of a narrower one.
        present = links.select(['converter', 'bp']).unique(maintain_order=True).sort(['converter', 'bp'])
        bp_width = int(present['bp'].max() or 0) + 1 if len(present) else 0  # type: ignore[arg-type]
        sources['pw_bp_present'] = present.with_columns(pl.lit(True).alias('value'))
        # A segment starts at every present breakpoint but the last.
        last = present.group_by('converter').agg(pl.col('bp').max().alias('last'))
        sources['pw_seg_present'] = (
            present.join(last, on='converter')
            .filter(pl.col('bp') < pl.col('last'))
            .select(['converter', 'bp', pl.lit(True).alias('value')])
        )

        gated = curves.filter(pl.col('has_status'))['converter'].unique(maintain_order=True).to_list()
        sources['has_piecewise'] = _flags('has_piecewise', 'converter', pw_convs)
        sources['pw_gated'] = _flags('pw_gated', 'converter', gated)
        sources['is_piecewise'] = _flags('is_piecewise', 'flow', pair_flow)
        for name, sign in (('pw_equal', '=='), ('pw_upper', '<='), ('pw_lower', '>=')):
            sources[name] = _flags(name, 'flow', identity.filter(pl.col('bound') == sign)['flow'].to_list())

        # Availability scales the envelope of the reference link — a curve's
        # first, which is what the eager lane bounds too.
        reference = identity.group_by('converter', maintain_order=True).first()
        sources['pw_ref'] = reference.select(['flow', pl.lit(1.0).alias('value')])
        widest = (
            links.join(reference.select(['converter', 'flow']), on=['converter', 'flow'])
            .group_by(['converter', 'time'])
            .agg(pl.col('value').max().alias('widest'))
        )
        sources['pw_avail_bound'] = _with_time_ordinals(
            curves.join(widest, on=['converter', 'time']).with_columns(
                (pl.col('availability') * pl.col('widest')).alias('value')
            ),
            dims,
        ).select(['converter', 'time', 'value'])

        # A gated curve's Status is keyed by the converter's own id, so the
        # lookup maps it to itself — mapping it to None reads as 'no Status'
        # and leaves the curve ungated.
        pw_status_of = {c: c for c in gated}
        of = dict(zip(identity['flow'], identity['converter'], strict=True))
        flow_index['converter_of'] = [of.get(f) or c for f, c in zip(flow_ids, flow_index['converter_of'], strict=True)]
        converter_ids = linear_convs + [c for c in pw_convs if c not in set(linear_convs)]
    else:
        for name, dcols in (
            ('has_piecewise', ('converter',)),
            ('pw_gated', ('converter',)),
            ('is_piecewise', ('flow',)),
            ('pw_equal', ('flow',)),
            ('pw_upper', ('flow',)),
            ('pw_lower', ('flow',)),
            ('pw_ref', ('flow',)),
            ('pw_bp_present', ('converter', 'bp')),
            ('pw_seg_present', ('converter', 'bp')),
            ('pw_bp_value', ('flow', 'bp', 'time')),
            ('pw_avail_bound', ('converter', 'time')),
        ):
            sources[name] = _empty(name, *dcols)
        converter_ids = linear_convs

    # Stated as a schema rather than inferred. Every column here is a label
    # or a lookup into one, and a system with no storages leaves
    # `charge_storage` all-null — which infers as a null column and fails the
    # join against the storage labels. The schema says what each column is
    # regardless of what this particular system happens to fill in.
    sources['flow'] = pl.DataFrame(
        {c: flow_index[c].tolist() for c in flow_index.columns},
        schema=dict.fromkeys(flow_index.columns, pl.String),
    )

    # Single-period models supply a length-1 period so one program serves both.
    period_labels = list(dims.period.values) if dims.period is not None else [0]
    period_ord = {v: i for i, v in enumerate(period_labels)}
    p_ordinals = list(range(len(period_labels)))

    def onto_periods(df: Any, axis: str) -> Any:
        """Put *df* on the program's period axis, whichever library holds it.

        A table already naming periods has its labels mapped to ordinals; one
        that does not is the same in every period, so it is crossed onto all
        of them. Both shapes arrive in pandas and in polars, and will keep
        doing so until the last container is converted.
        """
        if isinstance(df, pl.DataFrame):
            if axis in df.columns:
                return df.with_columns(pl.col(axis).replace_strict(period_ord, return_dtype=pl.Int64))
            return df.join(pl.DataFrame({axis: p_ordinals}, schema={axis: pl.Int64}), how='cross')
        if axis in df.columns:
            return df.assign(**{axis: [period_ord[v] for v in df[axis]]})
        return df.merge(pd.DataFrame({axis: p_ordinals}), how='cross')

    for name in PERIOD_PARAMS:
        if (df := sources.get(name)) is not None:
            sources[name] = onto_periods(df, 'period')

    for name in BUILD_PERIOD_PARAMS:
        df = sources.get(name)
        if df is not None and 'build_period' in df.columns:
            sources[name] = onto_periods(df, 'build_period')

    def _index_frame(dim: str, values: list[str], lookups: dict[str, dict[str, str | None]]) -> pl.DataFrame:
        """A dimension's index table plus the lookup columns declared over it.

        Stated as a schema for the same reason the flow index is: a lookup no
        system happens to fill is all-null, and a null column joins against
        nothing.
        """
        cols: dict[str, list[str | None]] = {dim: list(values)}
        cols.update({name: [m.get(v) for v in values] for name, m in lookups.items()})
        return pl.DataFrame(cols, schema=dict.fromkeys(cols, pl.String))

    def labels(values: Any) -> np.ndarray:
        """A string dimension's labels, carrying their type even when empty.

        A system with no storages still declares the dimension, and its
        lookups are string maps into it. A bare ``[]`` has no dtype to infer,
        binds as a null label space and fails the join against those columns —
        so the labels travel as a string array, which says what the space
        would have held.
        """
        return np.array(list(values), dtype=str)

    def axis(dim: str, values: Any) -> pl.DataFrame:
        """A dimension's labels as a one-column frame.

        A frame rather than a sequence because a strategy slices its sources:
        `solve_over` filters every table that carries the axis it cuts on, and
        a bare list is not a table it can cut. Costing nothing to always do,
        so the shipped sources are sliceable whether or not this build is.
        """
        return pl.DataFrame({dim: list(values)}, schema={dim: pl.Int64})

    coords: dict[str, Any] = {
        'time': axis('time', ordinals),
        'period': axis('period', p_ordinals),
        'build_period': axis('build_period', p_ordinals),
        'carrier': labels(data.carriers.ids),
        # Both kinds: a converter states linear equations, a piecewise curve,
        # or one of each. The axis is the union, or a curve's own converter
        # would not be a coordinate of the dimension its rows are keyed on.
        'converter': _index_frame('converter', converter_ids, {'pw_status_of': pw_status_of}),
        'eq_idx': axis('eq_idx', range(data.converters.width) if data.converters is not None else []),
        'storage': labels(storage_ids),
        'effect': labels(effect_ids),
        'status_entity': labels(entity_ids),
        # numpy, not a list: with no piecewise converter the width is 0 and a
        # bare `[]` has no integer type for the join to match.
        'bp': axis('bp', range(bp_width)),
    }
    _stamp_empty_dtypes(sources)
    return sources, coords
