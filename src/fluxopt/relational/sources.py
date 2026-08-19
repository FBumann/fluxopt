"""Bind a :class:`~fluxopt.model_data.ModelData` to the relational math program.

This is the data half of fluxopt's second backend. Both backends consume the
same ``ModelData``: :class:`~fluxopt.model.FlowSystemModel` turns it into a
linopy model, while this module emits the parameter tables that
:data:`MATH_PROGRAM` declares, for a memory-bounded streaming build.

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
from fluxopt.contributions import _apply_leontief, _leontief

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: The YAML program holding fluxopt's math. Shipped as package data.
MATH_PROGRAM = Path(__file__).with_name('core.yaml')

#: Parameters the YAML declares with a `period` axis. Anything emitted from an
#: array that has no period dim is cross-joined onto every period.
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
        'effects_per_running_hour',
        'effects_per_startup',
        'effects_per_size',
        'effects_fixed',
        'effects_per_capacity',
        'effects_fixed_capacity',
        'effects_fixed_mandatory',
        'objective_weight',
        'periodic_min',
        'periodic_max',
        'period_weight',
        'rate_max_at_size_max',
        'rate_min_at_size_max',
        'size_upper',
        'lifetime_window',
        'prior_capacity_active',
        'effects_per_size_recurring',
        'effects_fixed_recurring',
        'effects_per_size_at_build',
        'effects_fixed_at_build',
    }
)

#: Parameters carrying the `build_period` axis; its labels map to ordinals too.
BUILD_PERIOD_PARAMS = frozenset({'lifetime_window', 'effects_per_size_at_build', 'effects_fixed_at_build'})


def _size_upper(data: ModelData, fid: str) -> float:
    """Static upper bound on a flow's size — fixed value or sizing max."""
    fds = data.flows
    val = float(fds.size.sel(flow=fid).values)
    if not np.isnan(val):
        return val
    if fds.sizing is not None:
        zdim = fds.sizing.min.dims[0]
        ids = [str(v) for v in fds.sizing.min.coords[zdim].values]
        if fid in ids:
            return float(fds.sizing.max.sel({zdim: fid}).values)
    return 0.0


#: Index columns the program declares with an integer dtype; every other index
#: column is a string label.
_INT_DIMS = frozenset({'time', 'period', 'build_period', 'eq_idx'})

#: Parameters the program declares ``dtype: bool``.
_BOOL_PARAMS = frozenset(
    {
        'is_first',
        'is_last',
        'conversion_active',
        'is_cyclic',
        'has_status',
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
    """The ModelData uses a feature the relational program does not express yet.

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


def _reject_unsupported(data: ModelData) -> None:
    fds = data.flows
    for name, obj in (
        ('component status', fds.cstatus),
        ('piecewise', data.piecewise),
    ):
        if obj is not None:
            raise UnsupportedFeatureError(f'{name} is not supported by the relational backend yet')
    if fds.invest is not None and data.dims.period is None:
        raise UnsupportedFeatureError('investment requires multi-period optimization (periods must be specified)')
    if fds.sizing is not None and fds.status is not None:
        sz = {str(v) for v in fds.sizing.min.coords[fds.sizing.min.dims[0]].values}
        stt = {str(v) for v in fds.status.uptime_min.coords[fds.status.uptime_min.dims[0]].values}
        both = sz & stt
        profile = {f for f in both if str(fds.bound_type.sel(flow=f).values) == BoundType.PROFILE}
        if profile:
            # fluxopt rejects this combination too — no formulation exists.
            raise UnsupportedFeatureError(f'fixed profile with status+sizing has no formulation: {sorted(profile)}')


def build_sources(data: ModelData, objective: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Emit the parameter tables and coordinates for :data:`MATH_PROGRAM`.

    Args:
        data: The model data to bind. Both backends read the same object.
        objective: Effect ids mapped to their objective weight, as
            :class:`~fluxopt.model.FlowSystemModel` takes it.

    Returns:
        ``(sources, coords)`` ready to pass to ``lpspec.solve``.

    Raises:
        UnsupportedFeatureError: If *data* uses a feature the program does
            not express yet, rather than dropping it silently.
    """
    _reject_unsupported(data)
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
    zdim = ''
    sdim = ''
    has_sizing = xr.zeros_like(fds.size, dtype=bool)
    # Lump-domain accumulators, filled by the flow- and storage-sizing blocks.
    # (parameter name, entity dim, coefficients) — the entity dim is carried so
    # an absent term still emits a correctly keyed empty table.
    lump_terms: list[tuple[str, str, xr.DataArray | None]] = []
    lump_consts: list[tuple[str, xr.DataArray]] = []

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
        zdim = sz.min.dims[0]
        sizing_ids = [str(v) for v in sz.min.coords[zdim].values]
        has_sizing = xr.DataArray(
            [f in set(sizing_ids) for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']}
        )
        lb = xr.where(has_sizing, 0.0, lb)
        ub = xr.where(has_sizing, np.inf, ub)

    inv = fds.invest
    invest_ids: list[str] = []
    idim = ''
    has_invest = xr.zeros_like(fds.size, dtype=bool)
    if inv is not None:
        idim = inv.min.dims[0]
        invest_ids = [str(v) for v in inv.min.coords[idim].values]
        has_invest = xr.DataArray(
            [f in set(invest_ids) for f in flow_ids], dims=['flow'], coords={'flow': fds.size.coords['flow']}
        )
        lb = xr.where(has_invest, 0.0, lb)
        ub = xr.where(has_invest, np.inf, ub)

    st = fds.status
    if st is not None:
        sdim = st.uptime_min.dims[0]
        status_ids = [str(v) for v in st.uptime_min.coords[sdim].values]
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
    coeff = data.carriers.flow_coeff  # (carrier, flow), NaN = unconnected
    live = coeff.notnull() & (coeff != 0)
    car_idx = live.values.argmax(axis=0)
    carriers = coeff.coords['carrier'].values
    flow_index = pd.DataFrame({'flow': flow_ids, 'carrier_of': [str(carriers[i]) for i in car_idx]})
    sources['carrier_sign'] = pd.DataFrame(
        {'flow': flow_ids, 'value': [float(coeff.values[i, j]) for j, i in enumerate(car_idx)]},
    )

    # --- converters -------------------------------------------------------
    if data.converters is not None:
        cds = data.converters
        pair_flow = [str(v) for v in cds.pair_flow.values]
        pair_conv = [str(v) for v in cds.pair_converter.values]
        conv_of = dict(zip(pair_flow, pair_conv, strict=True))
        flow_index['converter_of'] = [conv_of.get(f) for f in flow_ids]
        pc = cds.pair_coeff.assign_coords(pair=pair_flow).rename({'pair': 'flow'})
        sources['conversion_factor'] = tidy(pc, drop_zero=True)
        sources['conversion_active'] = _tidy(cds.eq_mask.astype(float), drop_zero=True).assign(value=True)
    else:
        flow_index['converter_of'] = None
        sources['conversion_factor'] = _empty('conversion_factor', 'flow', 'eq_idx', 'time')
        sources['conversion_active'] = _empty('conversion_active', 'converter', 'eq_idx')

    # --- storage ----------------------------------------------------------
    storage_ids: list[str] = []
    if data.storages is not None:
        sds = data.storages
        storage_ids = [str(s) for s in sds.capacity.coords['storage'].values]
        charge = [str(v) for v in sds.charge_flow.values]
        discharge = [str(v) for v in sds.discharge_flow.values]
        chg_of = dict(zip(charge, storage_ids, strict=True))
        dis_of = dict(zip(discharge, storage_ids, strict=True))
        flow_index['charge_storage'] = [chg_of.get(f) for f in flow_ids]
        flow_index['discharge_storage'] = [dis_of.get(f) for f in flow_ids]

        def on_flow(da: xr.DataArray, fids: list[str]) -> pd.DataFrame:
            return tidy(da.assign_coords(storage=fids).rename({'storage': 'flow'}), drop_zero=True)

        sources['charge_gain'] = on_flow(sds.eta_c * dims.dt, charge)
        sources['discharge_draw'] = on_flow(dims.dt / sds.eta_d, discharge)
        sources['retention'] = tidy((1 - sds.loss) ** dims.dt, drop_zero=False)
        cap = sds.capacity
        sources['level_min'] = tidy((sds.rel_level_lb * cap).fillna(0.0), drop_zero=False)
        sources['level_max'] = tidy((sds.rel_level_ub * cap).fillna(np.inf), drop_zero=False)
        csz = sds.sizing
        if csz is not None:
            cdim = csz.min.dims[0]
            cap_ids = [str(v) for v in csz.min.coords[cdim].values]
            cmand = csz.mandatory.values.astype(bool)
            copt = [s for s, m in zip(cap_ids, cmand, strict=True) if not m]
            cmand_ids = [s for s, m in zip(cap_ids, cmand, strict=True) if m]
            sources['has_capacity_sizing'] = _flags('has_capacity_sizing', 'storage', cap_ids)
            sources['capacity_optional'] = _flags('capacity_optional', 'storage', copt)
            sources['capacity_min'] = pd.DataFrame({'storage': cap_ids, 'value': csz.min.values})
            sources['capacity_max'] = pd.DataFrame({'storage': cap_ids, 'value': csz.max.values})
            sources['relative_level_min'] = tidy(sds.rel_level_lb.sel(storage=cap_ids), drop_zero=True)
            sources['relative_level_max'] = tidy(sds.rel_level_ub.sel(storage=cap_ids), drop_zero=True)
            cren = {cdim: 'storage'}
            lump_terms += [
                ('effects_per_capacity', 'storage', csz.effects_per_size.rename(cren)),
                (
                    'effects_fixed_capacity',
                    'storage',
                    csz.effects_fixed.sel({cdim: copt}).rename(cren) if copt else None,
                ),
            ]
            if cmand_ids:
                cap_const = csz.effects_fixed.sel({cdim: cmand_ids}).rename(cren)
                lump_consts.append(('storage', cap_const))
        sources['is_cyclic'] = pd.DataFrame(
            {'storage': storage_ids, 'value': sds.cyclic.values.astype(bool)},
        )
        sources['prior_level'] = pd.DataFrame({'storage': storage_ids, 'value': sds.prior_level.fillna(0.0).values})
        for key, arr in (('final_level_min', sds.final_level_min), ('final_level_max', sds.final_level_max)):
            if arr is None:
                sources[key] = _empty(key, 'storage')
                continue
            live = arr.notnull()
            sources[key] = _tidy(arr.where(live), drop_zero=False)
        prevent = (
            sds.prevent_simultaneous.values.astype(bool)
            if sds.prevent_simultaneous is not None
            else np.zeros(len(storage_ids), dtype=bool)
        )
        prev_ids = [s for s, p in zip(storage_ids, prevent, strict=True) if p]
        sources['prevent_simultaneous'] = _flags('prevent_simultaneous', 'storage', prev_ids)
        sources['charge_size_bound'] = pd.DataFrame(
            {'storage': storage_ids, 'value': [_size_upper(data, f) for f in charge]},
        )
        sources['discharge_size_bound'] = pd.DataFrame(
            {'storage': storage_ids, 'value': [_size_upper(data, f) for f in discharge]},
        )
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
    ec_extra: list[tuple[str, xr.DataArray]] = []
    if st is not None:
        ren = {sdim: 'flow'}
        sources['has_status'] = _flags('has_status', 'flow', status_ids)
        sources['is_bounded'] = pd.DataFrame(
            {'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values},
        )
        sources['is_profile'] = pd.DataFrame(
            {'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values},
        )
        sel = {'flow': status_ids}
        sources['rate_min_when_on'] = tidy((size * fds.rel_lb).sel(sel), drop_zero=True)
        sources['rate_max_when_on'] = tidy((size * fds.rel_ub).sel(sel), drop_zero=True)
        sources['rate_fixed_when_on'] = tidy((size * fds.fixed_profile).sel(sel), drop_zero=True)

        up_min = st.uptime_min.rename(ren)
        up_max = st.uptime_max.rename(ren)
        horizon = float(dims.dt.sum())
        has_up = (up_min.notnull() | up_max.notnull()).values
        up_ids = [f for f, h in zip(status_ids, has_up, strict=True) if h]
        sources['has_uptime'] = _flags('has_uptime', 'flow', up_ids)
        min_ids = [f for f, h in zip(status_ids, up_min.notnull().values, strict=True) if h]
        sources['uptime_min'] = pd.DataFrame(
            {'flow': min_ids, 'value': up_min.sel(flow=min_ids).values if min_ids else []},
        )
        prev_up = st.previous_uptime.rename(ren) if st.previous_uptime is not None else None
        prev_dn = st.previous_downtime.rename(ren) if st.previous_downtime is not None else None
        envelope = fds.rel_lb.sel(sel)

        def _per_flow(arr: xr.DataArray | float) -> np.ndarray:
            return np.broadcast_to(np.asarray(arr, dtype=float), (len(status_ids),)).copy()

        uptime_big_m = horizon + (prev_up.fillna(0.0) if prev_up is not None else 0.0)
        mega_dn = horizon + (prev_dn.fillna(0.0) if prev_dn is not None else 0.0)
        sources['uptime_big_m'] = pd.DataFrame({'flow': status_ids, 'value': _per_flow(uptime_big_m)})
        sources['downtime_big_m'] = pd.DataFrame({'flow': status_ids, 'value': _per_flow(mega_dn)})
        sources['uptime_upper'] = tidy(
            up_max.fillna(uptime_big_m).broadcast_like(envelope).sel(flow=up_ids), drop_zero=False
        )

        # --- downtime: the same tracking with the state inverted -------------
        dn_min, dn_max = st.downtime_min.rename(ren), st.downtime_max.rename(ren)
        dn_ids = [f for f, h in zip(status_ids, (dn_min.notnull() | dn_max.notnull()).values, strict=True) if h]
        dn_min_ids = [f for f, h in zip(status_ids, dn_min.notnull().values, strict=True) if h]
        sources['has_downtime'] = _flags('has_downtime', 'flow', dn_ids)
        sources['downtime_min'] = pd.DataFrame(
            {'flow': dn_min_ids, 'value': dn_min.sel(flow=dn_min_ids).values if dn_min_ids else []},
        )
        sources['downtime_upper'] = tidy(
            dn_max.fillna(mega_dn).broadcast_like(envelope).sel(flow=dn_ids), drop_zero=False
        )

        # --- initial state and pre-horizon carry-over ------------------------
        init = st.initial.rename(ren)
        init_ids = [f for f, h in zip(status_ids, init.notnull().values, strict=True) if h]
        sources['initial_status'] = pd.DataFrame(
            {'flow': init_ids, 'value': init.sel(flow=init_ids).values if init_ids else []},
        )
        # Names are spelled out, not assembled, so a rename of the program is
        # greppable from here.
        for value_key, forced_key, prev, lo in (
            ('previous_uptime', 'forced_on_at_start', prev_up, up_min),
            ('previous_downtime', 'forced_off_at_start', prev_dn, dn_min),
        ):
            if prev is None:
                sources[value_key] = _empty(value_key, 'flow')
                sources[forced_key] = _empty(forced_key, 'flow')
                continue
            pids = [f for f, h in zip(status_ids, prev.notnull().values, strict=True) if h]
            sources[value_key] = (
                pd.DataFrame({'flow': pids, 'value': prev.sel(flow=pids).values}) if pids else _empty(value_key, 'flow')
            )
            forced = ((prev > 0) & lo.notnull() & (prev < lo)).values
            sources[forced_key] = _flags(forced_key, 'flow', [f for f, h in zip(status_ids, forced, strict=True) if h])

        ec_extra = [
            ('effects_per_running_hour', (st.effects_running.rename(ren) * dims.dt * dims.weights)),
            ('effects_per_startup', (st.effects_startup.rename(ren) * dims.weights)),
        ]
    else:
        for n in ('has_status', 'has_uptime'):
            sources[n] = _empty(n, 'flow')
        sources['is_bounded'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values})
        sources['is_profile'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values})
        for n in (
            'uptime_min',
            'uptime_big_m',
            'downtime_big_m',
            'downtime_min',
            'has_downtime',
            'initial_status',
            'previous_uptime',
            'previous_downtime',
            'forced_on_at_start',
            'forced_off_at_start',
        ):
            sources[n] = _empty(n, 'flow')
        for n in ('rate_min_when_on', 'rate_max_when_on', 'rate_fixed_when_on', 'uptime_upper', 'downtime_upper'):
            sources[n] = _empty(n, 'flow', 'time')
        ec_extra = []

    sources['dt'] = pd.DataFrame({'time': ordinals, 'value': dims.dt.values})
    sources['is_last'] = pd.DataFrame({'time': ordinals, 'value': [i == len(ordinals) - 1 for i in ordinals]})

    # --- sizing -----------------------------------------------------------
    if sz is not None:
        zren = {zdim: 'flow'}
        mandatory = sz.mandatory.values.astype(bool)
        opt_ids = [f for f, m in zip(sizing_ids, mandatory, strict=True) if not m]
        mand_ids = [f for f, m in zip(sizing_ids, mandatory, strict=True) if m]
        sources['has_sizing'] = _flags('has_sizing', 'flow', sizing_ids)
        sources['size_optional'] = _flags('size_optional', 'flow', opt_ids)
        sources['size_min'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.min.values})
        sources['size_max'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.max.values})
        zsel = {'flow': sizing_ids}
        smax = xr.DataArray(sz.max.values, dims=['flow'], coords={'flow': sizing_ids})
        sources['rate_max_at_size_max'] = tidy((fds.rel_ub.sel(zsel) * smax), drop_zero=True)
        sources['rate_min_at_size_max'] = tidy((fds.rel_lb.sel(zsel) * smax), drop_zero=True)
        lump_terms += [
            ('effects_per_size', 'flow', sz.effects_per_size.rename(zren)),
            ('effects_fixed', 'flow', sz.effects_fixed.sel({zdim: opt_ids}).rename(zren) if opt_ids else None),
        ]
        if mand_ids:
            lump_consts.append(('flow', sz.effects_fixed.sel({zdim: mand_ids}).rename(zren)))
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
        iren = {idim: 'flow'}
        period_labels_inv: list[Any] = list(dims.period.values) if dims.period is not None else []
        n_p = len(period_labels_inv)
        lifetime = inv.lifetime.values
        prior = inv.prior_size.values
        window = np.zeros((len(invest_ids), n_p, n_p))
        prior_active = np.zeros((len(invest_ids), n_p))
        for f_idx in range(len(invest_ids)):
            lt = lifetime[f_idx]
            lt_int = None if np.isnan(lt) else int(lt)
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
        sources['invest_mandatory'] = pd.DataFrame({'flow': invest_ids, 'value': inv.mandatory.values.astype(bool)})
        sources['invest_min'] = pd.DataFrame({'flow': invest_ids, 'value': inv.min.values})
        sources['invest_max'] = pd.DataFrame({'flow': invest_ids, 'value': inv.max.values})
        sources['prior_capacity'] = pd.DataFrame({'flow': invest_ids, 'value': prior})
        sources['has_prior_capacity'] = _flags(
            'has_prior_capacity', 'flow', [f for f, ps in zip(invest_ids, prior, strict=True) if ps > 0]
        )
        # Diagonal in (period, build_period): a one-time cost belongs to the
        # period the build happened in, not to every period the unit is alive.
        eye = xr.DataArray(
            np.eye(n_p),
            dims=['period', 'build_period'],
            coords={'period': period_labels_inv, 'build_period': period_labels_inv},
        )
        lump_terms += [
            (
                'effects_per_size_at_build',
                'flow',
                inv.effects_per_size_at_build.rename(iren).rename({'period': 'build_period'}) * eye,
            ),
            (
                'effects_fixed_at_build',
                'flow',
                inv.effects_fixed_at_build.rename(iren).rename({'period': 'build_period'}) * eye,
            ),
            ('effects_per_size_recurring', 'flow', inv.effects_per_size_recurring.rename(iren)),
            ('effects_fixed_recurring', 'flow', inv.effects_fixed_recurring.rename(iren)),
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
        (sizing_ids, sz.max if sz is not None else None),
        (invest_ids, inv.max if inv is not None else None),
    ):
        if maxima is None or not ids:
            continue
        per_flow = xr.DataArray(maxima.values, dims=['flow'], coords={'flow': ids}).reindex(flow=flow_ids)
        size_upper = xr.where(per_flow.notnull(), per_flow, size_upper)
    sources['size_upper'] = _tidy(size_upper, drop_zero=False)

    # --- effects: the sparse one -----------------------------------------
    eds = data.effects
    effect_ids = [str(e) for e in eds.total_min.coords['effect'].values]
    ec = fds.effect_coeff * dims.dt * dims.weights
    if eds.cf_temporal is not None:
        ec = _apply_leontief(_leontief(eds.cf_temporal), ec)
    sources['effects_per_flow_hour'] = tidy(ec, drop_zero=True)
    leo = _leontief(eds.cf_temporal) if eds.cf_temporal is not None else None
    for name, arr in ec_extra:
        sources[name] = tidy(_apply_leontief(leo, arr) if leo is not None else arr, drop_zero=True)
    for name in ('effects_per_running_hour', 'effects_per_startup'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'time'))

    # Lump domain: effect_lump = (I - cf_lump)^-1 . lump_direct, folded into
    # the coefficients so no self-referential effect_lump variable is needed.
    leo_lump = _leontief(eds.cf_temporal.mean('time')) if eds.cf_temporal is not None else None

    def fold(arr: xr.DataArray) -> xr.DataArray:
        """Apply the lump-domain Leontief inverse, if there are cross-effects."""
        return arr if leo_lump is None else _apply_leontief(leo_lump, arr)

    for name, entity_dim, arr in lump_terms:
        sources[name] = tidy(fold(arr), drop_zero=True) if arr is not None else _empty(name, entity_dim, 'effect')
    for name in ('effects_per_size', 'effects_fixed'):
        sources.setdefault(name, _empty(name, 'flow', 'effect'))
    for name in ('effects_per_capacity', 'effects_fixed_capacity'):
        sources.setdefault(name, _empty(name, 'storage', 'effect'))
    for name in ('effects_per_size_at_build', 'effects_fixed_at_build'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period', 'build_period'))
    for name in ('effects_per_size_recurring', 'effects_fixed_recurring'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period'))
    total_const: xr.DataArray | None = None
    for edim, arr in lump_consts:
        part = fold(arr).sum(edim)
        total_const = part if total_const is None else total_const + part
    # Dense over `effect`, zeros included. This is the constant side of
    # `effect_accounting`, where an absent row is not an absence: it is read as
    # a zero that binds the row it sits in. Every effect therefore states its
    # constant, even when that constant is nothing.
    zero_const = xr.zeros_like(xr.DataArray(np.zeros(len(effect_ids)), dims=['effect'], coords={'effect': effect_ids}))
    sources['effects_fixed_mandatory'] = _tidy(zero_const if total_const is None else total_const, drop_zero=False)
    # Objective weight x period weight, folded into one parameter.
    obj_w = xr.DataArray(
        [float(objective.get(e, 0.0)) for e in effect_ids],
        dims=['effect'],
        coords={'effect': effect_ids},
    )
    pw = eds.period_weights
    if pw is not None and dims.period_weights is not None:
        pw = pw.fillna(dims.period_weights)
    elif dims.period_weights is not None:
        pw = dims.period_weights
    sources['objective_weight'] = _tidy(obj_w if pw is None else obj_w * pw, drop_zero=True)

    # --- effect limits ----------------------------------------------------
    for key, arr in (
        ('periodic_min', eds.periodic_min),
        ('periodic_max', eds.periodic_max),
        ('total_min', eds.total_min),
        ('total_max', eds.total_max),
    ):
        live = arr.notnull()
        sources[key] = _tidy(arr.where(live), drop_zero=False)
    # Weights for the across-period sum: per-effect override, else global, else 1.
    ones = xr.ones_like(obj_w)
    tw = ones if pw is None else ones * pw
    sources['period_weight'] = _tidy(tw, drop_zero=True)

    # --- temporal boundary mask ------------------------------------------
    sources['is_first'] = pd.DataFrame({'time': ordinals, 'value': [i == 0 for i in ordinals]})

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
    for name in PERIOD_PARAMS:
        df = sources.get(name)
        if df is None:
            continue
        if 'period' in df.columns:
            sources[name] = df.assign(period=[period_ord[v] for v in df['period']])
        else:
            sources[name] = df.merge(pd.DataFrame({'period': p_ordinals}), how='cross')

    empty: list[str] = []
    for name in BUILD_PERIOD_PARAMS:
        df = sources.get(name)
        if df is not None and 'build_period' in df.columns:
            sources[name] = df.assign(build_period=[period_ord[v] for v in df['build_period']])

    def labels(values: Any) -> np.ndarray:
        """A string dimension's labels, carrying their type even when empty.

        A system with no storages still declares the dimension, and its
        lookups are string maps into it. A bare ``[]`` has no dtype to infer,
        binds as a null label space and fails the join against those columns —
        so the labels travel as a string array, which says what the space
        would have held.
        """
        return np.array(list(values), dtype=str)

    coords: dict[str, Any] = {
        'time': ordinals,
        'period': p_ordinals,
        'build_period': p_ordinals,
        'carrier': labels([str(c) for c in carriers]),
        'converter': labels(
            [str(c) for c in data.converters.eq_mask.coords['converter'].values] if data.converters is not None else []
        ),
        'eq_idx': list(range(data.converters.eq_mask.sizes['eq_idx'])) if data.converters is not None else empty,
        'storage': labels(storage_ids),
        'effect': labels(effect_ids),
    }
    _stamp_empty_dtypes(sources)
    return sources, coords
