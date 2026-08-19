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

from fluxopt.contract import BoundType, Dim
from fluxopt.leontief import apply_leontief, leontief
from fluxopt.validation import reject_varying_contribution_into_lump

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: The YAML program holding fluxopt's math. Shipped as package data.
PROGRAM = Path(__file__).with_name('program.yaml')

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
        'objective_weight',
        'prior_level',
        'periodic_min',
        'periodic_max',
        'period_weight',
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


def _reject_unsupported(data: ModelData) -> None:
    fds = data.flows
    if data.piecewise is not None:
        bad = sorted({str(m) for m in data.piecewise.method.values} & {'lp'})
        if bad:
            raise UnsupportedFeatureError(
                "piecewise method 'lp' is linopy's tangent-line relaxation, which this lane has no "
                'formulation for — the adjacency formulation it does have is exact, so it would '
                'answer a different question. Use the default method, or lpspec #695.'
            )
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
    zdim = ''
    sdim = ''
    has_sizing = xr.zeros_like(fds.size, dtype=bool)
    # Lump-domain accumulators, filled by the flow- and storage-sizing blocks.
    # (parameter name, entity dim, coefficients) — the entity dim is carried so
    # an absent term still emits a correctly keyed empty table.
    lump_terms: list[tuple[str, str, xr.DataArray | None]] = []

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
    carriers_of = [str(c) for c in data.carriers.carrier_of.values]
    flow_index = pd.DataFrame({'flow': flow_ids, 'carrier_of': carriers_of})
    sources['carrier_sign'] = pd.DataFrame({'flow': flow_ids, 'value': data.carriers.sign.values})

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
                    csz.effects_fixed.rename(cren),
                ),
            ]
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
    # Two containers, one axis. `flows.status` is a flow's own on/off decision;
    # `flows.cstatus` is a component's, governing several flows at once. They
    # carry the same fields and obey the same math, so they are bound to one
    # `status_entity` dimension and the program states the family once. Which
    # rows read which binary is `status_of`, and nothing else distinguishes
    # them.
    ec_extra: list[tuple[str, xr.DataArray]] = []
    cst = fds.cstatus
    blocks: list[tuple[Any, list[str]]] = []
    #: flow id -> the entity whose binary gates it. A self-status flow maps to
    #: itself; a governed flow to its component; an ungated flow to nothing.
    status_of: dict[str, str] = {}
    if st is not None:
        blocks.append((st, status_ids))
        status_of.update({f: f for f in status_ids})
    if cst is not None:
        cdim = cst.uptime_min.dims[0]
        comp_ids = [str(v) for v in cst.uptime_min.coords[cdim].values]
        blocks.append((cst, comp_ids))
        # A piecewise curve's flows are gated by its convexity row, which
        # already pins every weight to zero when the binary is off. Gating
        # them a second time per flow would be redundant, and wrong for the
        # links the curve only bounds.
        pw_comps = set(data.piecewise.converter_ids()) if data.piecewise is not None else set()
        if cst.governed_flows is not None:
            for row, cid in zip(cst.governed_flows.values, comp_ids, strict=True):
                if cid in pw_comps:
                    continue
                status_of.update({str(f): cid for f in row if str(f)})

    flow_index['status_of'] = [status_of.get(f) for f in flow_ids]
    entity_ids = [e for _, ids in blocks for e in ids]
    gated_ids = [f for f in flow_ids if f in status_of]

    sources['is_gated'] = _flags('is_gated', 'flow', gated_ids)
    sources['is_bounded'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values})
    sources['is_profile'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values})

    if blocks:

        def on_entity(field: str) -> xr.DataArray:
            """One field of every block, on the shared entity axis.

            A field a block leaves unset (``previous_uptime`` on a container
            with no prior rates) still has to occupy its entity's rows, or the
            concatenation would silently shorten the axis — so it arrives as
            NaN, which is what "not set" already means everywhere else here.
            """
            parts: list[xr.DataArray] = []
            for block, ids in blocks:
                arr = getattr(block, field)
                if arr is None:
                    arr = xr.DataArray(np.full(len(ids), np.nan), dims=['status_entity'])
                else:
                    arr = arr.rename({arr.dims[0]: 'status_entity'})
                parts.append(arr.assign_coords(status_entity=ids))
            return xr.concat(parts, dim='status_entity')

        def entity_frame(name: str, arr: xr.DataArray) -> None:
            """Emit *arr* keyed on the entity axis, live rows only."""
            live = arr.notnull()
            ids = [e for e, h in zip(entity_ids, live.values, strict=True) if h]
            sources[name] = (
                pd.DataFrame({'status_entity': ids, 'value': arr.values[live.values]})
                if ids
                else _empty(name, 'status_entity')
            )

        # Envelopes are per gated flow: an governed flow is sized like any
        # other, and `size * rel` is what the binary scales.
        gsel = {'flow': gated_ids}
        sources['rate_min_when_on'] = tidy((size * fds.rel_lb).sel(gsel), drop_zero=True)
        sources['rate_max_when_on'] = tidy((size * fds.rel_ub).sel(gsel), drop_zero=True)
        sources['rate_fixed_when_on'] = tidy((size * fds.fixed_profile).sel(gsel), drop_zero=True)

        up_min, up_max = on_entity('uptime_min'), on_entity('uptime_max')
        dn_min, dn_max = on_entity('downtime_min'), on_entity('downtime_max')
        prev_up, prev_dn = on_entity('previous_uptime'), on_entity('previous_downtime')
        horizon = float(dims.dt.sum())

        up_ids = [e for e, h in zip(entity_ids, (up_min.notnull() | up_max.notnull()).values, strict=True) if h]
        dn_ids = [e for e, h in zip(entity_ids, (dn_min.notnull() | dn_max.notnull()).values, strict=True) if h]
        sources['has_uptime'] = _flags('has_uptime', 'status_entity', up_ids)
        sources['has_downtime'] = _flags('has_downtime', 'status_entity', dn_ids)
        entity_frame('uptime_min', up_min)
        entity_frame('downtime_min', dn_min)
        entity_frame('initial_status', on_entity('initial'))

        uptime_big_m = horizon + prev_up.fillna(0.0)
        downtime_big_m = horizon + prev_dn.fillna(0.0)
        sources['uptime_big_m'] = pd.DataFrame({'status_entity': entity_ids, 'value': uptime_big_m.values})
        sources['downtime_big_m'] = pd.DataFrame({'status_entity': entity_ids, 'value': downtime_big_m.values})

        # The duration variables' own upper bound, over the entity's timeline.
        span = xr.DataArray(
            np.ones((len(entity_ids), len(time_labels))),
            dims=['status_entity', 'time'],
            coords={'status_entity': entity_ids, 'time': list(dims.time.values)},
        )
        sources['uptime_upper'] = tidy((up_max.fillna(uptime_big_m) * span).sel(status_entity=up_ids), drop_zero=False)
        sources['downtime_upper'] = tidy(
            (dn_max.fillna(downtime_big_m) * span).sel(status_entity=dn_ids), drop_zero=False
        )

        # Pre-horizon carry-over. Names are spelled out, not assembled, so a
        # rename of the program is greppable from here.
        for value_key, forced_key, prev, lo in (
            ('previous_uptime', 'forced_on_at_start', prev_up, up_min),
            ('previous_downtime', 'forced_off_at_start', prev_dn, dn_min),
        ):
            entity_frame(value_key, prev)
            forced = ((prev > 0) & lo.notnull() & (prev < lo)).values
            sources[forced_key] = _flags(
                forced_key, 'status_entity', [e for e, h in zip(entity_ids, forced, strict=True) if h]
            )

        ec_extra = [
            ('effects_per_running_hour', on_entity('effects_running') * dims.dt),
            ('effects_per_startup', on_entity('effects_startup')),
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
        zren = {zdim: 'flow'}
        mandatory = sz.mandatory.values.astype(bool)
        opt_ids = [f for f, m in zip(sizing_ids, mandatory, strict=True) if not m]
        sources['has_sizing'] = _flags('has_sizing', 'flow', sizing_ids)
        sources['size_optional'] = _flags('size_optional', 'flow', opt_ids)
        sources['size_min'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.min.values})
        sources['size_max'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.max.values})
        zsel = {'flow': sizing_ids}
        smax = xr.DataArray(sz.max.values, dims=['flow'], coords={'flow': sizing_ids})
        sources['rate_max_at_size_max'] = tidy((fds.rel_ub.sel(zsel) * smax), drop_zero=True)
        # Dense: this one also stands on the constant side of
        # `status_sizing_rate_min`, where a dropped zero is a bound rather
        # than an absent coefficient. A flow whose lower bound is zero is the
        # ordinary case, so dropping it would break exactly the common one.
        sources['rate_min_at_size_max'] = tidy((fds.rel_lb.sel(zsel) * smax), drop_zero=False)
        lump_terms += [
            ('effects_per_size', 'flow', sz.effects_per_size.rename(zren)),
            ('effects_fixed', 'flow', sz.effects_fixed.rename(zren)),
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
    # dt stays: a per-flow-hour rate times a duration is the step's energy.
    # The aggregation weight does not — the program applies it in the sum,
    # so a named contribution reads as the physical per-step quantity.
    ec = fds.effect_coeff * dims.dt
    if eds.cf_temporal is not None:
        ec = apply_leontief(leontief(eds.cf_temporal), ec)
    sources['effects_per_flow_hour'] = tidy(ec, drop_zero=True)
    leo = leontief(eds.cf_temporal) if eds.cf_temporal is not None else None
    for name, arr in ec_extra:
        sources[name] = tidy(apply_leontief(leo, arr) if leo is not None else arr, drop_zero=True)
    for name in ('effects_per_running_hour', 'effects_per_startup'):
        sources.setdefault(name, _empty(name, 'status_entity', 'effect', 'time'))

    # Lump domain: effect_lump = (I - cf_lump)^-1 . lump_direct, folded into
    # the coefficients so no self-referential effect_lump variable is needed.
    leo_lump = leontief(eds.cf_temporal.mean('time')) if eds.cf_temporal is not None else None

    def fold(arr: xr.DataArray) -> xr.DataArray:
        """Apply the lump-domain Leontief inverse, if there are cross-effects."""
        return arr if leo_lump is None else apply_leontief(leo_lump, arr)

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
    # `breakpoints` is already keyed per (converter, flow) pair, which is the
    # shape the program wants: a link is a row on `flow`, so nothing has to be
    # reshaped into link slots.
    linear_convs = (
        [str(c) for c in data.converters.eq_mask.coords['converter'].values] if data.converters is not None else []
    )
    bp_width = 0
    pw_status_of: dict[str, str | None] = {}
    pw = data.piecewise
    if pw is not None:
        pair_flow = [str(v) for v in pw.pair_flow.values]
        pair_conv = [str(v) for v in pw.pair_converter.values]
        pair_bound = [str(v) for v in pw.pair_bound.values]
        pw_convs = pw.converter_ids()

        bpv = pw.breakpoints.rename({Dim.PW_PAIR: 'flow', 'breakpoint': 'bp'}).assign_coords(flow=pair_flow)
        sources['pw_bp_value'] = tidy(bpv, drop_zero=True)

        # Which breakpoints a curve has. Curves of different width share one
        # `bp` axis, so the narrow ones carry NaN past their last point and
        # the mask is what stops a weight existing there.
        first_of = {c: pair_conv.index(c) for c in pw_convs}
        present = np.array([bpv.isel(flow=first_of[c]).notnull().any('time').values for c in pw_convs])
        n_bp = present.shape[1]
        bp_width = n_bp

        def bp_mask(name: str, grid: np.ndarray) -> None:
            rows = np.argwhere(grid)
            sources[name] = (
                pd.DataFrame(
                    {'converter': [pw_convs[i] for i, _ in rows], 'bp': [int(b) for _, b in rows], 'value': True}
                )
                if len(rows)
                else _empty(name, 'converter', 'bp')
            )

        bp_mask('pw_bp_present', present)
        # A segment starts at every present breakpoint but the last.
        seg = present.copy()
        for i, row in enumerate(present):
            live = np.nonzero(row)[0]
            if len(live):
                seg[i, live[-1]] = False
        bp_mask('pw_seg_present', seg)

        gated = [c for c in pw_convs if bool(pw.has_status.sel(pw_converter=c).item())]
        sources['has_piecewise'] = _flags('has_piecewise', 'converter', pw_convs)
        sources['pw_gated'] = _flags('pw_gated', 'converter', gated)
        sources['is_piecewise'] = _flags('is_piecewise', 'flow', pair_flow)
        for name, sign in (('pw_equal', '=='), ('pw_upper', '<='), ('pw_lower', '>=')):
            sources[name] = _flags(name, 'flow', [f for f, b in zip(pair_flow, pair_bound, strict=True) if b == sign])

        # Availability scales the envelope of the reference link — the first
        # of a curve's pairs, which is what the linopy lane bounds too.
        ref_flows = [pair_flow[first_of[c]] for c in pw_convs]
        sources['pw_ref'] = pd.DataFrame({'flow': ref_flows, 'value': 1.0})
        max_bp = xr.concat(
            [bpv.isel(flow=first_of[c]).max('bp') for c in pw_convs],
            dim=pd.Index(pw_convs, name='converter'),
        )
        avail = pw.availability.rename({Dim.PW_CONVERTER: 'converter'})
        sources['pw_avail_bound'] = tidy(avail * max_bp, drop_zero=False)
        # A gated curve's Status is keyed by the converter's own id, so the
        # lookup maps it to itself — `dict.fromkeys` would map it to None,
        # which reads as 'no Status' and leaves the curve ungated.
        pw_status_of = {c: c for c in gated}
        of = dict(zip(pair_flow, pair_conv, strict=True))
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
    for name in PERIOD_PARAMS:
        df = sources.get(name)
        if df is None:
            continue
        if 'period' in df.columns:
            sources[name] = df.assign(period=[period_ord[v] for v in df['period']])
        else:
            sources[name] = df.merge(pd.DataFrame({'period': p_ordinals}), how='cross')

    for name in BUILD_PERIOD_PARAMS:
        df = sources.get(name)
        if df is not None and 'build_period' in df.columns:
            sources[name] = df.assign(build_period=[period_ord[v] for v in df['build_period']])

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
        'carrier': labels([str(c) for c in data.carriers.unit.coords['carrier'].values]),
        # Both kinds: a converter states linear equations, a piecewise curve,
        # or one of each. The axis is the union, or a curve's own converter
        # would not be a coordinate of the dimension its rows are keyed on.
        'converter': _index_frame('converter', converter_ids, {'pw_status_of': pw_status_of}),
        'eq_idx': axis('eq_idx', range(data.converters.eq_mask.sizes['eq_idx']) if data.converters is not None else []),
        'storage': labels(storage_ids),
        'effect': labels(effect_ids),
        'status_entity': labels(entity_ids),
        # numpy, not a list: with no piecewise converter the width is 0 and a
        # bare `[]` has no integer type for the join to match.
        'bp': axis('bp', range(bp_width)),
    }
    _stamp_empty_dtypes(sources)
    return sources, coords
