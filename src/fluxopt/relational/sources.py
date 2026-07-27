"""Bind a :class:`~fluxopt.model_data.ModelData` to the relational math program.

This is the data half of fluxopt's second backend. Both backends consume the
same ``ModelData``: :class:`~fluxopt.model.FlowSystemModel` turns it into a
linopy model, while this module emits the parameter tables that
:data:`MATH_PROGRAM` declares, for a memory-bounded streaming build.

Sparsity is carried by *row absence* — a parameter keeps its declared rank
while its table holds only live entries. Arrays at or below a variable's own
grid (bounds) stay dense; only the ones whose rank exceeds it
(``effect_coeff``, ``conv_coeff``) are filtered, which is where the size is.

Every relation between entities is a coordinate on the ``flow`` dimension
rather than a matrix, so topology travels as rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
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
        'rate_lb',
        'rate_ub',
        'rate_lb_on',
        'rate_ub_on',
        'rate_fix_on',
        'rel_lb_size',
        'rel_ub_size',
        'profile_size',
        'uptime_ub',
        'downtime_ub',
        'ramp_up_limit',
        'ramp_down_limit',
        'ramp_up_coeff',
        'ramp_down_coeff',
        'effect_coeff',
        'on_coeff',
        'startup_coeff',
        'size_coeff',
        'ind_coeff',
        'cap_coeff',
        'cap_ind_coeff',
        'lump_const',
        'obj_weight',
        'periodic_min',
        'periodic_max',
        'has_periodic_min',
        'has_periodic_max',
        'total_weight',
        'bigm_ub',
        'bigm_lb',
        'size_ub',
        'invest_window',
        'invest_prior_active',
        'invest_recurring_size_coeff',
        'invest_recurring_fixed_coeff',
        'sab_coeff',
        'build_coeff',
    }
)

#: Parameters carrying the `build_period` axis; its labels map to ordinals too.
BUILD_PERIOD_PARAMS = frozenset({'invest_window', 'sab_coeff', 'build_coeff'})


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
        ``(sources, coords)`` ready to pass to ``farkas.solve``.

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
    lump_terms: list[tuple[str, xr.DataArray | None]] = []
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
    sources['rate_lb'] = tidy(lb, drop_zero=False)
    sources['rate_ub'] = tidy(ub, drop_zero=False)

    # --- carrier balance --------------------------------------------------
    coeff = data.carriers.flow_coeff  # (carrier, flow), NaN = unconnected
    live = coeff.notnull() & (coeff != 0)
    car_idx = live.values.argmax(axis=0)
    carriers = coeff.coords['carrier'].values
    flow_index = pd.DataFrame({'flow': flow_ids, 'carrier': [str(carriers[i]) for i in car_idx]})
    sources['carrier_sign'] = pd.DataFrame(
        {'flow': flow_ids, 'value': [float(coeff.values[i, j]) for j, i in enumerate(car_idx)]},
    )

    # --- converters -------------------------------------------------------
    if data.converters is not None:
        cds = data.converters
        pair_flow = [str(v) for v in cds.pair_flow.values]
        pair_conv = [str(v) for v in cds.pair_converter.values]
        conv_of = dict(zip(pair_flow, pair_conv, strict=True))
        flow_index['converter'] = [conv_of.get(f) for f in flow_ids]
        pc = cds.pair_coeff.assign_coords(pair=pair_flow).rename({'pair': 'flow'})
        sources['conv_coeff'] = tidy(pc, drop_zero=True)
        sources['eq_active'] = _tidy(cds.eq_mask.astype(float), drop_zero=True).assign(value=True)
    else:
        flow_index['converter'] = None
        sources['conv_coeff'] = pd.DataFrame({'flow': [], 'eq_idx': [], 'time': [], 'value': []})
        sources['eq_active'] = pd.DataFrame({'converter': [], 'eq_idx': [], 'value': []})

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

        sources['charge_factor'] = on_flow(sds.eta_c * dims.dt, charge)
        sources['discharge_factor'] = on_flow(dims.dt / sds.eta_d, discharge)
        sources['retention'] = tidy((1 - sds.loss) ** dims.dt, drop_zero=False)
        cap = sds.capacity
        sources['level_lb'] = tidy((sds.rel_level_lb * cap).fillna(0.0), drop_zero=False)
        sources['level_ub'] = tidy((sds.rel_level_ub * cap).fillna(np.inf), drop_zero=False)
        csz = sds.sizing
        if csz is not None:
            cdim = csz.min.dims[0]
            cap_ids = [str(v) for v in csz.min.coords[cdim].values]
            cmand = csz.mandatory.values.astype(bool)
            copt = [s for s, m in zip(cap_ids, cmand, strict=True) if not m]
            cmand_ids = [s for s, m in zip(cap_ids, cmand, strict=True) if m]
            sources['has_cap_sizing'] = pd.DataFrame({'storage': cap_ids, 'value': True})
            sources['cap_is_optional'] = pd.DataFrame({'storage': copt, 'value': True})
            sources['cap_min'] = pd.DataFrame({'storage': cap_ids, 'value': csz.min.values})
            sources['cap_max'] = pd.DataFrame({'storage': cap_ids, 'value': csz.max.values})
            sources['rel_level_lb'] = tidy(sds.rel_level_lb.sel(storage=cap_ids), drop_zero=True)
            sources['rel_level_ub'] = tidy(sds.rel_level_ub.sel(storage=cap_ids), drop_zero=True)
            cren = {cdim: 'storage'}
            lump_terms += [
                ('cap_coeff', csz.effects_per_size.rename(cren)),
                ('cap_ind_coeff', csz.effects_fixed.sel({cdim: copt}).rename(cren) if copt else None),
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
                sources[key] = pd.DataFrame({'storage': [], 'value': []})
                sources[f'has_{key.replace("final_level_", "final_")}'] = pd.DataFrame({'storage': [], 'value': []})
                continue
            live = arr.notnull()
            sources[key] = _tidy(arr.where(live), drop_zero=False)
            sources[f'has_{key.replace("final_level_", "final_")}'] = _tidy(live.astype(float), drop_zero=True).assign(
                value=True
            )
        prevent = (
            sds.prevent_simultaneous.values.astype(bool)
            if sds.prevent_simultaneous is not None
            else np.zeros(len(storage_ids), dtype=bool)
        )
        prev_ids = [s for s, p in zip(storage_ids, prevent, strict=True) if p]
        sources['prevent_simul'] = pd.DataFrame({'storage': prev_ids, 'value': True})
        sources['m_charge'] = pd.DataFrame(
            {'storage': storage_ids, 'value': [_size_upper(data, f) for f in charge]},
        )
        sources['m_discharge'] = pd.DataFrame(
            {'storage': storage_ids, 'value': [_size_upper(data, f) for f in discharge]},
        )
    else:
        flow_index['charge_storage'] = None
        flow_index['discharge_storage'] = None
        for name, dcols in (
            ('is_cyclic', ['storage']),
            ('prior_level', ['storage']),
            ('has_final_min', ['storage']),
            ('has_final_max', ['storage']),
            ('final_level_min', ['storage']),
            ('final_level_max', ['storage']),
            ('prevent_simul', ['storage']),
            ('m_charge', ['storage']),
            ('m_discharge', ['storage']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})
        for name, dcols in (
            ('charge_factor', ['flow', 'time']),
            ('discharge_factor', ['flow', 'time']),
            ('retention', ['storage', 'time']),
            ('level_lb', ['storage', 'time']),
            ('level_ub', ['storage', 'time']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})

    # --- status -----------------------------------------------------------
    ec_extra: list[tuple[str, xr.DataArray]] = []
    if st is not None:
        ren = {sdim: 'flow'}
        sources['has_status'] = pd.DataFrame({'flow': status_ids, 'value': True})
        sources['is_bounded'] = pd.DataFrame(
            {'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values},
        )
        sources['is_profile'] = pd.DataFrame(
            {'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values},
        )
        sel = {'flow': status_ids}
        sources['rate_lb_on'] = tidy((size * fds.rel_lb).sel(sel), drop_zero=True)
        sources['rate_ub_on'] = tidy((size * fds.rel_ub).sel(sel), drop_zero=True)
        sources['rate_fix_on'] = tidy((size * fds.fixed_profile).sel(sel), drop_zero=True)

        up_min = st.uptime_min.rename(ren)
        up_max = st.uptime_max.rename(ren)
        horizon = float(dims.dt.sum())
        has_up = (up_min.notnull() | up_max.notnull()).values
        up_ids = [f for f, h in zip(status_ids, has_up, strict=True) if h]
        sources['has_uptime'] = pd.DataFrame({'flow': up_ids, 'value': True})
        min_ids = [f for f, h in zip(status_ids, up_min.notnull().values, strict=True) if h]
        sources['has_uptime_min'] = pd.DataFrame({'flow': min_ids, 'value': True})
        sources['uptime_min'] = pd.DataFrame(
            {'flow': min_ids, 'value': up_min.sel(flow=min_ids).values if min_ids else []},
        )
        prev_up = st.previous_uptime.rename(ren) if st.previous_uptime is not None else None
        prev_dn = st.previous_downtime.rename(ren) if st.previous_downtime is not None else None
        envelope = fds.rel_lb.sel(sel)

        def _per_flow(arr: xr.DataArray | float) -> np.ndarray:
            return np.broadcast_to(np.asarray(arr, dtype=float), (len(status_ids),)).copy()

        mega_up = horizon + (prev_up.fillna(0.0) if prev_up is not None else 0.0)
        mega_dn = horizon + (prev_dn.fillna(0.0) if prev_dn is not None else 0.0)
        sources['mega_up'] = pd.DataFrame({'flow': status_ids, 'value': _per_flow(mega_up)})
        sources['mega_down'] = pd.DataFrame({'flow': status_ids, 'value': _per_flow(mega_dn)})
        sources['uptime_ub'] = tidy(up_max.fillna(mega_up).broadcast_like(envelope).sel(flow=up_ids), drop_zero=False)

        # --- downtime: the same tracking with the state inverted -------------
        dn_min, dn_max = st.downtime_min.rename(ren), st.downtime_max.rename(ren)
        dn_ids = [f for f, h in zip(status_ids, (dn_min.notnull() | dn_max.notnull()).values, strict=True) if h]
        dn_min_ids = [f for f, h in zip(status_ids, dn_min.notnull().values, strict=True) if h]
        sources['has_downtime'] = pd.DataFrame({'flow': dn_ids, 'value': True})
        sources['has_downtime_min'] = pd.DataFrame({'flow': dn_min_ids, 'value': True})
        sources['downtime_min'] = pd.DataFrame(
            {'flow': dn_min_ids, 'value': dn_min.sel(flow=dn_min_ids).values if dn_min_ids else []},
        )
        sources['downtime_ub'] = tidy(dn_max.fillna(mega_dn).broadcast_like(envelope).sel(flow=dn_ids), drop_zero=False)

        # --- initial state and pre-horizon carry-over ------------------------
        init = st.initial.rename(ren)
        init_ids = [f for f, h in zip(status_ids, init.notnull().values, strict=True) if h]
        sources['has_initial'] = pd.DataFrame({'flow': init_ids, 'value': True})
        sources['initial_on'] = pd.DataFrame(
            {'flow': init_ids, 'value': init.sel(flow=init_ids).values if init_ids else []},
        )
        sources['dt_first'] = pd.DataFrame({'flow': status_ids, 'value': float(dims.dt.values[0])})
        for tag, key, prev, lo in (
            ('up', 'prev_uptime', prev_up, up_min),
            ('down', 'prev_downtime', prev_dn, dn_min),
        ):
            if prev is None:
                sources[f'has_prev_{tag}'] = pd.DataFrame({'flow': [], 'value': []})
                sources[key] = pd.DataFrame({'flow': [], 'value': []})
                sources[f'force_{tag}_start'] = pd.DataFrame({'flow': [], 'value': []})
                continue
            pids = [f for f, h in zip(status_ids, prev.notnull().values, strict=True) if h]
            sources[f'has_prev_{tag}'] = pd.DataFrame({'flow': pids, 'value': True})
            sources[key] = pd.DataFrame({'flow': pids, 'value': prev.sel(flow=pids).values if pids else []})
            forced = ((prev > 0) & lo.notnull() & (prev < lo)).values
            sources[f'force_{tag}_start'] = pd.DataFrame(
                {'flow': [f for f, h in zip(status_ids, forced, strict=True) if h], 'value': True},
            )

        ec_extra = [
            ('on_coeff', (st.effects_running.rename(ren) * dims.dt * dims.weights)),
            ('startup_coeff', (st.effects_startup.rename(ren) * dims.weights)),
        ]
    else:
        for n in ('has_status', 'has_uptime', 'has_uptime_min'):
            sources[n] = pd.DataFrame({'flow': [], 'value': []})
        sources['is_bounded'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.BOUNDED).values})
        sources['is_profile'] = pd.DataFrame({'flow': flow_ids, 'value': (bt == BoundType.PROFILE).values})
        for n in (
            'uptime_min',
            'mega_up',
            'mega_down',
            'downtime_min',
            'has_downtime',
            'has_downtime_min',
            'has_initial',
            'initial_on',
            'has_prev_up',
            'prev_uptime',
            'has_prev_down',
            'prev_downtime',
            'force_up_start',
            'force_down_start',
            'dt_first',
        ):
            sources[n] = pd.DataFrame({'flow': [], 'value': []})
        for n in ('rate_lb_on', 'rate_ub_on', 'rate_fix_on', 'uptime_ub', 'downtime_ub'):
            sources[n] = pd.DataFrame({'flow': [], 'time': [], 'value': []})
        ec_extra = []

    dt_vals = dims.dt.values
    sources['dt_prev'] = pd.DataFrame({'time': ordinals[1:], 'value': dt_vals[:-1]})
    sources['is_last'] = pd.DataFrame({'time': ordinals, 'value': [i == len(ordinals) - 1 for i in ordinals]})

    # --- sizing -----------------------------------------------------------
    if sz is not None:
        zren = {zdim: 'flow'}
        mandatory = sz.mandatory.values.astype(bool)
        opt_ids = [f for f, m in zip(sizing_ids, mandatory, strict=True) if not m]
        mand_ids = [f for f, m in zip(sizing_ids, mandatory, strict=True) if m]
        sources['has_sizing'] = pd.DataFrame({'flow': sizing_ids, 'value': True})
        sources['is_optional'] = pd.DataFrame({'flow': opt_ids, 'value': True})
        sources['size_min'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.min.values})
        sources['size_max'] = pd.DataFrame({'flow': sizing_ids, 'value': sz.max.values})
        zsel = {'flow': sizing_ids}
        smax = xr.DataArray(sz.max.values, dims=['flow'], coords={'flow': sizing_ids})
        sources['bigm_ub'] = tidy((fds.rel_ub.sel(zsel) * smax), drop_zero=True)
        sources['bigm_lb'] = tidy((fds.rel_lb.sel(zsel) * smax), drop_zero=True)
        sources['has_size_min_pos'] = pd.DataFrame(
            {'flow': [f for f, v in zip(sizing_ids, sz.min.values, strict=True) if float(v) > 0], 'value': True},
        )
        lump_terms += [
            ('size_coeff', sz.effects_per_size.rename(zren)),
            ('ind_coeff', sz.effects_fixed.sel({zdim: opt_ids}).rename(zren) if opt_ids else None),
        ]
        if mand_ids:
            lump_consts.append(('flow', sz.effects_fixed.sel({zdim: mand_ids}).rename(zren)))
    else:
        for n in ('has_sizing', 'is_optional', 'size_min', 'size_max', 'has_size_min_pos'):
            sources[n] = pd.DataFrame({'flow': [], 'value': []})
        for n in ('bigm_ub', 'bigm_lb'):
            sources[n] = pd.DataFrame({'flow': [], 'time': [], 'value': []})

    if 'has_cap_sizing' not in sources:
        for n in ('has_cap_sizing', 'cap_is_optional', 'cap_min', 'cap_max'):
            sources[n] = pd.DataFrame({'storage': [], 'value': []})
        for n in ('rel_level_lb', 'rel_level_ub'):
            sources[n] = pd.DataFrame({'storage': [], 'time': [], 'value': []})

    # --- ramps ------------------------------------------------------------
    sized_set = set(sizing_ids) if sz is not None else set()
    for kind, arr in (('up', fds.ramp_up), ('down', fds.ramp_down)):
        if arr is None:
            sources[f'has_ramp_{kind}'] = pd.DataFrame({'flow': [], 'value': []})
            sources[f'ramp_{kind}_limit'] = pd.DataFrame({'flow': [], 'time': [], 'value': []})
            sources[f'ramp_{kind}_coeff'] = pd.DataFrame({'flow': [], 'time': [], 'value': []})
            continue
        nonflow = [d for d in arr.dims if d != 'flow']
        live = arr.notnull().any(nonflow) if nonflow else arr.notnull()
        ids = [str(f) for f in arr.coords['flow'].values[live.values]]
        sources[f'has_ramp_{kind}'] = pd.DataFrame({'flow': ids, 'value': True})
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
        sources['invest_window'] = _tidy(
            xr.DataArray(window, dims=['flow', 'period', 'build_period'], coords=coords_w), drop_zero=True
        )
        sources['invest_prior_active'] = _tidy(
            xr.DataArray(
                prior_active, dims=['flow', 'period'], coords={'flow': invest_ids, 'period': period_labels_inv}
            ),
            drop_zero=False,
        )
        sources['has_invest'] = pd.DataFrame({'flow': invest_ids, 'value': True})
        sources['invest_mandatory'] = pd.DataFrame({'flow': invest_ids, 'value': inv.mandatory.values.astype(bool)})
        sources['invest_min'] = pd.DataFrame({'flow': invest_ids, 'value': inv.min.values})
        sources['invest_max'] = pd.DataFrame({'flow': invest_ids, 'value': inv.max.values})
        sources['invest_prior_size'] = pd.DataFrame({'flow': invest_ids, 'value': prior})
        sources['has_invest_prior'] = pd.DataFrame(
            {'flow': [f for f, ps in zip(invest_ids, prior, strict=True) if ps > 0], 'value': True}
        )
        # Diagonal in (period, build_period): a one-time cost belongs to the
        # period the build happened in, not to every period the unit is alive.
        eye = xr.DataArray(
            np.eye(n_p),
            dims=['period', 'build_period'],
            coords={'period': period_labels_inv, 'build_period': period_labels_inv},
        )
        lump_terms += [
            ('sab_coeff', inv.effects_per_size_at_build.rename(iren).rename({'period': 'build_period'}) * eye),
            ('build_coeff', inv.effects_fixed_at_build.rename(iren).rename({'period': 'build_period'}) * eye),
            ('invest_recurring_size_coeff', inv.effects_per_size_recurring.rename(iren)),
            ('invest_recurring_fixed_coeff', inv.effects_fixed_recurring.rename(iren)),
        ]
    else:
        for name in (
            'has_invest',
            'invest_mandatory',
            'invest_min',
            'invest_max',
            'invest_prior_size',
            'has_invest_prior',
        ):
            sources[name] = pd.DataFrame({'flow': [], 'value': []})
        sources['invest_window'] = pd.DataFrame({'flow': [], 'period': [], 'build_period': [], 'value': []})
        sources['invest_prior_active'] = pd.DataFrame({'flow': [], 'period': [], 'value': []})

    # `sizing_rate` covers both mechanisms, so its envelope must span both.
    sized_ids = [*sizing_ids, *invest_ids]
    envelope_sel = {'flow': sized_ids}
    sources['rel_lb_size'] = tidy(fds.rel_lb.sel(envelope_sel), drop_zero=True)
    sources['rel_ub_size'] = tidy(fds.rel_ub.sel(envelope_sel), drop_zero=True)
    sources['profile_size'] = tidy(fds.fixed_profile.sel(envelope_sel), drop_zero=True)

    # upper bound of `size`: whichever mechanism sizes the flow
    size_ub = xr.full_like(fds.size, np.inf)
    for ids, maxima in (
        (sizing_ids, sz.max if sz is not None else None),
        (invest_ids, inv.max if inv is not None else None),
    ):
        if maxima is None or not ids:
            continue
        per_flow = xr.DataArray(maxima.values, dims=['flow'], coords={'flow': ids}).reindex(flow=flow_ids)
        size_ub = xr.where(per_flow.notnull(), per_flow, size_ub)
    sources['size_ub'] = _tidy(size_ub, drop_zero=False)

    # --- effects: the sparse one -----------------------------------------
    eds = data.effects
    effect_ids = [str(e) for e in eds.total_min.coords['effect'].values]
    ec = fds.effect_coeff * dims.dt * dims.weights
    if eds.cf_temporal is not None:
        ec = _apply_leontief(_leontief(eds.cf_temporal), ec)
    sources['effect_coeff'] = tidy(ec, drop_zero=True)
    leo = _leontief(eds.cf_temporal) if eds.cf_temporal is not None else None
    for name, arr in ec_extra:
        sources[name] = tidy(_apply_leontief(leo, arr) if leo is not None else arr, drop_zero=True)
    for name in ('on_coeff', 'startup_coeff'):
        sources.setdefault(name, pd.DataFrame({'flow': [], 'effect': [], 'time': [], 'value': []}))

    # Lump domain: effect_lump = (I - cf_lump)^-1 . lump_direct, folded into
    # the coefficients so no self-referential effect_lump variable is needed.
    leo_lump = _leontief(eds.cf_temporal.mean('time')) if eds.cf_temporal is not None else None

    def fold(arr: xr.DataArray) -> xr.DataArray:
        """Apply the lump-domain Leontief inverse, if there are cross-effects."""
        return arr if leo_lump is None else _apply_leontief(leo_lump, arr)

    for name, arr in lump_terms:
        sources[name] = (
            tidy(fold(arr), drop_zero=True)
            if arr is not None
            else pd.DataFrame({'flow': [], 'effect': [], 'value': []})
        )
    for name in ('size_coeff', 'ind_coeff'):
        sources.setdefault(name, pd.DataFrame({'flow': [], 'effect': [], 'value': []}))
    for name in ('cap_coeff', 'cap_ind_coeff'):
        sources.setdefault(name, pd.DataFrame({'storage': [], 'effect': [], 'value': []}))
    for name in ('sab_coeff', 'build_coeff'):
        sources.setdefault(
            name, pd.DataFrame({'flow': [], 'effect': [], 'period': [], 'build_period': [], 'value': []})
        )
    for name in ('invest_recurring_size_coeff', 'invest_recurring_fixed_coeff'):
        sources.setdefault(name, pd.DataFrame({'flow': [], 'effect': [], 'period': [], 'value': []}))
    total_const: xr.DataArray | None = None
    for edim, arr in lump_consts:
        part = fold(arr).sum(edim)
        total_const = part if total_const is None else total_const + part
    sources['lump_const'] = (
        _tidy(total_const, drop_zero=True) if total_const is not None else pd.DataFrame({'effect': [], 'value': []})
    )
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
    sources['obj_weight'] = _tidy(obj_w if pw is None else obj_w * pw, drop_zero=True)

    # --- effect limits ----------------------------------------------------
    for key, arr in (
        ('periodic_min', eds.periodic_min),
        ('periodic_max', eds.periodic_max),
        ('total_min', eds.total_min),
        ('total_max', eds.total_max),
    ):
        live = arr.notnull()
        sources[key] = _tidy(arr.where(live), drop_zero=False)
        sources[f'has_{key}'] = _tidy(live.astype(float), drop_zero=True).assign(value=True)
    # Weights for the across-period sum: per-effect override, else global, else 1.
    ones = xr.ones_like(obj_w)
    tw = ones if pw is None else ones * pw
    sources['total_weight'] = _tidy(tw, drop_zero=True)

    # --- temporal boundary mask ------------------------------------------
    sources['is_first'] = pd.DataFrame({'time': ordinals, 'value': [i == 0 for i in ordinals]})

    sources['flow'] = flow_index
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

    coords: dict[str, Any] = {
        'time': ordinals,
        'period': p_ordinals,
        'build_period': p_ordinals,
        'carrier': [str(c) for c in carriers],
        'converter': [str(c) for c in data.converters.eq_mask.coords['converter'].values]
        if data.converters is not None
        else empty,
        'eq_idx': list(range(data.converters.eq_mask.sizes['eq_idx'])) if data.converters is not None else empty,
        'storage': storage_ids,
        'effect': effect_ids,
    }
    return sources, coords
