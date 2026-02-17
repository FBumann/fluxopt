from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linopy import Model, Variable

from fluxopt.constraints.status import add_duration_tracking, add_switch_transitions
from fluxopt.results import SolvedModel
from fluxopt.types import as_dataarray

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


class FlowSystemModel:
    def __init__(self, data: ModelData, solver: str = 'highs', *, silent: bool = True) -> None:
        """Initialize the flow system optimization model.

        Args:
            data: Pre-built model data.
            solver: Solver backend name.
            silent: Suppress solver output.
        """
        self.data = data
        self.solver = solver
        self.silent = silent
        self.m = Model()

    def _add_variables(
        self,
        *,
        lower: Any = None,
        upper: Any = None,
        coords: list[xr.DataArray],
        name: str,
        binary: bool = False,
    ) -> Variable:
        """Add a variable with bounds auto-aligned to coords via as_dataarray.

        Args:
            lower: Lower bound (scalar, array, or DataArray).
            upper: Upper bound (scalar, array, or DataArray).
            coords: Coordinate arrays defining the variable dimensions.
            name: Variable name.
            binary: Create a binary variable instead.
        """
        coord_dict = {str(c.dims[0]): c.values for c in coords}
        kwargs: dict[str, Any] = {'coords': coords, 'name': name, 'binary': binary}
        if not binary:
            if lower is not None:
                kwargs['lower'] = as_dataarray(lower, coord_dict)
            if upper is not None:
                kwargs['upper'] = as_dataarray(upper, coord_dict)
        return self.m.add_variables(**kwargs)

    def build(self) -> None:
        """Build all variables, constraints, and the objective."""
        # Phase 1: Decision variables
        self._create_flow_variables()
        self._create_sizing_variables()
        self._create_status_variables()
        # Phase 2: Flow rate constraints
        self._constrain_flow_rates_plain()
        self._constrain_flow_rates_sizing()
        self._constrain_flow_rates_status()
        # Phase 3: Feature constraints
        self._constrain_sizing()
        self._constrain_status()
        # Phase 4: System
        self._create_bus_balance()
        self._create_converter_constraints()
        self._create_storage()
        self._create_effects()
        self._set_objective()

    def solve(self, silent: bool = True) -> SolvedModel:
        """Solve the model and return a SolvedModel.

        Args:
            silent: Suppress solver output.
        """
        self.m.solve(solver_name=self.solver, io_api='direct', output_flag=not silent)
        return SolvedModel.from_model(self)

    def _create_flow_variables(self) -> None:
        """Create flow rate decision variables P_{f,t} >= 0."""
        ds = self.data.flows
        self.flow_rate = self.m.add_variables(
            lower=0, coords=[ds.rel_lb.coords['flow'], ds.rel_lb.coords['time']], name='flow--rate'
        )

    def _create_sizing_variables(self) -> None:
        """Create sizing decision variables for flows and storages."""
        # --- Flow sizing ---
        fds = self.data.flows
        if fds.sizing_min is not None:
            assert fds.sizing_max is not None
            assert fds.sizing_mandatory is not None
            sizing_ids = fds.sizing_min.coords['sizing_flow'].values
            flow_coord = xr.DataArray(sizing_ids, dims=['flow'])
            upper = fds.sizing_max.rename({'sizing_flow': 'flow'})
            self.flow_size = self._add_variables(lower=0, upper=upper, coords=[flow_coord], name='flow--size')
            mandatory = fds.sizing_mandatory
            optional_ids = sizing_ids[~mandatory.values]
            if len(optional_ids):
                self.flow_size_indicator = self._add_variables(
                    binary=True,
                    coords=[xr.DataArray(optional_ids, dims=['flow'])],
                    name='flow--size_indicator',
                )

        # --- Storage capacity sizing ---
        sds = self.data.storages
        if sds is not None and sds.sizing_min is not None:
            assert sds.sizing_max is not None
            assert sds.sizing_mandatory is not None
            sizing_ids = sds.sizing_min.coords['sizing_storage'].values
            stor_coord = xr.DataArray(sizing_ids, dims=['storage'])
            upper = sds.sizing_max.rename({'sizing_storage': 'storage'})
            self.storage_capacity = self._add_variables(
                lower=0, upper=upper, coords=[stor_coord], name='storage--capacity'
            )
            mandatory = sds.sizing_mandatory
            optional_ids = sizing_ids[~mandatory.values]
            if len(optional_ids):
                self.storage_size_indicator = self._add_variables(
                    binary=True,
                    coords=[xr.DataArray(optional_ids, dims=['storage'])],
                    name='storage--size_indicator',
                )

    def _create_status_variables(self) -> None:
        """Create binary on/off variables for flows with Status."""
        ds = self.data.flows
        if ds.status_min_uptime is None:
            return

        status_ids = ds.status_min_uptime.coords['status_flow'].values
        flow_coord = xr.DataArray(status_ids, dims=['flow'])
        time_coord = ds.rel_lb.coords['time']

        self.flow_on = self._add_variables(binary=True, coords=[flow_coord, time_coord], name='flow--on')
        self.flow_startup = self._add_variables(binary=True, coords=[flow_coord, time_coord], name='flow--startup')
        self.flow_shutdown = self._add_variables(binary=True, coords=[flow_coord, time_coord], name='flow--shutdown')

    def _status_flow_ids(self) -> set[str]:
        """Return ids of flows with Status, or empty set."""
        ds = self.data.flows
        if ds.status_min_uptime is None:
            return set()
        return set(ds.status_min_uptime.coords['status_flow'].values)

    def _sizing_flow_ids(self) -> set[str]:
        """Return ids of investable flows, or empty set."""
        if not hasattr(self, 'flow_size'):
            return set()
        return set(self.flow_size.coords['flow'].values)

    def _constrain_flow_rates_plain(self) -> None:
        """Apply flow rate bounds for fixed-size flows without Status.

        P in [size * rel_lb, size * rel_ub] or P = size * profile.
        """
        ds = self.data.flows
        sizing_ids = self._sizing_flow_ids()
        status_ids = self._status_flow_ids()
        exclude = sizing_ids | status_ids

        size = ds.size
        rel_lb = ds.rel_lb
        rel_ub = ds.rel_ub
        fixed = ds.fixed_profile
        is_bounded = ds.bound_type == 'bounded'
        is_profile = ds.bound_type == 'profile'

        # Mask: has fixed size AND not sizing AND not status
        is_plain = size.notnull()
        if exclude:
            exclude_mask = xr.DataArray(
                [fid in exclude for fid in size.coords['flow'].values],
                dims=['flow'],
                coords={'flow': size.coords['flow']},
            )
            is_plain = is_plain & ~exclude_mask

        plain_bounded = is_bounded & is_plain
        if plain_bounded.any():
            mask = plain_bounded.broadcast_like(rel_lb)
            self.m.add_constraints(self.flow_rate >= size * rel_lb, name='flow_lb', mask=mask)
            self.m.add_constraints(self.flow_rate <= size * rel_ub, name='flow_ub', mask=mask)

        plain_profile = is_profile & is_plain
        if plain_profile.any():
            mask = plain_profile.broadcast_like(fixed) & fixed.notnull()
            self.m.add_constraints(self.flow_rate == size * fixed, name='flow_fix', mask=mask)

    def _constrain_flow_rates_sizing(self) -> None:
        """Apply flow rate bounds for investable flows without Status.

        P in [S * rel_lb, S * rel_ub] or P = S * profile.
        """
        if not hasattr(self, 'flow_size'):
            return

        ds = self.data.flows
        status_ids = self._status_flow_ids()

        all_invest_ids = list(self.flow_size.coords['flow'].values)
        invest_ids = [fid for fid in all_invest_ids if fid not in status_ids]
        if not invest_ids:
            return

        fr = self.flow_rate.sel(flow=invest_ids)
        rl = ds.rel_lb.sel(flow=invest_ids)
        ru = ds.rel_ub.sel(flow=invest_ids)
        fp = ds.fixed_profile.sel(flow=invest_ids)
        inv_bounded = (ds.bound_type == 'bounded').sel(flow=invest_ids)
        inv_profile = (ds.bound_type == 'profile').sel(flow=invest_ids)
        fs = self.flow_size.sel(flow=invest_ids)

        var_mask = inv_bounded.broadcast_like(rl)
        if var_mask.any():
            self.m.add_constraints(fr >= rl * fs, name='flow_lb_invest', mask=var_mask)
            self.m.add_constraints(fr <= ru * fs, name='flow_ub_invest', mask=var_mask)

        if inv_profile.any():
            fix_mask = inv_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == fp * fs, name='flow_fix_invest', mask=fix_mask)

    def _constrain_flow_rates_status(self) -> None:
        """Apply semi-continuous flow rate bounds for flows with Status.

        Fixed-size: P <= size * rel_ub * on, P >= size * rel_lb * on.
        Sizing: big-M formulation via _constrain_flow_rates_status_sizing().
        """
        if not hasattr(self, 'flow_on'):
            return

        ds = self.data.flows
        all_status_ids = list(self.flow_on.coords['flow'].values)
        sizing_ids = self._sizing_flow_ids()
        overlap = sizing_ids & set(all_status_ids)

        # Dispatch status+sizing flows to dedicated method
        if overlap:
            self._constrain_flow_rates_status_sizing(sorted(overlap))

        # Status-only flows (fixed size)
        status_ids = [fid for fid in all_status_ids if fid not in overlap]
        if not status_ids:
            return

        fr = self.flow_rate.sel(flow=status_ids)
        size = ds.size.sel(flow=status_ids)
        rl = ds.rel_lb.sel(flow=status_ids)
        ru = ds.rel_ub.sel(flow=status_ids)
        fp = ds.fixed_profile.sel(flow=status_ids)
        is_bounded = (ds.bound_type == 'bounded').sel(flow=status_ids)
        is_profile = (ds.bound_type == 'profile').sel(flow=status_ids)

        stat_bounded = is_bounded & size.notnull()
        if stat_bounded.any():
            mask = stat_bounded.broadcast_like(rl)
            self.m.add_constraints(fr >= size * rl * self.flow_on, name='flow_lb_status', mask=mask)
            self.m.add_constraints(fr <= size * ru * self.flow_on, name='flow_ub_status', mask=mask)

        stat_profile = is_profile & size.notnull()
        if stat_profile.any():
            mask = stat_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == size * fp * self.flow_on, name='flow_fix_status', mask=mask)

    def _constrain_flow_rates_status_sizing(self, flow_ids: list[str]) -> None:
        """Apply big-M flow rate bounds for flows with both Status and Sizing.

        Three constraints decouple the binary on/off from the continuous size:
          P <= on * M⁺          (big-M: forces on=1 when P>0)
          P <= S  * rel_ub      (rate limited by invested size)
          P >= (on - 1) * M⁻ + S * rel_lb  (enforces minimum when on=1)

        Where M⁺ = max_size * rel_ub, M⁻ = max_size * rel_lb.

        Args:
            flow_ids: Flows that have both Status and Sizing.
        """
        ds = self.data.flows
        assert ds.sizing_max is not None

        fr = self.flow_rate.sel(flow=flow_ids)
        on = self.flow_on.sel(flow=flow_ids)
        fs = self.flow_size.sel(flow=flow_ids)
        rl = ds.rel_lb.sel(flow=flow_ids)
        ru = ds.rel_ub.sel(flow=flow_ids)

        # max_size lives on the sizing_flow dim — rename and select
        max_size = ds.sizing_max.rename({'sizing_flow': 'flow'}).sel(flow=flow_ids)

        big_m_ub = max_size * ru  # M⁺
        big_m_lb = max_size * rl  # M⁻

        self.m.add_constraints(fr <= on * big_m_ub, name='flow_ub_status_sizing_bigm')
        self.m.add_constraints(fr <= fs * ru, name='flow_ub_status_sizing_size')
        self.m.add_constraints(fr >= (on - 1) * big_m_lb + fs * rl, name='flow_lb_status_sizing')

        # on <= S: prevents on=1 when size=0, which would incorrectly
        # charge running/startup costs for a non-existent unit.
        self.m.add_constraints(on <= fs, name='flow_on_requires_size')

    def _constrain_sizing(self) -> None:
        """Constrain sizing variables: S in [min, max] gated by indicator."""
        # --- Flow sizing ---
        if hasattr(self, 'flow_size'):
            fds = self.data.flows
            assert fds.sizing_min is not None
            assert fds.sizing_max is not None
            assert fds.sizing_mandatory is not None
            smin = fds.sizing_min.rename({'sizing_flow': 'flow'})
            mandatory = fds.sizing_mandatory.rename({'sizing_flow': 'flow'})

            mand_ids = self.flow_size.coords['flow'].values[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.flow_size.sel(flow=mand_ids) >= smin.sel(flow=mand_ids),
                    name='invest_mand_lb',
                )

            opt_ids = self.flow_size.coords['flow'].values[~mandatory.values]
            if len(opt_ids):
                smax = fds.sizing_max.rename({'sizing_flow': 'flow'})
                fs = self.flow_size.sel(flow=opt_ids)
                self.m.add_constraints(fs >= smin.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_lb')
                self.m.add_constraints(fs <= smax.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_ub')

        # --- Storage capacity sizing ---
        if hasattr(self, 'storage_capacity'):
            sds = self.data.storages
            assert sds is not None
            assert sds.sizing_min is not None
            assert sds.sizing_max is not None
            assert sds.sizing_mandatory is not None
            smin = sds.sizing_min.rename({'sizing_storage': 'storage'})
            mandatory = sds.sizing_mandatory.rename({'sizing_storage': 'storage'})

            mand_ids = self.storage_capacity.coords['storage'].values[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.storage_capacity.sel(storage=mand_ids) >= smin.sel(storage=mand_ids),
                    name='stor_invest_mand_lb',
                )

            opt_ids = self.storage_capacity.coords['storage'].values[~mandatory.values]
            if len(opt_ids):
                smax = sds.sizing_max.rename({'sizing_storage': 'storage'})
                sc = self.storage_capacity.sel(storage=opt_ids)
                self.m.add_constraints(
                    sc >= smin.sel(storage=opt_ids) * self.storage_size_indicator, name='stor_invest_lb'
                )
                self.m.add_constraints(
                    sc <= smax.sel(storage=opt_ids) * self.storage_size_indicator, name='stor_invest_ub'
                )

    def _constrain_status(self) -> None:
        """Add switch transition and duration tracking constraints for status flows."""
        if not hasattr(self, 'flow_on'):
            return

        ds = self.data.flows
        assert ds.status_min_uptime is not None
        assert ds.status_max_uptime is not None
        assert ds.status_min_downtime is not None
        assert ds.status_max_downtime is not None
        assert ds.status_initial is not None

        # Rename status_flow -> flow to align with variable dims
        min_up = ds.status_min_uptime.rename({'status_flow': 'flow'})
        max_up = ds.status_max_uptime.rename({'status_flow': 'flow'})
        min_down = ds.status_min_downtime.rename({'status_flow': 'flow'})
        max_down = ds.status_max_downtime.rename({'status_flow': 'flow'})
        initial = ds.status_initial.rename({'status_flow': 'flow'})

        prev_up = (
            ds.status_previous_uptime.rename({'status_flow': 'flow'}) if ds.status_previous_uptime is not None else None
        )
        prev_down = (
            ds.status_previous_downtime.rename({'status_flow': 'flow'})
            if ds.status_previous_downtime is not None
            else None
        )

        # Filter to flows with known initial state
        has_initial = initial.notnull()
        previous_state = initial.sel(flow=initial.coords['flow'][has_initial]) if has_initial.any() else None

        add_switch_transitions(
            self.m,
            self.flow_on,
            self.flow_startup,
            self.flow_shutdown,
            name='status',
            previous_state=previous_state,
        )

        dt = self.data.dt

        # Uptime tracking
        has_any_up = min_up.notnull().any() | max_up.notnull().any()
        if has_any_up:
            add_duration_tracking(
                self.m,
                self.flow_on,
                dt,
                name='uptime',
                minimum=min_up,
                maximum=max_up,
                previous=prev_up,
            )

        # Downtime tracking: state = 1 - on
        has_any_down = min_down.notnull().any() | max_down.notnull().any()
        if has_any_down:
            add_duration_tracking(
                self.m,
                1 - self.flow_on,
                dt,
                name='downtime',
                minimum=min_down,
                maximum=max_down,
                previous=prev_down,
            )

    def _create_bus_balance(self) -> None:
        """Create bus balance: ``sum_f(coeff * P) = 0`` for all buses and timesteps."""
        d = self.data
        coeff = d.buses.flow_coeff  # (bus, flow) — NaN for unconnected

        # Replace NaN with 0 for summation (unconnected flows contribute nothing)
        coeff_filled = coeff.fillna(0)

        # Bus balance: sum over flow dim of (coeff * flow_rate) == 0
        lhs = (coeff_filled * self.flow_rate).sum('flow')
        self.m.add_constraints(lhs == 0, name='bus_balance')

    def _create_converter_constraints(self) -> None:
        """Create conversion constraints: ``sum_f(a_f * P) = 0`` per converter and equation."""
        d = self.data
        if d.converters is None:
            return

        ds = d.converters
        coeff = ds.flow_coeff  # (converter, eq_idx, flow, time) — NaN for absent
        eq_mask = ds.eq_mask  # (converter, eq_idx)

        # Replace NaN with 0 for summation
        coeff_filled = coeff.fillna(0)

        # Need to align flow_rate with converter flow coord
        # The converter dataset may have a different flow coord than the full flow_rate
        # Select only the flows that appear in the converter dataset
        conv_flow_ids = ds.flow_coeff.coords['flow'].values
        flow_rate_sel = self.flow_rate.sel(flow=conv_flow_ids)

        lhs = (coeff_filled * flow_rate_sel).sum('flow')

        # Broadcast eq_mask (converter, eq_idx) to (converter, eq_idx, time)
        mask_3d = eq_mask.expand_dims(time=ds.flow_coeff.coords['time'])
        self.m.add_constraints(lhs == 0, name='conversion', mask=mask_3d)

    def _create_effects(self) -> None:
        """Effect tracking: accumulate flow contributions into effect totals."""
        d = self.data
        ds = d.effects

        effect_ids = ds.min_total.coords['effect']
        time = ds.min_per_hour.coords['time']

        if len(effect_ids) == 0:
            return

        # effect_per_timestep[effect, time]
        self.effect_per_timestep = self.m.add_variables(coords=[effect_ids, time], name='effect--per_timestep')

        # Flow contributions: sum_f(coeff_{f,k,t} * P_{f,t} * dt_t) for each (effect, time)
        effect_coeff = d.flows.effect_coeff  # (flow, effect, time)
        has_any_coeff = (effect_coeff != 0).any()

        contribution: Any = 0
        if has_any_coeff:
            contribution = (effect_coeff * self.flow_rate * d.dt).sum('flow')

        # Status running costs: sum_f(running_coeff[f,k,t] * on[f,t] * dt[t])
        if d.flows.status_effects_running is not None:
            er = d.flows.status_effects_running.rename({'status_flow': 'flow'})
            if (er != 0).any():
                contribution = contribution + (er * self.flow_on * d.dt).sum('flow')

        # Status startup costs: sum_f(startup_coeff[f,k,t] * startup[f,t]) — per event, no dt
        if d.flows.status_effects_startup is not None:
            es = d.flows.status_effects_startup.rename({'status_flow': 'flow'})
            if (es != 0).any():
                contribution = contribution + (es * self.flow_startup).sum('flow')

        # Cross-effect per-timestep contributions: cf_per_hour[k,j,t] * ept[j,t]
        if ds.cf_per_hour is not None:
            source_pts = self.effect_per_timestep.rename({'effect': 'source_effect'})
            cf_contribution = (ds.cf_per_hour * source_pts).sum('source_effect')
            contribution = contribution + cf_contribution

        self.m.add_constraints(self.effect_per_timestep == contribution, name='effect_per_ts_eq')

        # effect_total[effect] = sum_t(ept * w) + invest_direct + invest_cross
        self.effect_total = self.m.add_variables(coords=[effect_ids], name='effect--total')
        rhs = (self.effect_per_timestep * d.weights).sum('time')

        # Accumulate direct investment contributions per effect
        invest_direct: Any = 0

        # Flow sizing: per-size costs
        if hasattr(self, 'flow_size'):
            assert d.flows.sizing_effects_per_size is not None
            eps = d.flows.sizing_effects_per_size.rename({'sizing_flow': 'flow'})
            if (eps != 0).any():
                invest_direct = invest_direct + (eps * self.flow_size).sum('flow')

        # Flow sizing: fixed costs (binary * cost)
        if hasattr(self, 'flow_size_indicator'):
            assert d.flows.sizing_effects_fixed is not None
            opt_ids = list(self.flow_size_indicator.coords['flow'].values)
            ef = d.flows.sizing_effects_fixed.rename({'sizing_flow': 'flow'}).sel(flow=opt_ids)
            if (ef != 0).any():
                invest_direct = invest_direct + (ef * self.flow_size_indicator).sum('flow')

        # Storage sizing: per-size costs
        if (
            hasattr(self, 'storage_capacity')
            and d.storages is not None
            and d.storages.sizing_effects_per_size is not None
        ):
            eps = d.storages.sizing_effects_per_size.rename({'sizing_storage': 'storage'})
            if (eps != 0).any():
                invest_direct = invest_direct + (eps * self.storage_capacity).sum('storage')

        # Storage sizing: fixed costs
        if (
            hasattr(self, 'storage_size_indicator')
            and d.storages is not None
            and d.storages.sizing_effects_fixed is not None
        ):
            opt_ids = list(self.storage_size_indicator.coords['storage'].values)
            ef = d.storages.sizing_effects_fixed.rename({'sizing_storage': 'storage'}).sel(storage=opt_ids)
            if (ef != 0).any():
                invest_direct = invest_direct + (ef * self.storage_size_indicator).sum('storage')

        rhs = rhs + invest_direct

        # Cross-effect investment contributions: cf_invest[k,j] * invest_direct[j]
        if ds.cf_invest is not None and not isinstance(invest_direct, int):
            source_inv = invest_direct.rename({'effect': 'source_effect'})
            rhs = rhs + (ds.cf_invest * source_inv).sum('source_effect')

        self.m.add_constraints(self.effect_total == rhs, name='effect_total_eq')

        # Bounds on effect_total
        min_total = ds.min_total  # (effect,) — NaN = unbounded
        max_total = ds.max_total

        has_min = min_total.notnull()
        if has_min.any():
            self.m.add_constraints(self.effect_total >= min_total, name='effect_min_total', mask=has_min)

        has_max = max_total.notnull()
        if has_max.any():
            self.m.add_constraints(self.effect_total <= max_total, name='effect_max_total', mask=has_max)

        # Per-hour bounds on effect_per_timestep
        min_ph = ds.min_per_hour  # (effect, time) — NaN = unbounded
        has_min_ph = min_ph.notnull()
        if has_min_ph.any():
            self.m.add_constraints(self.effect_per_timestep >= min_ph, name='effect_min_ph', mask=has_min_ph)

        max_ph = ds.max_per_hour
        has_max_ph = max_ph.notnull()
        if has_max_ph.any():
            self.m.add_constraints(self.effect_per_timestep <= max_ph, name='effect_max_ph', mask=has_max_ph)

    def _create_storage(self) -> None:
        """Create storage variables, charge balance, and initial/cyclic conditions."""
        d = self.data
        if d.storages is None:
            return
        ds = d.storages

        stor_ids = ds.capacity.coords['storage']
        time_extra = d.time_extra
        te_vals = list(time_extra.values)

        # storage_level[storage, time_extra] >= 0
        self.storage_level = self.m.add_variables(lower=0, coords=[stor_ids, time_extra], name='storage--level')

        # --- Capacity bounds on storage_level ---
        cap = ds.capacity  # (storage,) — NaN for uncapped/investable
        has_fixed_cap = cap.notnull()
        has_invest_cap = hasattr(self, 'storage_capacity')

        # Fixed-capacity storages: level <= capacity (parameter)
        if has_fixed_cap.any():
            cap_2d = cap.broadcast_like(
                xr.DataArray(np.nan, dims=['storage', 'time_extra'], coords=[stor_ids, time_extra])
            )
            mask_cap = has_fixed_cap.broadcast_like(cap_2d)
            self.m.add_constraints(self.storage_level <= cap_2d, name='cs_cap', mask=mask_cap)

        # Investable storages: level <= capacity (variable)
        if has_invest_cap:
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            cs_invest = self.storage_level.sel(storage=invest_ids)
            self.m.add_constraints(cs_invest <= self.storage_capacity, name='cs_cap_invest')

        # --- Relative charge state bounds ---
        # For fixed-capacity storages
        if has_fixed_cap.any():
            rel_cs_lb = ds.rel_cs_lb
            rel_cs_ub = ds.rel_cs_ub

            abs_cs_lb = rel_cs_lb * cap
            has_cs_lb = has_fixed_cap.broadcast_like(rel_cs_lb) & (abs_cs_lb > 1e-12)
            if has_cs_lb.any():
                self.m.add_constraints(self.storage_level >= abs_cs_lb, name='cs_lb', mask=has_cs_lb)

            abs_cs_ub = rel_cs_ub * cap
            cap_2d_check = cap.broadcast_like(rel_cs_ub)
            has_cs_ub = has_fixed_cap.broadcast_like(rel_cs_ub) & (abs_cs_ub < cap_2d_check - 1e-12)
            if has_cs_ub.any():
                self.m.add_constraints(self.storage_level <= abs_cs_ub, name='cs_ub', mask=has_cs_ub)

        # For investable storages: relative bounds use capacity variable
        if has_invest_cap:
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            rel_cs_lb_inv = ds.rel_cs_lb.sel(storage=invest_ids)
            rel_cs_ub_inv = ds.rel_cs_ub.sel(storage=invest_ids)
            cs_invest = self.storage_level.sel(storage=invest_ids)

            has_lb = (rel_cs_lb_inv > 1e-12).any('time_extra')
            if has_lb.any():
                lb_mask = has_lb.broadcast_like(rel_cs_lb_inv) & (rel_cs_lb_inv > 1e-12)
                self.m.add_constraints(
                    cs_invest >= rel_cs_lb_inv * self.storage_capacity,
                    name='cs_lb_invest',
                    mask=lb_mask,
                )

            has_ub = (rel_cs_ub_inv < 1 - 1e-12).any('time_extra')
            if has_ub.any():
                ub_mask = has_ub.broadcast_like(rel_cs_ub_inv) & (rel_cs_ub_inv < 1 - 1e-12)
                self.m.add_constraints(
                    cs_invest <= rel_cs_ub_inv * self.storage_capacity,
                    name='cs_ub_invest',
                    mask=ub_mask,
                )

        # Map charge/discharge flows to storage dimension via sel + rename
        charge_fids = [str(v) for v in ds.charge_flow.values]
        discharge_fids = [str(v) for v in ds.discharge_flow.values]
        stor_vals = stor_ids.values

        charge_rates = self.flow_rate.sel(flow=charge_fids).rename({'flow': 'storage'})
        charge_rates = charge_rates.assign_coords({'storage': stor_vals})
        discharge_rates = self.flow_rate.sel(flow=discharge_fids).rename({'flow': 'storage'})
        discharge_rates = discharge_rates.assign_coords({'storage': stor_vals})

        # Precompute pure-xarray coefficients (no linopy overhead)
        loss_factor = 1 - ds.loss * d.dt  # (storage, time)
        charge_factor = ds.eta_c * d.dt  # (storage, time)
        discharge_factor = d.dt / ds.eta_d  # (storage, time)

        # Slice charge_state into cs[t] and cs[t+1], rename time_extra -> time for alignment
        cs_curr = self.storage_level.isel(time_extra=slice(None, -1)).rename({'time_extra': 'time'})
        cs_curr = cs_curr.assign_coords({'time': d.time})
        cs_next = self.storage_level.isel(time_extra=slice(1, None)).rename({'time_extra': 'time'})
        cs_next = cs_next.assign_coords({'time': d.time})

        # Fully vectorized energy balance over (storage, time)
        balance_lhs = (
            cs_next - cs_curr * loss_factor - charge_rates * charge_factor + discharge_rates * discharge_factor
        )
        self.m.add_constraints(balance_lhs == 0, name='storage_balance')

        # Initial/cyclic conditions (vectorized over storages)
        cyclic_mask = ds.cyclic.values.astype(bool)

        if np.any(cyclic_mask):
            cyc_ids = [str(s) for s, c in zip(stor_ids.values, cyclic_mask, strict=True) if c]
            cs_first = self.storage_level.sel(storage=cyc_ids, time_extra=te_vals[0])
            cs_last = self.storage_level.sel(storage=cyc_ids, time_extra=te_vals[-1])
            self.m.add_constraints(cs_last == cs_first, name='cs_cyclic')

        if np.any(~cyclic_mask):
            noncyc_ids = [str(s) for s, c in zip(stor_ids.values, cyclic_mask, strict=True) if not c]
            initial = ds.initial_charge.sel(storage=noncyc_ids)
            cap_sel = cap.sel(storage=noncyc_ids)
            # Relative initial (0-1) * capacity; no capacity -> use raw value (should be 0)
            abs_initial = xr.where(cap_sel.notnull(), initial * cap_sel, initial)
            cs_first = self.storage_level.sel(storage=noncyc_ids, time_extra=te_vals[0])
            self.m.add_constraints(cs_first == abs_initial, name='cs_init')

    def _set_objective(self) -> None:
        """Set objective: minimize the total of the objective effect."""
        obj_effect = self.data.effects.objective_effect
        self.m.add_objective(self.effect_total.sel(effect=obj_effect).sum())
