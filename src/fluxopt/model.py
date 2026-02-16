from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linopy import Model, Variable

from fluxopt.results import SolvedModel
from fluxopt.types import as_dataarray

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


class FlowSystemModel:
    def __init__(self, data: ModelData, solver: str = 'highs', *, silent: bool = True) -> None:
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
        """Wrapper around linopy Model.add_variables that aligns bounds to coords.

        Bounds are converted to DataArrays aligned with the target *coords* via
        :func:`~fluxopt.types.as_dataarray`, preventing spurious extra dimensions
        that linopy can introduce when raw DataArray bounds carry mismatched dims.
        """
        coord_dict = {str(c.dims[0]): c.values for c in coords}
        kwargs: dict[str, Any] = {'coords': coords, 'name': name, 'binary': binary}
        if not binary:
            if lower is not None:
                kwargs['lower'] = as_dataarray(lower, coord_dict, broadcast=True)
            if upper is not None:
                kwargs['upper'] = as_dataarray(upper, coord_dict, broadcast=True)
        return self.m.add_variables(**kwargs)

    def build(self) -> None:
        # Phase 1: Decision variables
        self._create_flow_variables()
        self._create_sizing_variables()
        # Phase 2: Constraints
        self._constrain_flow_rates()
        self._constrain_sizing()
        self._create_bus_balance()
        self._create_converter_constraints()
        self._create_storage()
        self._create_effects()
        self._set_objective()

    def solve(self, silent: bool = True) -> SolvedModel:
        self.m.solve(solver_name=self.solver, io_api='direct', output_flag=not silent)
        return SolvedModel.from_model(self)

    def _create_flow_variables(self) -> None:
        """Create flow rate decision variable P_{f,t} >= 0."""
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

    def _constrain_flow_rates(self) -> None:
        """Apply flow rate bounds for all flow types.

        Fixed-size: P in [size * p_lb, size * p_ub] or P = size * pi
        Investable: P in [S * p_lb, S * p_ub] or P = S * pi
        Unsized: no bounds beyond P >= 0
        """
        ds = self.data.flows
        size = ds.size
        rel_lb = ds.rel_lb
        rel_ub = ds.rel_ub
        fixed = ds.fixed_profile
        is_bounded = ds.bound_type == 'bounded'
        is_profile = ds.bound_type == 'profile'

        # --- Fixed-size flows (parameter capacity) ---
        fixed_bounded = is_bounded & size.notnull()
        if fixed_bounded.any():
            mask = fixed_bounded.broadcast_like(rel_lb)
            self.m.add_constraints(self.flow_rate >= size * rel_lb, name='flow_lb', mask=mask)
            self.m.add_constraints(self.flow_rate <= size * rel_ub, name='flow_ub', mask=mask)

        fixed_profile = is_profile & size.notnull()
        if fixed_profile.any():
            mask = fixed_profile.broadcast_like(fixed) & fixed.notnull()
            self.m.add_constraints(self.flow_rate == size * fixed, name='flow_fix', mask=mask)

        # --- Investable flows (variable capacity) ---
        if hasattr(self, 'flow_size'):
            invest_ids = list(self.flow_size.coords['flow'].values)
            fr = self.flow_rate.sel(flow=invest_ids)
            rl = rel_lb.sel(flow=invest_ids)
            ru = rel_ub.sel(flow=invest_ids)
            fp = fixed.sel(flow=invest_ids)
            inv_bounded = is_bounded.sel(flow=invest_ids)
            inv_profile = is_profile.sel(flow=invest_ids)

            var_mask = inv_bounded.broadcast_like(rl)
            if var_mask.any():
                self.m.add_constraints(fr >= rl * self.flow_size, name='flow_lb_invest', mask=var_mask)
                self.m.add_constraints(fr <= ru * self.flow_size, name='flow_ub_invest', mask=var_mask)

            if inv_profile.any():
                fix_mask = inv_profile.broadcast_like(fp) & fp.notnull()
                self.m.add_constraints(fr == fp * self.flow_size, name='flow_fix_invest', mask=fix_mask)

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

    def _create_bus_balance(self) -> None:
        """Bus balance: sum_f(coeff_{b,f} * P_{f,t}) = 0  for all b, t."""
        d = self.data
        coeff = d.buses.flow_coeff  # (bus, flow) — NaN for unconnected

        # Replace NaN with 0 for summation (unconnected flows contribute nothing)
        coeff_filled = coeff.fillna(0)

        # Bus balance: sum over flow dim of (coeff * flow_rate) == 0
        lhs = (coeff_filled * self.flow_rate).sum('flow')
        self.m.add_constraints(lhs == 0, name='bus_balance')

    def _create_converter_constraints(self) -> None:
        """Conversion: sum_f(a_f * P_{f,t}) = 0  for all converter, eq_idx, t."""
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

        if has_any_coeff:
            contribution = (effect_coeff * self.flow_rate * d.dt).sum('flow')
            self.m.add_constraints(self.effect_per_timestep == contribution, name='effect_per_ts_eq')
        else:
            self.m.add_constraints(self.effect_per_timestep == 0, name='effect_per_ts_eq')

        # effect_total[effect] = sum_t(ept * w) + investment contributions
        self.effect_total = self.m.add_variables(coords=[effect_ids], name='effect--total')
        rhs = (self.effect_per_timestep * d.weights).sum('time')

        # Flow sizing: per-size costs
        if hasattr(self, 'flow_size'):
            assert d.flows.sizing_effects_per_size is not None
            eps = d.flows.sizing_effects_per_size.rename({'sizing_flow': 'flow'})
            if (eps != 0).any():
                rhs = rhs + (eps * self.flow_size).sum('flow')

        # Flow sizing: fixed costs (binary * cost)
        if hasattr(self, 'flow_size_indicator'):
            assert d.flows.sizing_effects_fixed is not None
            opt_ids = list(self.flow_size_indicator.coords['flow'].values)
            ef = d.flows.sizing_effects_fixed.rename({'sizing_flow': 'flow'}).sel(flow=opt_ids)
            if (ef != 0).any():
                rhs = rhs + (ef * self.flow_size_indicator).sum('flow')

        # Storage sizing: per-size costs
        if (
            hasattr(self, 'storage_capacity')
            and d.storages is not None
            and d.storages.sizing_effects_per_size is not None
        ):
            eps = d.storages.sizing_effects_per_size.rename({'sizing_storage': 'storage'})
            if (eps != 0).any():
                rhs = rhs + (eps * self.storage_capacity).sum('storage')

        # Storage sizing: fixed costs
        if (
            hasattr(self, 'storage_size_indicator')
            and d.storages is not None
            and d.storages.sizing_effects_fixed is not None
        ):
            opt_ids = list(self.storage_size_indicator.coords['storage'].values)
            ef = d.storages.sizing_effects_fixed.rename({'sizing_storage': 'storage'}).sel(storage=opt_ids)
            if (ef != 0).any():
                rhs = rhs + (ef * self.storage_size_indicator).sum('storage')

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
        """Storage dynamics: E_{s,t+1} = E_{s,t}(1 - delta*dt) + P^c eta^c dt - P^d/eta^d dt."""
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
        """Objective: min Phi_{k*} where k* is the effect with is_objective=True."""
        obj_effect = self.data.effects.objective_effect
        self.m.add_objective(self.effect_total.sel(effect=obj_effect).sum())
