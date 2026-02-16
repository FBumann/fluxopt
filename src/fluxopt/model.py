from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from linopy import Model

from fluxopt.results import SolvedModel

if TYPE_CHECKING:
    from fluxopt.tables import ModelData


class FlowSystemModel:
    def __init__(self, data: ModelData, solver: str = 'highs', *, silent: bool = True) -> None:
        self.data = data
        self.solver = solver
        self.silent = silent
        self.m = Model()

    def build(self) -> None:
        self._create_flow_variables()
        self._create_bus_balance()
        self._create_converter_constraints()
        self._create_effects()
        self._create_storage()
        self._set_objective()

    def solve(self, silent: bool = True) -> SolvedModel:
        self.m.solve(solver_name=self.solver, io_api='direct', output_flag=not silent)
        return SolvedModel.from_model(self)

    def _create_flow_variables(self) -> None:
        """Create flow rate variables and apply bounds.

        Sized: P̄_f · p̲_{f,t} <= P_{f,t} <= P̄_f · p̄_{f,t}
        Fixed: P_{f,t} = P̄_f · π_{f,t}
        """
        d = self.data
        ds = d.flows

        flow_ids = ds.coords['flow']
        time = ds.coords['time']

        # Create flow_rate variable indexed by (flow, time) >= 0
        self.flow_rate = self.m.add_variables(lower=0, coords=[flow_ids, time], name='flow_rate')

        size = ds['size']  # (flow,) — NaN for unsized
        rel_lb = ds['rel_lb']  # (flow, time)
        rel_ub = ds['rel_ub']  # (flow, time)
        fixed = ds['fixed_profile']  # (flow, time) — NaN where not fixed

        sized_mask = size.notnull()  # (flow,) bool
        has_fixed = fixed.notnull().any('time')  # (flow,) bool

        # Variable (non-fixed) sized flows: apply bounds
        variable_sized = sized_mask & ~has_fixed  # (flow,)
        if variable_sized.any():
            lb = rel_lb * size  # broadcasts (flow,time) * (flow,) -> (flow,time)
            ub = rel_ub * size
            mask_2d = variable_sized.broadcast_like(rel_lb)  # (flow, time)
            self.m.add_constraints(self.flow_rate >= lb, name='flow_lb', mask=mask_2d)
            self.m.add_constraints(self.flow_rate <= ub, name='flow_ub', mask=mask_2d)

        # Fixed profile flows
        if has_fixed.any():
            # Multiply by size where sized, else use as-is
            abs_fixed = xr.where(sized_mask, fixed * size, fixed)
            mask_fixed = has_fixed.broadcast_like(fixed) & fixed.notnull()
            self.m.add_constraints(self.flow_rate == abs_fixed, name='flow_fix', mask=mask_fixed)

    def _create_bus_balance(self) -> None:
        """Bus balance: sum_f(coeff_{b,f} · P_{f,t}) = 0  for all b, t."""
        d = self.data
        coeff = d.buses['flow_coeff']  # (bus, flow) — NaN for unconnected

        # Replace NaN with 0 for summation (unconnected flows contribute nothing)
        coeff_filled = coeff.fillna(0)

        # Bus balance: sum over flow dim of (coeff * flow_rate) == 0
        lhs = (coeff_filled * self.flow_rate).sum('flow')
        self.m.add_constraints(lhs == 0, name='bus_balance')

    def _create_converter_constraints(self) -> None:
        """Conversion: sum_f(a_f · P_{f,t}) = 0  for all converter, eq_idx, t."""
        d = self.data
        if not d.converters.data_vars:
            return

        ds = d.converters
        coeff = ds['flow_coeff']  # (converter, eq_idx, flow, time) — NaN for absent
        eq_mask = ds['eq_mask']  # (converter, eq_idx)

        # Replace NaN with 0 for summation
        coeff_filled = coeff.fillna(0)

        # Need to align flow_rate with converter flow coord
        # The converter dataset may have a different flow coord than the full flow_rate
        # Select only the flows that appear in the converter dataset
        conv_flow_ids = ds.coords['flow'].values
        flow_rate_sel = self.flow_rate.sel(flow=conv_flow_ids)

        lhs = (coeff_filled * flow_rate_sel).sum('flow')

        # Broadcast eq_mask (converter, eq_idx) to (converter, eq_idx, time)
        mask_3d = eq_mask.expand_dims(time=ds.coords['time'])
        self.m.add_constraints(lhs == 0, name='conversion', mask=mask_3d)

    def _create_effects(self) -> None:
        """Effect tracking: accumulate flow contributions into effect totals."""
        d = self.data
        ds = d.effects

        effect_ids = ds.coords['effect']
        time = ds.coords['time']

        if len(effect_ids) == 0:
            return

        # effect_per_timestep[effect, time]
        self.effect_per_timestep = self.m.add_variables(coords=[effect_ids, time], name='effect_per_timestep')

        # Flow contributions: sum_f(coeff_{f,k,t} * P_{f,t} * dt_t) for each (effect, time)
        effect_coeff = d.flows['effect_coeff']  # (flow, effect, time)
        has_any_coeff = (effect_coeff != 0).any()

        if has_any_coeff:
            contribution = (effect_coeff * self.flow_rate * d.dt).sum('flow')
            self.m.add_constraints(self.effect_per_timestep == contribution, name='effect_per_ts_eq')
        else:
            self.m.add_constraints(self.effect_per_timestep == 0, name='effect_per_ts_eq')

        # effect_temporal[effect] = sum_t(effect_per_timestep * weight)
        self.effect_temporal = self.m.add_variables(coords=[effect_ids], name='effect_temporal')
        weighted_sum = (self.effect_per_timestep * d.weights).sum('time')
        self.m.add_constraints(self.effect_temporal == weighted_sum, name='effect_temporal_eq')

        # effect_total[effect] = effect_temporal (future: + periodic)
        self.effect_total = self.m.add_variables(coords=[effect_ids], name='effect_total')
        self.m.add_constraints(self.effect_total == self.effect_temporal, name='effect_total_eq')

        # Bounds on effect_total
        min_total = ds['min_total']  # (effect,) — NaN = unbounded
        max_total = ds['max_total']

        has_min = min_total.notnull()
        if has_min.any():
            self.m.add_constraints(self.effect_total >= min_total, name='effect_min_total', mask=has_min)

        has_max = max_total.notnull()
        if has_max.any():
            self.m.add_constraints(self.effect_total <= max_total, name='effect_max_total', mask=has_max)

        # Per-hour bounds on effect_per_timestep
        min_ph = ds['min_per_hour']  # (effect, time) — NaN = unbounded
        has_min_ph = min_ph.notnull()
        if has_min_ph.any():
            self.m.add_constraints(self.effect_per_timestep >= min_ph, name='effect_min_ph', mask=has_min_ph)

        max_ph = ds['max_per_hour']
        has_max_ph = max_ph.notnull()
        if has_max_ph.any():
            self.m.add_constraints(self.effect_per_timestep <= max_ph, name='effect_max_ph', mask=has_max_ph)

    def _create_storage(self) -> None:
        """Storage dynamics: E_{s,t+1} = E_{s,t}(1 - δΔt) + P^c η^c Δt - P^d/η^d Δt."""
        d = self.data
        ds = d.storages
        if not ds.data_vars:
            return

        stor_ids = ds.coords['storage']
        time_extra = d.time_extra
        time_list = list(d.time.values)
        te_list = list(time_extra.values)

        # charge_state[storage, time_extra] >= 0
        self.charge_state = self.m.add_variables(lower=0, coords=[stor_ids, time_extra], name='charge_state')

        # Capacity upper bound
        cap = ds['capacity']  # (storage,) — NaN for unsized
        has_cap = cap.notnull()
        if has_cap.any():
            cap_broadcast = cap.broadcast_like(
                xr.DataArray(np.nan, dims=['storage', 'time_extra'], coords=[stor_ids, time_extra])
            )
            mask_cap = has_cap.broadcast_like(cap_broadcast)
            self.m.add_constraints(self.charge_state <= cap_broadcast, name='cs_cap', mask=mask_cap)

        # Relative charge state bounds (absolute = relative * capacity or large M)
        rel_cs_lb = ds['rel_cs_lb']  # (storage, time_extra)
        rel_cs_ub = ds['rel_cs_ub']

        cap_or_large = xr.where(cap.notnull(), cap, 1e9)

        abs_cs_lb = rel_cs_lb * cap_or_large
        has_cs_lb = abs_cs_lb > 1e-12
        if has_cs_lb.any():
            self.m.add_constraints(self.charge_state >= abs_cs_lb, name='cs_lb', mask=has_cs_lb)

        abs_cs_ub = rel_cs_ub * cap_or_large
        has_cs_ub = abs_cs_ub < cap_or_large - 1e-12
        if has_cs_ub.any():
            self.m.add_constraints(self.charge_state <= abs_cs_ub, name='cs_ub', mask=has_cs_ub)

        # Storage balance: per-storage, per-timestep
        for si, stor_id in enumerate(stor_ids.values):
            charge_fid = str(ds['charge_flow'].values[si])
            discharge_fid = str(ds['discharge_flow'].values[si])

            for ti, t in enumerate(time_list):
                t_next = te_list[ti + 1]
                dt_val = float(d.dt.sel(time=t).values)
                eta_c_val = float(ds['eta_c'].values[si, ti])
                eta_d_val = float(ds['eta_d'].values[si, ti])
                loss_val = float(ds['loss'].values[si, ti])

                loss_factor = 1.0 - loss_val * dt_val
                charge_factor = eta_c_val * dt_val
                discharge_factor = dt_val / eta_d_val

                cs_next = self.charge_state.sel(storage=stor_id, time_extra=t_next)
                cs_curr = self.charge_state.sel(storage=stor_id, time_extra=t)
                charge = self.flow_rate.sel(flow=charge_fid, time=t)
                discharge = self.flow_rate.sel(flow=discharge_fid, time=t)

                self.m.add_constraints(
                    cs_next == cs_curr * loss_factor + charge * charge_factor - discharge * discharge_factor,
                    name=f'storage_bal_{stor_id}_{ti}',
                )

            # Initial/cyclic condition
            is_cyclic = bool(ds['cyclic'].values[si])
            if is_cyclic:
                cs_first = self.charge_state.sel(storage=stor_id, time_extra=te_list[0])
                cs_last = self.charge_state.sel(storage=stor_id, time_extra=te_list[-1])
                self.m.add_constraints(cs_last == cs_first, name=f'cs_cyclic_{stor_id}')
            else:
                initial = float(ds['initial_charge'].values[si])
                cap_val = float(cap.values[si]) if not np.isnan(cap.values[si]) else 1e9
                initial_abs = initial * cap_val if initial <= 1.0 else initial
                cs_first = self.charge_state.sel(storage=stor_id, time_extra=te_list[0])
                self.m.add_constraints(cs_first == initial_abs, name=f'cs_init_{stor_id}')

    def _set_objective(self) -> None:
        """Objective: min Φ_{k*} where k* is the effect with is_objective=True."""
        obj_effect = self.data.effects.attrs['objective_effect']
        self.m.add_objective(self.effect_total.sel(effect=obj_effect).sum())
