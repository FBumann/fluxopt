from __future__ import annotations

import polars as pl
import pyoframe as pf

from energysys.data import ModelData
from energysys.results import SolvedModel


class EnergySystemModel:
    def __init__(self, data: ModelData, solver: str = 'highs'):
        self.data = data
        self.m = pf.Model(solver)

    def build(self) -> None:
        self._create_flow_variables()
        self._create_bus_balance()
        self._create_converter_constraints()
        self._create_effects()
        self._create_storage()
        self._set_objective()

    def solve(self, silent: bool = True) -> SolvedModel:
        self.m.attr.Silent = silent
        self.m.optimize()
        return SolvedModel.from_model(self)

    def _create_flow_variables(self) -> None:
        d = self.data
        m = self.m

        # Create flow_rate variable indexed by (flow, time)
        index = d.flows.bounds.select('flow', 'time')
        m.flow_rate = pf.Variable(index, lb=0)

        # Apply bounds as constraints
        lb_param = pf.Param(d.flows.bounds.select('flow', 'time', pl.col('lb').alias('value')))
        ub_param = pf.Param(d.flows.bounds.select('flow', 'time', pl.col('ub').alias('value')))
        m.flow_lb = m.flow_rate >= lb_param
        m.flow_ub = m.flow_rate <= ub_param

        # Fix profile flows (lb == ub == fixed value)
        if len(d.flows.fixed) > 0:
            fixed_param = pf.Param(d.flows.fixed)
            m.flow_fix = m.flow_rate.drop_extras() == fixed_param

    def _create_bus_balance(self) -> None:
        d = self.data
        m = self.m

        if len(d.buses.flow_coefficients) == 0:
            return

        # Bus balance: sum(coeff * flow_rate) == 0 for each (bus, time)
        m.bus_balance = (pf.Param(d.buses.flow_coefficients) * m.flow_rate).sum('flow') == 0

    def _create_converter_constraints(self) -> None:
        d = self.data
        m = self.m

        if len(d.converters.flow_coefficients) == 0:
            return

        # Conversion: sum(coeff * flow_rate) == 0 for each (converter, eq_idx, time)
        m.conversion = (pf.Param(d.converters.flow_coefficients) * m.flow_rate).sum('flow') == 0

    def _create_effects(self) -> None:
        d = self.data
        m = self.m

        effects_index = d.effects.index
        timesteps = d.timesteps

        if len(effects_index) == 0:
            return

        # Effect per timestep variable
        m.effect_per_timestep = pf.Variable(effects_index, timesteps)

        # Effect total variable
        m.effect_total = pf.Variable(effects_index)

        # Track effects: effect_per_timestep = sum_flow(coeff * flow_rate * dt)
        if len(d.effects.flow_coefficients) > 0:
            contribution = pf.Param(d.effects.flow_coefficients) * m.flow_rate * pf.Param(d.dt)
            m.effect_tracking = m.effect_per_timestep == contribution.sum('flow')
        else:
            # No flow contributions - effects are zero
            m.effect_tracking = m.effect_per_timestep == 0

        # Total: effect_total = sum_time(effect_per_timestep)
        m.effect_total_eq = m.effect_total == m.effect_per_timestep.sum('time')

        # Effect bounds
        for row in d.effects.bounds.iter_rows(named=True):
            effect_label = row['effect']
            if row['min_total'] is not None:
                setattr(
                    m,
                    f'effect_min_total_{effect_label}',
                    m.effect_total.filter(effect=effect_label) >= row['min_total'],
                )
            if row['max_total'] is not None:
                setattr(
                    m,
                    f'effect_max_total_{effect_label}',
                    m.effect_total.filter(effect=effect_label) <= row['max_total'],
                )

        # Per-hour bounds
        if len(d.effects.time_bounds) > 0:
            for row in d.effects.time_bounds.iter_rows(named=True):
                effect_label = row['effect']
                time_label = row['time']
                if row['min_per_hour'] is not None:
                    setattr(
                        m,
                        f'effect_min_ph_{effect_label}_{time_label}',
                        m.effect_per_timestep.filter(effect=effect_label, time=time_label) >= row['min_per_hour'],
                    )
                if row['max_per_hour'] is not None:
                    setattr(
                        m,
                        f'effect_max_ph_{effect_label}_{time_label}',
                        m.effect_per_timestep.filter(effect=effect_label, time=time_label) <= row['max_per_hour'],
                    )

    def _create_storage(self) -> None:
        d = self.data
        m = self.m

        if len(d.storages.index) == 0:
            return

        storages_index = d.storages.index
        timesteps = d.timesteps
        time_list = timesteps['time'].to_list()

        # charge_state has one extra step (N+1 for N timesteps)
        steps = time_list + ['_end']
        steps_df = pl.DataFrame({'time': steps})

        m.charge_state = pf.Variable(storages_index, steps_df, lb=0)

        # Charge state bounds from capacity * relative bounds
        for row in d.storages.params.iter_rows(named=True):
            stor = row['storage']
            cap = row['capacity']
            if cap is not None:
                # Set upper bound on charge_state
                setattr(
                    m,
                    f'cs_cap_{stor}',
                    m.charge_state.filter(storage=stor) <= cap,
                )

        # Time-varying charge state bounds (relative to capacity)
        for row in d.storages.params.iter_rows(named=True):
            stor = row['storage']
            cap = row['capacity'] or 1e9
            time_params = d.storages.time_params.filter(pl.col('storage') == stor)
            for tp_row in time_params.iter_rows(named=True):
                t = tp_row['time']
                cs_lb = tp_row['cs_lb'] * cap
                cs_ub = tp_row['cs_ub'] * cap
                if cs_lb > 0:
                    setattr(
                        m,
                        f'cs_lb_{stor}_{t}',
                        m.charge_state.filter(storage=stor, time=t) >= cs_lb,
                    )
                if cs_ub < cap:
                    setattr(
                        m,
                        f'cs_ub_{stor}_{t}',
                        m.charge_state.filter(storage=stor, time=t) <= cs_ub,
                    )

        # Storage balance: cs[t+1] = cs[t] * (1 - loss*dt) + charge*eta_c*dt - discharge/eta_d*dt
        flow_map = d.storages.flow_map

        for row in flow_map.iter_rows(named=True):
            stor = row['storage']
            charge_flow = row['charge_flow']
            discharge_flow = row['discharge_flow']

            stor_params = d.storages.params.filter(pl.col('storage') == stor).row(0, named=True)
            time_params = d.storages.time_params.filter(pl.col('storage') == stor)

            # Build per-timestep balance constraints
            for i, t in enumerate(time_list):
                tp = time_params.filter(pl.col('time') == t).row(0, named=True)
                dt_row = d.dt.filter(pl.col('time') == t).row(0, named=True)
                dt_val = dt_row['dt']

                loss_factor = 1.0 - tp['loss'] * dt_val
                charge_factor = tp['eta_c'] * dt_val
                discharge_factor = dt_val / tp['eta_d']

                t_next = time_list[i + 1] if i + 1 < len(time_list) else '_end'

                cs_next = m.charge_state.pick(storage=stor, time=t_next)
                cs_curr = m.charge_state.pick(storage=stor, time=t)
                charge = m.flow_rate.pick(flow=charge_flow, time=t)
                discharge = m.flow_rate.pick(flow=discharge_flow, time=t)

                setattr(
                    m,
                    f'storage_bal_{stor}_{t}',
                    cs_next == cs_curr * loss_factor + charge * charge_factor - discharge * discharge_factor,
                )

            # Initial condition
            initial = stor_params['initial_charge']
            cyclic = stor_params['cyclic']

            if cyclic:
                # Last step == first step
                cs_first = m.charge_state.pick(storage=stor, time=time_list[0])
                cs_last = m.charge_state.pick(storage=stor, time='_end')
                setattr(m, f'cs_cyclic_{stor}', cs_last == cs_first)
            else:
                cap = stor_params['capacity'] or 1e9
                initial_abs = initial * cap if isinstance(initial, float) and initial <= 1.0 else initial
                cs_first = m.charge_state.pick(storage=stor, time=time_list[0])
                setattr(m, f'cs_init_{stor}', cs_first == float(initial_abs))

    def _set_objective(self) -> None:
        d = self.data
        m = self.m

        obj_effect = d.effects.objective_effect
        m.minimize = m.effect_total.filter(effect=obj_effect).sum()
