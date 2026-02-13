from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pyoframe as pf

from fluxopt.results import SolvedModel

if TYPE_CHECKING:
    from fluxopt.tables import ModelData


@dataclass
class _TemporalSource:
    name: str  # e.g., 'flow' → variable named 'contributions(flow)'
    index: pl.DataFrame  # sparse index, e.g., (flow, effect, time)
    expression: pf.Expression  # the contribution expression with native dims
    sum_dim: str  # dim to sum over → (effect, time), e.g., 'flow'


class FlowSystemModel:
    def __init__(self, data: ModelData, solver: str = 'highs'):
        self.data = data
        self.m = pf.Model(solver)
        self._temporal_sources: list[_TemporalSource] = []

    def add_temporal_contribution(
        self,
        name: str,
        index: pl.DataFrame,
        expression: pf.Expression,
        sum_dim: str,
    ) -> None:
        self._temporal_sources.append(_TemporalSource(name=name, index=index, expression=expression, sum_dim=sum_dim))

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
        """Create flow rate variables and apply bounds.

        Sized: P̄_f · p̲_{f,t} <= P_{f,t} <= P̄_f · p̄_{f,t}
        Fixed: P_{f,t} = P̄_f · π_{f,t}
        """
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

        # Register flow contributions to effects
        if len(d.flows.effect_coefficients) > 0:
            expr = pf.Param(d.flows.effect_coefficients) * m.flow_rate * pf.Param(d.dt)
            self.add_temporal_contribution(
                name='flow',
                index=d.flows.effect_coefficients.select('flow', 'effect', 'time'),
                expression=expr,
                sum_dim='flow',
            )

    def _create_bus_balance(self) -> None:
        """Bus balance: sum_f(coeff_{b,f} · P_{f,t}) = 0  for all b, t."""
        d = self.data
        m = self.m

        if len(d.buses.flow_coefficients) == 0:
            return

        # Bus balance: sum(coeff * flow_rate) == 0 for each (bus, time)
        m.bus_balance = (pf.Param(d.buses.flow_coefficients) * m.flow_rate).sum('flow') == 0

    def _create_converter_constraints(self) -> None:
        """Conversion: sum_f(a_f · P_{f,t}) = 0  for all converter, eq_idx, t."""
        d = self.data
        m = self.m

        if len(d.converters.flow_coefficients) == 0:
            return

        # Conversion: sum(coeff * flow_rate) == 0 for each (converter, eq_idx, time)
        m.conversion = (pf.Param(d.converters.flow_coefficients) * m.flow_rate).sum('flow') == 0

    def _create_effects(self) -> None:
        """Effect tracking via registered temporal sources.

        For each source: contributions(source) = expression  [source_dim, effect, time]
        Accumulate:      effect(per_timestep)  = sum over source dims via keep_extras()
        Temporal total:  effect(temporal)       = (effect(per_timestep) * weight).sum('time')
        Grand total:     effect(total)          = effect(temporal)
        """
        d = self.data
        m = self.m

        effects_index = d.effects.index
        timesteps = d.timesteps

        if len(effects_index) == 0:
            return

        # 1. Per-source contribution variables
        for src in self._temporal_sources:
            var_name = f'contributions({src.name})'
            var = pf.Variable(src.index)
            setattr(m, var_name, var)
            setattr(m, f'{var_name}_tracking', var == src.expression)

        # 2. Accumulate into effect(per_timestep)
        m.effect_per_timestep = pf.Variable(effects_index, timesteps)

        if self._temporal_sources:
            acc: pf.Expression | None = None
            for src in self._temporal_sources:
                var = getattr(m, f'contributions({src.name})')
                summed = var.sum(src.sum_dim)
                acc = summed if acc is None else acc.keep_extras() + summed.keep_extras()
            m.effect_per_timestep_tracking = m.effect_per_timestep == acc
        else:
            m.effect_per_timestep_tracking = m.effect_per_timestep == 0

        # 3. effect(temporal) = (effect(per_timestep) * weight).sum('time')
        m.effect_temporal = pf.Variable(effects_index)
        m.effect_temporal_eq = m.effect_temporal == (m.effect_per_timestep * pf.Param(d.weights)).sum('time')

        # 4. effect(total) = effect(temporal) (future: + effect(periodic))
        m.effect_total = pf.Variable(effects_index)
        m.effect_total_eq = m.effect_total == m.effect_temporal

        # 5. Bounds on effect(total)
        min_total_df = d.effects.bounds.filter(pl.col('min_total').is_not_null()).select(
            'effect', pl.col('min_total').alias('value')
        )
        max_total_df = d.effects.bounds.filter(pl.col('max_total').is_not_null()).select(
            'effect', pl.col('max_total').alias('value')
        )
        if len(min_total_df) > 0:
            m.effect_min_total = m.effect_total.drop_extras() >= pf.Param(min_total_df)
        if len(max_total_df) > 0:
            m.effect_max_total = m.effect_total.drop_extras() <= pf.Param(max_total_df)

        # 6. Per-hour bounds on effect(per_timestep)
        if len(d.effects.time_bounds_lb) > 0:
            m.effect_min_ph = m.effect_per_timestep.drop_extras() >= pf.Param(d.effects.time_bounds_lb)
        if len(d.effects.time_bounds_ub) > 0:
            m.effect_max_ph = m.effect_per_timestep.drop_extras() <= pf.Param(d.effects.time_bounds_ub)

    def _create_storage(self) -> None:
        """Storage dynamics: E_{s,t+1} = E_{s,t}(1 - δΔt) + P^c η^c Δt - P^d/η^d Δt."""
        d = self.data
        m = self.m

        if len(d.storages.index) == 0:
            return

        storages_index = d.storages.index
        time_list = d.timesteps['time'].to_list()

        # charge_state has one extra step (N+1 for N timesteps) — use charge_state_times
        m.charge_state = pf.Variable(storages_index, d.charge_state_times, lb=0)

        # Charge state capacity upper bound — vectorized (expand to all charge_state_times)
        cap_df = d.storages.params.filter(pl.col('capacity').is_not_null()).select(
            'storage', pl.col('capacity').alias('value')
        )
        if len(cap_df) > 0:
            cap_expanded = cap_df.join(d.charge_state_times, how='cross').select('storage', 'time', 'value')
            m.cs_cap = m.charge_state <= pf.Param(cap_expanded)

        # Time-varying charge state bounds — vectorized using pre-computed absolute bounds
        cs_lb_df = d.storages.cs_bounds.filter(pl.col('cs_lb') > 0).select(
            'storage', 'time', pl.col('cs_lb').alias('value')
        )
        cs_ub_active = d.storages.cs_bounds.join(d.storages.params.select('storage', 'capacity'), on='storage').filter(
            pl.col('cs_ub') < pl.col('capacity')
        )
        cs_ub_df = cs_ub_active.select('storage', 'time', pl.col('cs_ub').alias('value'))

        if len(cs_lb_df) > 0:
            m.cs_lb = m.charge_state >= pf.Param(cs_lb_df)
        if len(cs_ub_df) > 0:
            m.cs_ub = m.charge_state <= pf.Param(cs_ub_df)

        # Storage balance: cs[t+1] = cs[t] * (1 - loss*dt) + charge*eta_c*dt - discharge/eta_d*dt
        flow_map = d.storages.flow_map
        cs_time_list = d.charge_state_times['time'].to_list()

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

                t_next = cs_time_list[i + 1]

                cs_next = m.charge_state.pick(storage=stor, time=t_next)
                cs_curr = m.charge_state.pick(storage=stor, time=t)
                charge = m.flow_rate.pick(flow=charge_flow, time=t)
                discharge = m.flow_rate.pick(flow=discharge_flow, time=t)

                setattr(
                    m,
                    f'storage_bal_{stor}_{i}',
                    cs_next == cs_curr * loss_factor + charge * charge_factor - discharge * discharge_factor,
                )

            # Initial condition
            initial = stor_params['initial_charge']
            cyclic = stor_params['cyclic']

            if cyclic:
                # Last step == first step
                cs_first = m.charge_state.pick(storage=stor, time=time_list[0])
                cs_last = m.charge_state.pick(storage=stor, time=cs_time_list[-1])
                setattr(m, f'cs_cyclic_{stor}', cs_last == cs_first)
            else:
                cap = stor_params['capacity'] or 1e9
                initial_abs = initial * cap if isinstance(initial, float) and initial <= 1.0 else initial
                cs_first = m.charge_state.pick(storage=stor, time=time_list[0])
                setattr(m, f'cs_init_{stor}', cs_first == float(initial_abs))

    def _set_objective(self) -> None:
        """Objective: min Φ_{k*} where k* is the effect with is_objective=True."""
        d = self.data
        m = self.m

        obj_effect = d.effects.objective_effect
        m.minimize = m.effect_total.filter(effect=obj_effect).sum()
