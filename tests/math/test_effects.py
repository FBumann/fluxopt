from __future__ import annotations

import pytest

from fluxopt import Bus, Effect, Flow, Port, Sizing, optimize


class TestEffects:
    def test_single_cost_effect(self, timesteps_3):
        """Total cost = sum(rate * coeff * dt)."""
        demand = [50.0, 80.0, 60.0]
        sink_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        expected = sum(d * 0.04 for d in demand)
        assert result.objective == pytest.approx(expected, abs=1e-6)

    def test_multiple_effects(self, timesteps_3):
        """Track cost and CO2 simultaneously, minimize cost."""
        sink_flow = Flow(
            bus='elec',
            size=100,
            fixed_relative_profile=[0.5, 0.8, 0.6],
        )
        source_flow = Flow(
            bus='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True), Effect('co2', unit='kg')],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        demand_total = 50 + 80 + 60
        expected_cost = demand_total * 0.04
        expected_co2 = demand_total * 0.5

        assert result.objective == pytest.approx(expected_cost, abs=1e-6)
        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total == pytest.approx(expected_co2, abs=1e-6)

    def test_effect_maximum_total(self, timesteps_3):
        """Effect max_total constraint limits total emissions."""
        sink_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        # Two sources with different cost/co2 tradeoffs
        cheap_dirty = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
        expensive_clean = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

        co2_limit = 100.0  # demand_total = 190, so can't use all cheap
        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True), Effect('co2', maximum_total=co2_limit)],
            ports=[
                Port('cheap_src', imports=[cheap_dirty]),
                Port('clean_src', imports=[expensive_clean]),
                Port('demand', exports=[sink_flow]),
            ],
        )

        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total <= co2_limit + 1e-6

    def test_time_varying_cost(self, timesteps_3):
        """Time-varying costs are tracked correctly."""
        prices = [0.02, 0.08, 0.04]
        sink_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': prices})

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        expected = 50 * 0.02 + 50 * 0.08 + 50 * 0.04
        assert result.objective == pytest.approx(expected, abs=1e-6)


class TestContributionFrom:
    def test_contribution_from_self_reference_raises(self, timesteps_3):
        """Self-referencing contribution_from raises ValueError."""
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match='cannot reference itself'):
            optimize(
                timesteps=timesteps_3,
                buses=[Bus('elec')],
                effects=[Effect('cost', is_objective=True, contribution_from={'cost': 0.5})],
                ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
            )

    def test_contribution_from_circular_raises(self, timesteps_3):
        """Circular contribution_from dependency raises ValueError."""
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match='Circular contribution_from dependency'):
            optimize(
                timesteps=timesteps_3,
                buses=[Bus('elec')],
                effects=[
                    Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                    Effect('co2', unit='kg', contribution_from={'cost': 0.01}),
                ],
                ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
            )

    def test_contribution_from_carbon_pricing(self, timesteps_3):
        """CO2 at 0.5 kg/MWh, carbon price 50 €/t → cost includes CO2 * 50."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(
            bus='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 190 MWh
        direct_cost = total_energy * 0.04
        co2_total = total_energy * 0.5
        co2_cost = co2_total * 50
        expected_cost = direct_cost + co2_cost
        assert result.objective == pytest.approx(expected_cost, abs=1e-6)

    def test_contribution_from_source_unaffected(self, timesteps_3):
        """Source effect total is unchanged by contribution_from on target."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(
            bus='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        total_energy = sum(demand)
        expected_co2 = total_energy * 0.5
        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total == pytest.approx(expected_co2, abs=1e-6)

    def test_contribution_from_transitive(self, timesteps_3):
        """PE → CO2 → cost chain: transitivity via variable chaining."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(
            bus='elec',
            size=200,
            effects_per_flow_hour={'pe': 2.0},  # 2 kWh_PE per kWh_el
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg', contribution_from={'pe': 0.3}),  # 0.3 kg_CO2/kWh_PE
                Effect('pe', unit='kWh'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 190 MWh
        pe_total = total_energy * 2.0  # 380
        co2_total = pe_total * 0.3  # 114
        cost_total = co2_total * 50  # 5700

        assert float(result.effect_totals.sel(effect='pe').values) == pytest.approx(pe_total, abs=1e-6)
        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-6)
        assert result.objective == pytest.approx(cost_total, abs=1e-6)

    def test_contribution_from_per_hour(self, timesteps_3):
        """Time-varying carbon price overrides scalar for per-timestep."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(
            bus='elec',
            size=200,
            effects_per_flow_hour={'co2': 0.5},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        carbon_prices = [40.0, 50.0, 60.0]
        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect(
                    'cost',
                    is_objective=True,
                    contribution_from={'co2': 50},  # scalar for invest
                    contribution_from_per_hour={'co2': carbon_prices},  # time-varying for ops
                ),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        # per_ts[co2, t] = demand[t] * 0.5 (dt=1)
        # per_ts[cost, t] = carbon_price[t] * per_ts[co2, t]
        # total[cost] = sum(per_ts[cost, t])  (no invest here)
        expected = sum(d * 0.5 * p for d, p in zip(demand, carbon_prices, strict=True))
        assert result.objective == pytest.approx(expected, abs=1e-6)

    def test_contribution_from_investment(self, timesteps_3):
        """Sizing CO2 priced into cost via contribution_from."""
        demand = [50.0, 50.0, 50.0]
        source = Flow(
            bus='elec',
            size=Sizing(min_size=50, max_size=200, mandatory=True, effects_per_size={'co2': 10}),
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 150 MWh
        # Solver picks min feasible size = 50 MW (demand = 50 MW)
        invest_size = 50.0
        direct_cost = total_energy * 0.04
        op_co2 = total_energy * 0.5
        invest_co2 = invest_size * 10
        co2_total = op_co2 + invest_co2
        # cost = direct_cost + 50 * (op_co2 per_ts contribution summed) + 50 * invest_co2
        cost_total = direct_cost + op_co2 * 50 + invest_co2 * 50

        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-6)
        assert result.objective == pytest.approx(cost_total, abs=1e-6)

    def test_contribution_from_investment_transitive(self, timesteps_3):
        """PE → CO2 → cost: 3-level chain with investment costs propagates correctly."""
        demand = [50.0, 50.0, 50.0]
        source = Flow(
            bus='elec',
            size=Sizing(min_size=50, max_size=200, mandatory=True, effects_per_size={'pe': 5}),
            effects_per_flow_hour={'pe': 2.0},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg', contribution_from={'pe': 0.3}),
                Effect('pe', unit='kWh'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 150 MWh
        invest_size = 50.0

        # Temporal chain: pe_op=300, co2_op=90, cost_op=4500
        pe_op = total_energy * 2.0  # 300
        co2_op = pe_op * 0.3  # 90
        cost_op = co2_op * 50  # 4500

        # Periodic chain: pe_inv=250, co2_inv=75, cost_inv=3750
        pe_inv = invest_size * 5  # 250
        co2_inv = pe_inv * 0.3  # 75
        cost_inv = co2_inv * 50  # 3750

        pe_total = pe_op + pe_inv  # 550
        co2_total = co2_op + co2_inv  # 165
        cost_total = cost_op + cost_inv  # 8250

        assert float(result.effect_totals.sel(effect='pe').values) == pytest.approx(pe_total, abs=1e-4)
        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-4)
        assert result.objective == pytest.approx(cost_total, abs=1e-4)
