from __future__ import annotations

import pytest

from fluxopt import Bus, Effect, Flow, Port, solve


class TestEffects:
    def test_single_cost_effect(self, timesteps_3):
        """Total cost = sum(rate * coeff * dt)."""
        demand = [50.0, 80.0, 60.0]
        sink_flow = Flow('sink', bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow('source', bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        expected = sum(d * 0.04 for d in demand)
        assert result.objective_value == pytest.approx(expected, abs=1e-6)

    def test_multiple_effects(self, timesteps_3):
        """Track cost and CO2 simultaneously, minimize cost."""
        sink_flow = Flow(
            'sink',
            bus='elec',
            size=100,
            fixed_relative_profile=[0.5, 0.8, 0.6],
        )
        source_flow = Flow(
            'source',
            bus='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True), Effect('co2', unit='kg')],
            components=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        demand_total = 50 + 80 + 60
        expected_cost = demand_total * 0.04
        expected_co2 = demand_total * 0.5

        assert result.objective_value == pytest.approx(expected_cost, abs=1e-6)
        co2_total = result.effects.filter(result.effects['effect'] == 'co2')['total'][0]
        assert co2_total == pytest.approx(expected_co2, abs=1e-6)

    def test_effect_maximum_total(self, timesteps_3):
        """Effect max_total constraint limits total emissions."""
        sink_flow = Flow('sink', bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        # Two sources with different cost/co2 tradeoffs
        cheap_dirty = Flow('cheap', bus='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
        expensive_clean = Flow('clean', bus='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

        co2_limit = 100.0  # demand_total = 190, so can't use all cheap
        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True), Effect('co2', maximum_total=co2_limit)],
            components=[
                Port('cheap_src', imports=[cheap_dirty]),
                Port('clean_src', imports=[expensive_clean]),
                Port('demand', exports=[sink_flow]),
            ],
        )

        co2_total = result.effects.filter(result.effects['effect'] == 'co2')['total'][0]
        assert co2_total <= co2_limit + 1e-6

    def test_time_varying_cost(self, timesteps_3):
        """Time-varying costs are tracked correctly."""
        prices = [0.02, 0.08, 0.04]
        sink_flow = Flow('sink', bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        source_flow = Flow('source', bus='elec', size=200, effects_per_flow_hour={'cost': prices})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        expected = 50 * 0.02 + 50 * 0.08 + 50 * 0.04
        assert result.objective_value == pytest.approx(expected, abs=1e-6)
