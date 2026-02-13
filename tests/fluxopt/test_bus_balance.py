from __future__ import annotations

import pytest

from fluxopt import Bus, Effect, Flow, Port, solve


class TestBusBalance:
    def test_source_matches_fixed_demand(self, timesteps_3):
        """Source flow must match fixed demand through bus balance."""
        demand = [50.0, 80.0, 60.0]
        sink_flow = Flow('sink(elec)', bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow('source(elec)', bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        source_rates = result.flow_rate('source(elec)')['value'].to_list()
        for actual, expected in zip(source_rates, demand, strict=False):
            assert actual == pytest.approx(expected, abs=1e-6)

    def test_cost_tracking(self, timesteps_3):
        """Total cost = sum(flow_rate * cost_per_hour * dt)."""
        sink_flow = Flow('sink(elec)', bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow('source(elec)', bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[Port('grid', imports=[source_flow]), Port('demand', exports=[sink_flow])],
        )

        expected_cost = (50 + 80 + 60) * 0.04
        assert result.objective_value == pytest.approx(expected_cost, abs=1e-6)

    def test_two_sources_one_bus(self, timesteps_3):
        """Optimizer picks cheaper source."""
        demand_flow = Flow('demand', bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        cheap_flow = Flow('cheap', bus='elec', size=200, effects_per_flow_hour={'cost': 0.02})
        expensive_flow = Flow('expensive', bus='elec', size=200, effects_per_flow_hour={'cost': 0.10})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Port('cheap_src', imports=[cheap_flow]),
                Port('exp_src', imports=[expensive_flow]),
                Port('demand', exports=[demand_flow]),
            ],
        )

        cheap_rates = result.flow_rate('cheap')['value'].to_list()
        exp_rates = result.flow_rate('expensive')['value'].to_list()
        for rate in cheap_rates:
            assert rate == pytest.approx(50.0, abs=1e-6)
        for rate in exp_rates:
            assert rate == pytest.approx(0.0, abs=1e-6)
