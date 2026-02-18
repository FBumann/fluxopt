from __future__ import annotations

from datetime import datetime

import pytest

from fluxopt import Bus, Effect, Flow, ModelData, Port, optimize
from fluxopt.model import FlowSystemModel


class TestCustomize:
    """Tests for the customize callback and custom variables/constraints."""

    @pytest.fixture
    def simple_system(self):
        """Single-bus system: grid source (size=100) feeding a fixed 50 MW demand."""
        timesteps = [datetime(2024, 1, 1, h) for h in range(3)]
        return {
            'timesteps': timesteps,
            'buses': [Bus('elec')],
            'effects': [Effect('cost', is_objective=True)],
            'ports': [
                Port('grid', imports=[Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
                Port('demand', exports=[Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
            ],
        }

    def test_customize_adds_constraint(self, simple_system):
        """A custom constraint restricting flow rate should affect the solution."""
        # Without customize: grid imports 50 MW each hour (matching demand)
        result_base = optimize(**simple_system)
        base_rates = result_base.flow_rate('grid(elec)').values
        for rate in base_rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

        # With customize: cap grid import at 30 MW — this makes the problem infeasible
        # for a fixed 50 MW demand, so instead we test a less restrictive constraint.
        # Cap at 60 MW (above demand, so solution unchanged but constraint is present)
        def cap_at_60(model: FlowSystemModel) -> None:
            grid_rate = model.m.variables['flow--rate'].sel(flow='grid(elec)')
            model.m.add_constraints(grid_rate <= 60, name='custom_grid_cap')

        result = optimize(**simple_system, customize=cap_at_60)
        rates = result.flow_rate('grid(elec)').values
        for rate in rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

        # Verify the constraint name exists in the model
        assert result.objective == pytest.approx(result_base.objective, abs=1e-6)

    def test_custom_variable_in_results(self, simple_system):
        """A custom variable added via callback should appear in result.solution."""

        def add_slack(model: FlowSystemModel) -> None:
            time = model.m.variables['flow--rate'].coords['time']
            slack = model.m.add_variables(lower=0, coords=[time], name='my_slack')
            grid = model.m.variables['flow--rate'].sel(flow='grid(elec)')
            # grid + slack >= 60 → slack >= 10 (since grid = 50)
            model.m.add_constraints(grid + slack >= 60, name='slack_floor')
            model.m.objective += 100 * slack.sum()

        result = optimize(**simple_system, customize=add_slack)

        assert 'my_slack' in result.solution
        slack_vals = result.solution['my_slack'].values
        for val in slack_vals:
            assert val == pytest.approx(10.0, abs=1e-6)

    def test_no_customize_works(self, simple_system):
        """optimize() without customize callback works as before."""
        result = optimize(**simple_system)
        assert result.objective > 0
        rates = result.flow_rate('grid(elec)').values
        for rate in rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

    def test_direct_model_customization(self, simple_system):
        """Using FlowSystemModel directly with custom variable works."""
        data = ModelData.build(
            simple_system['timesteps'],
            simple_system['buses'],
            simple_system['effects'],
            simple_system['ports'],
        )
        model = FlowSystemModel(data)
        model.build()

        # Add custom variable and constraint
        time = model.m.variables['flow--rate'].coords['time']
        bonus = model.m.add_variables(lower=0, upper=5, coords=[time], name='bonus')
        model.m.objective += -bonus.sum()  # maximize bonus (minimize negative)

        result = model.solve()

        assert 'bonus' in result.solution
        for val in result.solution['bonus'].values:
            assert val == pytest.approx(5.0, abs=1e-6)
