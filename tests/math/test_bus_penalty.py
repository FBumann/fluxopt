from __future__ import annotations

import numpy as np
import pytest
from conftest import ts

from fluxopt import Bus, Effect, Flow, Port, optimize


class TestBusPenaltyShortage:
    def test_shortage_produces_penalty(self):
        """Demand exceeds supply capacity → shortage slack > 0, penalty in objective."""
        # Source can only provide 30 MW, demand is 50 MW → 20 MW shortage each step
        source = Flow(bus='elec', size=30, effects_per_flow_hour={'cost': 1})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=1000)],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        shortage = result.bus_shortage.sel(bus='elec').values
        surplus = result.bus_surplus.sel(bus='elec').values

        # Shortage should be 20 MW at each timestep
        np.testing.assert_allclose(shortage, [20, 20, 20], atol=1e-4)
        np.testing.assert_allclose(surplus, [0, 0, 0], atol=1e-4)

        # Objective = source cost (30*1*3) + penalty (20*1000*3)
        expected = 30 * 1 * 3 + 20 * 1000 * 3
        assert result.objective == pytest.approx(expected, abs=1e-2)


class TestBusPenaltySurplus:
    def test_surplus_produces_penalty(self):
        """Fixed supply exceeds demand → surplus slack > 0."""
        # Source is fixed at 80 MW, demand is 50 MW → 30 MW surplus each step
        source = Flow(bus='elec', size=80, fixed_relative_profile=[1.0, 1.0, 1.0])
        sink = Flow(bus='elec', size=50, effects_per_flow_hour={'cost': 0})

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=500)],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        surplus = result.bus_surplus.sel(bus='elec').values
        shortage = result.bus_shortage.sel(bus='elec').values

        np.testing.assert_allclose(surplus, [30, 30, 30], atol=1e-4)
        np.testing.assert_allclose(shortage, [0, 0, 0], atol=1e-4)

        # Penalty cost = 30 * 500 * 3 = 45000
        penalty_total = float(result.effect_totals.sel(effect='penalty').values)
        assert penalty_total == pytest.approx(30 * 500 * 3, abs=1e-2)


class TestBusPenaltyBalanced:
    def test_balanced_zero_penalty(self):
        """When supply matches demand, slacks are zero and penalty is zero."""
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=1000)],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        surplus = result.bus_surplus.sel(bus='elec').values
        shortage = result.bus_shortage.sel(bus='elec').values

        np.testing.assert_allclose(surplus, [0, 0, 0], atol=1e-4)
        np.testing.assert_allclose(shortage, [0, 0, 0], atol=1e-4)

        # Objective = flow cost only, no penalty
        expected_cost = (50 + 80 + 60) * 0.04
        assert result.objective == pytest.approx(expected_cost, abs=1e-4)


class TestBusPenaltyMixed:
    def test_mixed_hard_and_soft_buses(self):
        """One bus penalized, one hard-balanced. Hard bus still enforces strict balance."""
        # Electricity bus: hard balanced, supply matches demand
        elec_source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        elec_sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        # Heat bus: penalized, source can't meet demand → shortage
        heat_source = Flow(bus='heat', size=20, effects_per_flow_hour={'cost': 1})
        heat_sink = Flow(bus='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec'), Bus('heat', imbalance_penalty=500)],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port('elec_src', imports=[elec_source]),
                Port('elec_demand', exports=[elec_sink]),
                Port('heat_src', imports=[heat_source]),
                Port('heat_demand', exports=[heat_sink]),
            ],
        )

        # Heat bus shortage: 50 - 20 = 30 MW per step
        heat_shortage = result.bus_shortage.sel(bus='heat').values
        np.testing.assert_allclose(heat_shortage, [30, 30, 30], atol=1e-4)

        # Electricity bus should have zero slack (hard balanced)
        elec_surplus = result.bus_surplus.sel(bus='elec').values
        elec_shortage = result.bus_shortage.sel(bus='elec').values
        np.testing.assert_allclose(elec_surplus, [0, 0, 0], atol=1e-4)
        np.testing.assert_allclose(elec_shortage, [0, 0, 0], atol=1e-4)


class TestPenaltyEffectAutoCreation:
    def test_auto_creates_penalty_effect(self):
        """Penalty effect is auto-created when not in user effect list."""
        source = Flow(bus='elec', size=30, effects_per_flow_hour={'cost': 1})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=100)],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        # Penalty effect should exist in results
        assert 'penalty' in result.effect_totals.coords['effect'].values

    def test_user_defined_penalty_effect_bounds(self):
        """User-defined Effect('penalty', maximum_total=X) → bounds respected.

        Source cost=100 is expensive, penalty=10 is cheap. Without cap the
        optimizer would take all demand as shortage. Cap limits penalty total.
        """
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 100})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        # Uncapped penalty would be 50*10*3 = 1500, but cap at 1000
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=10)],
            effects=[Effect('cost', is_objective=True), Effect('penalty', maximum_total=1000)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        penalty_total = float(result.effect_totals.sel(effect='penalty').values)
        assert penalty_total <= 1000 + 1e-4


class TestPenaltyAsRegularEffect:
    def test_effects_per_flow_hour_with_penalty(self):
        """Penalty effect can be used in effects_per_flow_hour like any other effect."""
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'penalty': 50})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec', imbalance_penalty=1000)],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        # No shortage/surplus expected (supply can meet demand)
        # Penalty total = flow contribution: 50 * 50 * 3 = 7500 (from effects_per_flow_hour)
        penalty_total = float(result.effect_totals.sel(effect='penalty').values)
        assert penalty_total == pytest.approx(50 * 50 * 3, abs=1e-2)


class TestNoPenalty:
    def test_no_penalty_buses_unchanged(self):
        """Without imbalance_penalty, behavior is unchanged from before."""
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[source]), Port('demand', exports=[sink])],
        )

        expected_cost = (50 + 80 + 60) * 0.04
        assert result.objective == pytest.approx(expected_cost, abs=1e-6)
        # No slack variables in solution
        assert 'bus--surplus' not in result.solution
        assert 'bus--shortage' not in result.solution
        # Penalty effect exists but is zero
        penalty_total = float(result.effect_totals.sel(effect='penalty').values)
        assert penalty_total == pytest.approx(0, abs=1e-6)
