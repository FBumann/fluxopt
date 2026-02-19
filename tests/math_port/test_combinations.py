"""Mathematical correctness tests for COMBINATIONS of features.

These tests verify that status parameters, investment sizing, and effects
work correctly when combined — catching interaction bugs that single-feature
tests miss.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port, Sizing, Status

from .conftest import _waste, ts


class TestPiecewiseWithInvestment:
    """Tests combining PiecewiseConversion with InvestParameters."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_conversion_with_investment_sizing(self, optimize):
        """Proves: PiecewiseConversion and InvestParameters on the same converter."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_invest_cost_with_optional_skip(self, optimize):
        """Proves: Piecewise investment cost function works with optional investment."""


class TestPiecewiseWithStatus:
    """Tests combining PiecewiseConversion with StatusParameters."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_nonlinear_conversion_with_startup_cost(self, optimize):
        """Proves: PiecewiseConversion and startup costs interact correctly."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_minimum_load_with_status(self, optimize):
        """Proves: Piecewise gap enforces minimum load, interacting with status."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_no_zero_point_with_status(self, optimize):
        """Proves: Piecewise WITHOUT off-state piece interacts with StatusParameters."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_piecewise_no_zero_point_startup_cost(self, optimize):
        """Proves: Piecewise without zero point + startup cost work together."""


class TestPiecewiseThreeSegments:
    """Tests for piecewise conversion with 3+ segments."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_three_segment_piecewise(self, optimize):
        """Proves: 3-segment PiecewiseConversion correctly selects the optimal segment."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_three_segment_low_load_selection(self, optimize):
        """Proves: With 3 segments, low demand correctly uses segment 1."""

    @pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
    def test_three_segment_mid_load_selection(self, optimize):
        """Proves: With 3 segments, mid demand correctly uses segment 2."""


class TestStatusWithEffects:
    """Tests for StatusParameters contributing to non-standard effects."""

    def test_startup_cost_on_co2_effect(self, optimize):
        """Proves: effects_per_startup can contribute to a non-cost effect (CO2),
        and that this correctly interacts with effect constraints.

        CO2 capped at maximum_total=60. Boiler startup emits 50kg CO2.
        Demand=[0,20,0,20] → 2 startups = 100kg CO2. Exceeds cap!
        Optimizer must reduce startups by keeping boiler running continuously.

        Sensitivity: Without CO2 cap, 2 startups optimal. With cap=60, forced to 1 startup.
        """
        result = optimize(
            timesteps=ts(4),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_total=60),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                _waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior=[0],
                        status=Status(effects_per_startup={'CO2': 50}),
                    ),
                ),
            ],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 60.0 + 1e-5
        # Verify only 1 startup (continuous operation)
        on = result.solution['flow--on'].sel(flow='Boiler(Heat)').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup, got {startups}: on={on}'

    def test_effects_per_active_hour_on_multiple_effects(self, optimize):
        """Proves: effects_per_running_hour can contribute to multiple effects simultaneously.

        Boiler with effects_per_running_hour={'cost': 10, 'CO2': 5}.
        Demand=[20,20]. Boiler on 2 hours.

        Sensitivity: Without effects_per_running_hour, costs=40, CO2=0.
        With it, costs = 40 + 2*10 = 60, CO2 = 2*5 = 10.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2'),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.1,
                        status=Status(effects_per_running_hour={'cost': 10, 'CO2': 5}),
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 10.0, rtol=1e-5)


class TestInvestWithRelativeMinimum:
    """Tests combining Sizing with relative_minimum."""

    def test_invest_sizing_respects_relative_minimum(self, optimize):
        """Proves: relative_minimum on an invested flow forces the boiler OFF at
        low-demand timesteps, requiring expensive backup.

        Sensitivity: Without relative_minimum: size=50, ON both hours, fuel=55, total=80.
        With it: size=50, OFF at t=0, backup=5*10=50, total=125.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([5, 50])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                Port(
                    'Backup',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 10}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        relative_minimum=0.5,
                        size=Sizing(
                            min_size=0,
                            max_size=100,
                            mandatory=True,
                            effects_per_size={'cost': 0.5},
                        ),
                        status=Status(),
                    ),
                ),
            ],
        )
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 50.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 125.0, rtol=1e-4)
        # Verify boiler is OFF at t=0
        assert result.solution['flow--on'].sel(flow='Boiler(Heat)').values[0] < 0.5


class TestConversionWithTimeVaryingEffects:
    """Tests for conversion factors with time-varying effects."""

    def test_time_varying_effects_per_flow_hour(self, optimize):
        """Proves: Time-varying effects_per_flow_hour correctly applies different rates
        per timestep when combined with conversion.

        Boiler eta=0.5. Gas cost = [1, 3]. Demand=[20, 10].
        t=0: fuel=40, cost=40. t=1: fuel=20, cost=60. Total=100.

        Sensitivity: If mean(2) were used: cost=120. Only per-timestep gives 100.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 10])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': np.array([1, 3])}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_effects_per_flow_hour_with_dual_output_conversion(self, optimize):
        """Proves: effects_per_flow_hour applied to individual flows of a multi-output
        converter correctly accumulates effects for each flow independently.

        CHP: fuel→heat+elec. Fuel costs 1€/kWh, elec earns -2€/kWh.
        CO2: fuel emits 0.5 kg/kWh, elec avoids -0.3 kg/kWh.
        Demand=50 heat per timestep.

        Sensitivity: Total uniquely determined by conversion factors + effects.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Elec'), Bus('Gas')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2'),
            ],
            ports=[
                Port(
                    'HeatDemand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                    ],
                ),
                Port(
                    'ElecGrid',
                    exports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': -2, 'CO2': -0.3}),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.5}),
                    ],
                ),
            ],
            converters=[
                Converter.chp(
                    'CHP',
                    eta_th=0.5,
                    eta_el=0.4,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat'),
                    electrical_flow=Flow(bus='Elec'),
                ),
            ],
        )
        # Per ts: fuel=100, elec=40. costs: 100-80=20. CO2: 50-12=38. Total: costs=40, CO2=76.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 76.0, rtol=1e-5)


@pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
class TestPiecewiseInvestWithStatus:
    """Tests combining piecewise investment costs with status parameters."""

    def test_piecewise_invest_with_startup_cost(self, optimize):
        """Proves: Piecewise investment cost and startup cost work together."""


class TestStatusWithMultipleConstraints:
    """Tests combining multiple status parameters on the same flow."""

    @pytest.mark.skip(reason='startup_limit not supported in fluxopt')
    def test_startup_limit_with_max_downtime(self, optimize):
        """Proves: startup_limit and max_downtime interact correctly."""

    def test_min_uptime_with_min_downtime(self, optimize):
        """Proves: min_uptime and min_downtime together force a regular on/off pattern.

        Boiler: min_uptime=2, min_downtime=2, prior=[0].
        Demand=[20]*6. Backup at eta=0.5.

        Sensitivity: Without these constraints, boiler could run all 6 hours.
        With constraints, forced into block pattern → backup needed for off blocks.
        """
        result = optimize(
            timesteps=ts(6),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20, 20, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior=[0],
                        status=Status(min_uptime=2, min_downtime=2),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        on = result.solution['flow--on'].sel(flow='CheapBoiler(Heat)').values

        # Verify min_uptime: each on-block is ≥2 hours
        on_block_len = 0
        for i, s in enumerate(on):
            if s > 0.5:
                on_block_len += 1
            else:
                if on_block_len > 0:
                    assert on_block_len >= 2, f'min_uptime violated: on-block of {on_block_len} at t<{i}: on={on}'
                on_block_len = 0

        # Verify min_downtime: each off-block is ≥2 hours (within horizon)
        off_block_len = 0
        for i, s in enumerate(on):
            if s < 0.5:
                off_block_len += 1
            else:
                if 0 < off_block_len < 2 and i - off_block_len > 0:
                    assert off_block_len >= 2, f'min_downtime violated: off-block of {off_block_len} at t<{i}: on={on}'
                off_block_len = 0

        # Total cost between all-cheap and all-backup
        assert result.effect_totals.sel(effect='cost').item() > 120 - 1e-5
        assert result.effect_totals.sel(effect='cost').item() < 240 + 1e-5


class TestEffectsWithConversion:
    """Tests for effects interacting with conversion and other constraints."""

    @pytest.mark.skip(reason='share_from_periodic not supported in fluxopt')
    def test_effect_share_with_investment(self, optimize):
        """Proves: share_from_periodic works correctly with investment costs."""

    def test_effect_maximum_with_status_contribution(self, optimize):
        """Proves: Effect maximum_total correctly accounts for contributions from
        StatusParameters (effects_per_startup) when constraining.

        CO2 has maximum_total=20. Boiler startup emits 15 kg CO2.
        Fuel emits 0.1 kg CO2/kWh. Demand=[0,10,0,10].
        2 startups = 30 kg CO2 (exceeds cap). Forced to 1 startup.

        Sensitivity: Without CO2 cap, 2 startups → CO2=30+fuel. With cap=20, ≤1 startup.
        """
        result = optimize(
            timesteps=ts(4),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_total=20),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 10, 0, 10])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.1}),
                    ],
                ),
                _waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior=[0],
                        status=Status(effects_per_startup={'CO2': 15}),
                    ),
                ),
            ],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 20.0 + 1e-5


class TestInvestWithEffects:
    """Tests combining investment with effect constraints."""

    @pytest.mark.skip(reason='maximum_periodic not supported in fluxopt')
    def test_invest_per_size_on_non_cost_effect(self, optimize):
        """Proves: effects_per_size can contribute to a non-cost effect."""
