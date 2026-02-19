"""Mathematical correctness tests for component-level features.

Tests for component-specific behavior including heat pumps. Component-level
StatusParameters, Transmission, and other advanced components are not yet
supported in fluxopt.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port

from .conftest import ts


class TestComponentStatus:
    """Tests for StatusParameters applied at the component level (not flow level)."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_startup_cost(self, optimize):
        """Proves: StatusParameters on LinearConverter applies startup cost when
        the component transitions to active."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(4)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter(
                'Boiler',
                inputs=[fx.Flow(bus='Gas', flow_id='fuel', size=100)],
                outputs=[fx.Flow(bus='Heat', flow_id='heat', size=100)],
                conversion_factors=[{'fuel': 1, 'heat': 1}],
                status_parameters=fx.StatusParameters(effects_per_startup=100),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 240.0, rtol=1e-5)

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_min_uptime(self, optimize):
        """Proves: min_uptime on component level forces the entire component
        to stay on for consecutive hours."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_active_hours_max(self, optimize):
        """Proves: active_hours_max on component level limits total operating hours."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_effects_per_active_hour(self, optimize):
        """Proves: effects_per_active_hour on component level adds cost per active hour."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_active_hours_min(self, optimize):
        """Proves: active_hours_min on component level forces minimum operating hours."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_max_uptime(self, optimize):
        """Proves: max_uptime on component level limits continuous operation."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_min_downtime(self, optimize):
        """Proves: min_downtime on component level prevents quick restart."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_max_downtime(self, optimize):
        """Proves: max_downtime on component level forces restart after idle."""

    @pytest.mark.skip(reason='component-level status not supported in fluxopt')
    def test_component_status_startup_limit(self, optimize):
        """Proves: startup_limit on component level caps number of startups."""


class TestTransmission:
    """Tests for Transmission component with losses and structural constraints."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_relative_losses(self, optimize):
        """Proves: relative_losses correctly reduces transmitted energy."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_absolute_losses(self, optimize):
        """Proves: absolute_losses adds fixed loss when transmission is active."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_bidirectional(self, optimize):
        """Proves: Bidirectional transmission allows flow in both directions."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_prevent_simultaneous_bidirectional(self, optimize):
        """Proves: prevent_simultaneous_flows_in_both_directions=True prevents both
        directions from being active at the same timestep."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_status_startup_cost(self, optimize):
        """Proves: StatusParameters on Transmission applies startup cost."""


class TestHeatPump:
    """Tests for HeatPump component with COP and source heat."""

    def test_heatpump_cop(self, optimize):
        """Proves: HeatPump splits demand into electrical and source via COP.

        demand=30/t, COP=3 → elec=10/t, source=20/t → cost=20.
        Sensitivity: Without source flow (COP=1 effective), cost=60.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Elec'), Bus('Env')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'cost': 1})]),
                Port('Environment', imports=[Flow(bus='Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=3.0,
                    electrical_flow=Flow(bus='Elec'),
                    source_flow=Flow(bus='Env'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

    def test_heatpump_variable_cop(self, optimize):
        """Proves: HeatPump accepts time-varying COP array.

        demand=20/t, COP=[2,4] → elec=[10,5]=15, source=[10,15] → cost=15.
        Sensitivity: If scalar cop=3 used, elec=13.33. Only time-varying gives 15.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Elec'), Bus('Env')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'cost': 1})]),
                Port('Environment', imports=[Flow(bus='Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=np.array([2.0, 4.0]),
                    electrical_flow=Flow(bus='Elec'),
                    source_flow=Flow(bus='Env'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 15.0, rtol=1e-5)


@pytest.mark.skip(reason='cooling_tower factory not supported in fluxopt')
class TestCoolingTower:
    """Tests for CoolingTower component."""

    def test_cooling_tower_specific_electricity(self, optimize):
        """Proves: CoolingTower correctly applies specific_electricity_demand."""


@pytest.mark.skip(reason='power2heat factory not supported in fluxopt')
class TestPower2Heat:
    """Tests for Power2Heat component."""

    def test_power2heat_efficiency(self, optimize):
        """Proves: Power2Heat applies thermal_efficiency to electrical input."""


@pytest.mark.skip(reason='heat_pump_with_source factory not supported in fluxopt')
class TestHeatPumpWithSource:
    """Tests for HeatPumpWithSource component with COP and heat source."""

    def test_heatpump_with_source_cop(self, optimize):
        """Proves: HeatPumpWithSource applies COP to compute electrical consumption,
        drawing the remainder from a heat source."""


@pytest.mark.skip(reason='prevent_simultaneous not supported in fluxopt')
class TestSourceAndSink:
    """Tests for SourceAndSink component."""

    def test_source_and_sink_prevent_simultaneous(self, optimize):
        """Proves: SourceAndSink with prevent_simultaneous_flow_rates=True prevents
        buying and selling in the same timestep."""
