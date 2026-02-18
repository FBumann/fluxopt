"""Mathematical correctness tests for conversion & efficiency."""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port

from .conftest import ts


class TestConversionEfficiency:
    def test_boiler_efficiency(self, optimize):
        """Proves: Boiler applies Q_fu = Q_th / eta to compute fuel consumption.

        Sensitivity: If eta were ignored (treated as 1.0), cost would be 40 instead of 50.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 20, 10])),
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
                    thermal_efficiency=0.8,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        # fuel = (10+20+10)/0.8 = 50, cost@1€/kWh = 50
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 50.0, rtol=1e-5)

    def test_variable_efficiency(self, optimize):
        """Proves: Boiler accepts a time-varying efficiency array and applies it per timestep.

        Sensitivity: If a scalar mean (0.75) were used, cost=26.67. If only the first
        value (0.5) were broadcast, cost=40. Only per-timestep application yields 30.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
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
                    thermal_efficiency=np.array([0.5, 1.0]),
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        # fuel = 10/0.5 + 10/1.0 = 30
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 30.0, rtol=1e-5)

    def test_chp_dual_output(self, optimize):
        """Proves: CHP conversion factors for both thermal and electrical output are correct.
        fuel = Q_th / eta_th, P_el = fuel * eta_el. Revenue from P_el reduces total cost.

        Sensitivity: If electrical output were zero (eta_el broken), cost=200 instead of 40.
        If eta_th were wrong (e.g. 1.0), fuel=100 and cost changes to -60.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Elec'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
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
                        Flow(bus='Elec', effects_per_flow_hour={'cost': -2}),
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
                Converter.chp(
                    'CHP',
                    eta_el=0.4,
                    eta_th=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    electrical_flow=Flow(bus='Elec'),
                    thermal_flow=Flow(bus='Heat'),
                ),
            ],
        )
        # Per timestep: fuel = 50/0.5 = 100, elec = 100*0.4 = 40
        # Per timestep cost = 100*1 - 40*2 = 20, total = 2*20 = 40
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)
