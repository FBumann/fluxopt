"""Mathematical correctness tests for flow constraints."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port

from .conftest import ts, waste


class TestFlowConstraints:
    def test_relative_minimum(self, optimize):
        """Proves: relative_minimum enforces a minimum flow rate as a fraction of size
        when the unit is active (status=1).

        Boiler (size=100, relative_minimum=0.4). When on, must produce at least 40 kW.
        Demand=[30,30]. Since 30 < 40, boiler must produce 40 and excess is absorbed.

        Sensitivity: Without relative_minimum, boiler produces exactly 30 each timestep
        → cost=60. With relative_minimum=0.4, must produce 40 → cost=80.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow(bus='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100, relative_minimum=0.4),
                ),
            ],
        )
        # Must produce at least 40 (relative_minimum=0.4 * size=100)
        # cost = 2 * 40 = 80 (vs 60 without the constraint)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        # Verify flow rate is at least 40
        flow = result.flow_rate('Boiler(Heat)').values
        assert all(f >= 40.0 - 1e-5 for f in flow), f'Flow below relative_minimum: {flow}'

    def test_relative_maximum(self, optimize):
        """Proves: relative_maximum limits the maximum flow rate as a fraction of size.

        Source (size=100, relative_maximum=0.5). Max output = 50 kW.
        Demand=[60,60]. Can only get 50 from CheapSrc, rest from ExpensiveSrc.

        Sensitivity: Without relative_maximum, CheapSrc covers all 60 → cost=120.
        With relative_maximum=0.5, CheapSrc capped at 50 (2*50*1=100),
        ExpensiveSrc covers 10 each timestep (2*10*5=100) → total cost=200.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([60, 60])),
                    ],
                ),
                Port(
                    'CheapSrc',
                    imports=[
                        Flow(bus='Heat', size=100, relative_maximum=0.5, effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                Port(
                    'ExpensiveSrc',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 5}),
                    ],
                ),
            ],
        )
        # CheapSrc capped at 50 (relative_maximum=0.5 * size=100): 2 * 50 * 1 = 100
        # ExpensiveSrc covers remaining 10 each timestep: 2 * 10 * 5 = 100
        # Total = 200
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 200.0, rtol=1e-5)
        # Verify CheapSrc flow rate is at most 50
        flow = result.flow_rate('CheapSrc(Heat)').values
        assert all(f <= 50.0 + 1e-5 for f in flow), f'Flow above relative_maximum: {flow}'

    @pytest.mark.skip(reason='flow_hours constraint not supported in fluxopt')
    def test_flow_hours_max(self, optimize):
        """Proves: flow_hours_max limits the total cumulative flow-hours per period.

        CheapSrc (flow_hours_max=30). Total allowed = 30 kWh over horizon.
        Demand=[20,20,20] (total=60). Must split between cheap and expensive.

        Sensitivity: Without flow_hours_max, all from CheapSrc → cost=60.
        With flow_hours_max=30, CheapSrc limited to 30, ExpensiveSrc covers 30 → cost=180.
        """
        raise NotImplementedError  # TODO: implement flow_hours constraint

    @pytest.mark.skip(reason='flow_hours constraint not supported in fluxopt')
    def test_flow_hours_min(self, optimize):
        """Proves: flow_hours_min forces a minimum total cumulative flow-hours per period.

        ExpensiveSrc (flow_hours_min=40). Must produce at least 40 kWh total.
        Demand=[30,30] (total=60). CheapSrc is preferred but ExpensiveSrc must hit 40.

        Sensitivity: Without flow_hours_min, all from CheapSrc → cost=60.
        With flow_hours_min=40, ExpensiveSrc forced to produce 40 → cost=220.
        """
        raise NotImplementedError  # TODO: implement flow_hours constraint

    @pytest.mark.skip(reason='load_factor not supported in fluxopt')
    def test_load_factor_max(self, optimize):
        """Proves: load_factor_max limits utilization to (flow_hours) / (size * total_hours).

        CheapSrc (size=50, load_factor_max=0.5). Over 2 hours, max flow_hours = 50 * 2 * 0.5 = 50.
        Demand=[40,40] (total=80). CheapSrc capped at 50 total.

        Sensitivity: Without load_factor_max, CheapSrc covers 80 → cost=80.
        With load_factor_max=0.5, CheapSrc limited to 50, ExpensiveSrc covers 30 → cost=200.
        """
        raise NotImplementedError  # TODO: implement load_factor constraint

    @pytest.mark.skip(reason='load_factor not supported in fluxopt')
    def test_load_factor_min(self, optimize):
        """Proves: load_factor_min forces minimum utilization (flow_hours) / (size * total_hours).

        ExpensiveSrc (size=100, load_factor_min=0.3). Over 2 hours, min flow_hours = 100 * 2 * 0.3 = 60.
        Demand=[30,30] (total=60). ExpensiveSrc must produce at least 60.

        Sensitivity: Without load_factor_min, all from CheapSrc → cost=60.
        With load_factor_min=0.3, ExpensiveSrc forced to produce 60 → cost=300.
        """
        raise NotImplementedError  # TODO: implement load_factor constraint
