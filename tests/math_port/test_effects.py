"""Mathematical correctness tests for effects & objective."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port

from .conftest import _waste, ts


class TestEffects:
    def test_effects_per_flow_hour(self, optimize):
        """Proves: effects_per_flow_hour correctly accumulates flow * rate for each
        named effect independently.

        Source has costs=2€/kWh and CO2=0.5kg/kWh. Total flow=30.

        Sensitivity: If effects_per_flow_hour were ignored, both effects=0. If only
        one effect were applied, the other would be wrong. Both values (60€, 15kg)
        are uniquely determined by the rates and total flow.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2'),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 20])),
                    ],
                ),
                Port(
                    'HeatSrc',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 2, 'CO2': 0.5}),
                    ],
                ),
            ],
        )
        # costs = (10+20)*2 = 60, CO2 = (10+20)*0.5 = 15
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 15.0, rtol=1e-5)

    @pytest.mark.skip(reason='share_from_temporal not supported in fluxopt')
    def test_share_from_temporal(self, optimize):
        """Proves: share_from_temporal correctly adds a weighted fraction of one effect's
        temporal sum into another effect's total.

        costs has share_from_temporal={'CO2': 0.5}. Direct costs=20, CO2=200.
        Shared portion: 0.5 * 200 = 100. Total costs = 20 + 100 = 120.

        Sensitivity: Without the share mechanism, costs=20 (6* less). The 120
        value is impossible without share_from_temporal working.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True, share_from_temporal={'CO2': 0.5})
        fs.add(
            fx.Bus('Heat'),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port(
                'HeatSrc', imports=[fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour={'costs': 1, 'CO2': 10})]
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 200.0, rtol=1e-5)

    def test_effect_maximum_total(self, optimize):
        """Proves: maximum_total on an effect constrains the optimizer to respect an
        upper bound on cumulative effect, forcing suboptimal dispatch.

        CO2 capped at 15kg. Dirty source: 1€+1kgCO2/kWh. Clean source: 10€+0kgCO2/kWh.
        Demand=20. Optimizer must split: 15 from Dirty + 5 from Clean.

        Sensitivity: Without the CO2 cap, all 20 from Dirty → cost=20 instead of 65.
        The 3.25* cost increase proves the constraint is binding.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_total=15),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                    ],
                ),
                Port(
                    'Dirty',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                Port(
                    'Clean',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 10, 'CO2': 0}),
                    ],
                ),
            ],
        )
        # Without CO2 limit: all from Dirty = 20€
        # With CO2 max=15: 15 from Dirty (15€), 5 from Clean (50€) → total 65€
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 65.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 15.0, rtol=1e-5)

    def test_effect_minimum_total(self, optimize):
        """Proves: minimum_total on an effect forces cumulative effect to reach at least
        the specified value, even if it means using a dirtier source.

        CO2 floor at 25kg. Dirty source: 1€+1kgCO2/kWh. Clean source: 1€+0kgCO2/kWh.
        Demand=20. Must produce ≥25 CO2 → Dirty ≥ 25 kWh, excess absorbed by dump.

        Sensitivity: Without minimum_total, optimizer could use all Clean → CO2=0.
        With minimum_total=25, forced to use ≥25 from Dirty → CO2≥25.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', minimum_total=25),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
                    ],
                ),
                Port(
                    'Dirty',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                Port(
                    'Clean',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 0}),
                    ],
                ),
                _waste('Heat'),
            ],
        )
        # Must produce ≥25 CO2. Only Dirty emits CO2 at 1kg/kWh → Dirty ≥ 25 kWh.
        # Demand only 20, so 5 excess absorbed by dump. cost = 25
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 25.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 25.0, rtol=1e-5)

    def test_effect_maximum_per_hour(self, optimize):
        """Proves: maximum_per_hour on an effect caps the per-timestep contribution,
        forcing the optimizer to spread dirty production across timesteps.

        CO2 max_per_hour=8. Dirty: 1€+1kgCO2/kWh. Clean: 5€+0kgCO2/kWh.
        Demand=[15,5]. Without cap, Dirty covers all → CO2=[15,5], cost=20.
        With cap=8/ts, Dirty limited to 8 per ts → Dirty=[8,5], Clean=[7,0].

        Sensitivity: Without max_per_hour, all from Dirty → cost=20.
        With cap, cost = (8+5)*1 + 7*5 = 48.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_per_hour=8),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([15, 5])),
                    ],
                ),
                Port(
                    'Dirty',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                Port(
                    'Clean',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 5, 'CO2': 0}),
                    ],
                ),
            ],
        )
        # t=0: Dirty=8 (capped), Clean=7. t=1: Dirty=5, Clean=0.
        # cost = (8+5)*1 + 7*5 = 13 + 35 = 48
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 48.0, rtol=1e-5)

    def test_effect_minimum_per_hour(self, optimize):
        """Proves: minimum_per_hour on an effect forces a minimum per-timestep
        contribution, even when zero would be cheaper.

        CO2 min_per_hour=10. Dirty: 1€+1kgCO2/kWh. Demand=[5,5].
        Without floor, Dirty=5 each ts → CO2=[5,5]. With floor, Dirty must
        produce ≥10 each ts → excess absorbed by dump.

        Sensitivity: Without min_per_hour, cost=10. With it, cost=20.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', minimum_per_hour=10),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([5, 5])),
                    ],
                ),
                Port(
                    'Dirty',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                _waste('Heat'),
            ],
        )
        # Must emit ≥10 CO2 each ts → Dirty ≥ 10 each ts → cost = 20
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 20.0, rtol=1e-5)

    @pytest.mark.skip(reason='maximum_temporal not supported in fluxopt')
    def test_effect_maximum_temporal(self, optimize):
        """Proves: maximum_temporal caps the sum of an effect's per-timestep contributions
        over the period, forcing suboptimal dispatch.

        CO2 maximum_temporal=12. Dirty: 1€+1kgCO2/kWh. Clean: 5€+0kgCO2/kWh.
        Demand=[10,10]. Without cap, all Dirty → CO2=20, cost=20.
        With temporal cap=12, Dirty limited to 12 total, Clean covers 8.

        Sensitivity: Without maximum_temporal, cost=20. With cap, cost=12+40=52.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', maximum_temporal=12)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add(
            fx.Bus('Heat'),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port(
                'Dirty', imports=[fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]
            ),
            fx.Port(
                'Clean', imports=[fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour={'costs': 5, 'CO2': 0})]
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 52.0, rtol=1e-5)
        assert_allclose(fs.solution['CO2'].item(), 12.0, rtol=1e-5)

    @pytest.mark.skip(reason='minimum_temporal not supported in fluxopt')
    def test_effect_minimum_temporal(self, optimize):
        """Proves: minimum_temporal forces the sum of an effect's per-timestep contributions
        to reach at least the specified value.

        CO2 minimum_temporal=25. Dirty: 1€+1kgCO2/kWh. Demand=[10,10] (total=20).
        Must produce ≥25 CO2 → Dirty ≥25, but demand only 20.
        Excess absorbed by bus with imbalance_penalty=0.

        Sensitivity: Without minimum_temporal, Dirty=20 → cost=20.
        With floor=25, Dirty=25 → cost=25.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', minimum_temporal=25)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add(
            fx.Bus('Heat', imbalance_penalty_per_flow_hour=0),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port(
                'Dirty', imports=[fx.Flow(bus='Heat', flow_id='heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['CO2'].item(), 25.0, rtol=1e-5)
        assert_allclose(fs.solution['costs'].item(), 25.0, rtol=1e-5)

    @pytest.mark.skip(reason='share_from_periodic not supported in fluxopt')
    def test_share_from_periodic(self, optimize):
        """Proves: share_from_periodic adds a weighted fraction of one effect's periodic
        (investment/fixed) sum into another effect's total.

        costs has share_from_periodic={'CO2': 10}. Boiler invest emits 5 kgCO2 fixed.
        Direct costs = invest(100) + fuel(20) = 120. CO2 periodic = 5.
        Shared: 10 * 5 = 50. Total costs = 120 + 50 = 170.

        Sensitivity: Without share_from_periodic, costs=120. With it, costs=170.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg')
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True, share_from_periodic={'CO2': 10})
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'Boiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat',
                    flow_id='heat',
                    size=fx.InvestParameters(fixed_size=50, effects_of_investment={'costs': 100, 'CO2': 5}),
                ),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 170.0, rtol=1e-5)

    @pytest.mark.skip(reason='maximum_periodic not supported in fluxopt')
    def test_effect_maximum_periodic(self, optimize):
        """Proves: maximum_periodic limits the total periodic (investment-related) effect.

        Two boilers: CheapBoiler (invest=10€, CO2_periodic=100kg) and
        ExpensiveBoiler (invest=50€, CO2_periodic=10kg).
        CO2 has maximum_periodic=50. CheapBoiler's 100kg exceeds this.

        Sensitivity: Without limit, CheapBoiler chosen → cost=30.
        With limit=50, ExpensiveBoiler needed → cost=70.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', maximum_periodic=50)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat',
                    flow_id='heat',
                    size=fx.InvestParameters(fixed_size=50, effects_of_investment={'costs': 10, 'CO2': 100}),
                ),
            ),
            fx.Converter.boiler(
                'ExpensiveBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat',
                    flow_id='heat',
                    size=fx.InvestParameters(fixed_size=50, effects_of_investment={'costs': 50, 'CO2': 10}),
                ),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 70.0, rtol=1e-5)

    @pytest.mark.skip(reason='minimum_periodic not supported in fluxopt')
    def test_effect_minimum_periodic(self, optimize):
        """Proves: minimum_periodic forces a minimum total periodic effect.

        Boiler with optional investment (invest=100€, CO2_periodic=50kg).
        CO2 has minimum_periodic=40.

        Sensitivity: Without minimum_periodic, no investment → cost=40.
        With minimum_periodic=40, must invest → cost=120.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        co2 = fx.Effect('CO2', 'kg', minimum_periodic=40)
        costs = fx.Effect('costs', '€', is_standard=True, is_objective=True)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            costs,
            co2,
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'InvestBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat',
                    flow_id='heat',
                    size=fx.InvestParameters(fixed_size=50, effects_of_investment={'costs': 100, 'CO2': 50}),
                ),
            ),
            fx.Converter.boiler(
                'Backup',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(bus='Heat', flow_id='heat', size=100),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 120.0, rtol=1e-5)
