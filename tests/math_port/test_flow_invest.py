"""Mathematical correctness tests for Flow investment decisions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port, Sizing, Status

from .conftest import ts


class TestFlowInvest:
    def test_invest_size_optimized(self, optimize):
        """Proves: Sizing correctly sizes the unit to match peak demand
        when there is a per-size investment cost.

        Sensitivity: If sizing were broken (e.g. forced to max=200), invest cost
        would be 10+200=210, total=290 instead of 140. If sized to 0, infeasible.
        Only size=50 (peak demand) minimizes the sum of invest + fuel cost.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 50, 20])),
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
                        size=Sizing(
                            min_size=0,
                            max_size=200,
                            mandatory=False,
                            effects_fixed={'cost': 10},
                            effects_per_size={'cost': 1},
                        ),
                    ),
                ),
            ],
        )
        # size = 50 (peak), invest cost = 10 + 50*1 = 60, fuel = 80
        # total = 140
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 50.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 140.0, rtol=1e-5)

    def test_invest_optional_not_built(self, optimize):
        """Proves: Optional investment is correctly skipped when the fixed investment
        cost outweighs operational savings.

        InvestBoiler has eta=1.0 (efficient) but 99999€ fixed invest cost.
        CheapBoiler has eta=0.5 (inefficient) but no invest cost.

        Sensitivity: If investment cost were ignored (free invest), InvestBoiler
        would be built and used → fuel=20 instead of 40.
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
                    'InvestBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=Sizing(min_size=0, max_size=100, mandatory=False, effects_fixed={'cost': 99999}),
                    ),
                ),
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        assert_allclose(result.solution['flow--size_indicator'].sel(flow='InvestBoiler(Heat)').item(), 0.0, atol=1e-5)
        # All demand served by CheapBoiler: fuel = 20/0.5 = 40
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)

    def test_invest_minimum_size(self, optimize):
        """Proves: Sizing min_size forces the invested capacity to be
        at least the specified value, even when demand is much smaller.

        Demand peak=10, min_size=100, cost_per_size=1 → must invest 100.

        Sensitivity: Without min_size, optimal invest=10 → cost=10+20=30.
        With min_size=100, invest cost=100 → cost=120.
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
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=Sizing(min_size=100, max_size=200, mandatory=True, effects_per_size={'cost': 1}),
                    ),
                ),
            ],
        )
        # Must invest at least 100, cost_per_size=1 → invest=100
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 100.0, rtol=1e-5)
        # fuel=20, invest=100 → total=120
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 120.0, rtol=1e-5)

    def test_invest_fixed_size(self, optimize):
        """Proves: min_size==max_size creates a binary invest-or-not decision at exactly the
        specified capacity — no continuous sizing.

        FixedBoiler: fixed_size=80, invest_cost=10€, eta=1.0.
        Backup: eta=0.5, no invest. Demand=[30,30], gas=1€/kWh.

        Sensitivity: The key assertion is that invested size is exactly 80, not 30.
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
            ],
            converters=[
                Converter.boiler(
                    'FixedBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=Sizing(min_size=80, max_size=80, mandatory=False, effects_fixed={'cost': 10}),
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
        # size must be exactly 80 (not optimized to 30)
        assert_allclose(result.sizes.sel(flow='FixedBoiler(Heat)').item(), 80.0, rtol=1e-5)
        assert_allclose(result.solution['flow--size_indicator'].sel(flow='FixedBoiler(Heat)').item(), 1.0, atol=1e-5)
        # fuel=60 (all from FixedBoiler @eta=1), invest=10, total=70
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 70.0, rtol=1e-5)

    @pytest.mark.skip(reason='piecewise sizing not supported in fluxopt')
    def test_piecewise_invest_cost(self, optimize):
        """Proves: piecewise_effects_of_investment applies non-linear investment costs
        where the cost-per-size changes across size segments (economies of scale).

        Segment 1: size 0→50, cost 0→100 (2€/kW).
        Segment 2: size 50→200, cost 100→250 (1€/kW, cheaper per unit).
        Demand peak=80. invest=130, fuel=80, total=210.

        Sensitivity: If linear cost at 2€/kW throughout, invest=160 → total=240.
        """
        raise NotImplementedError  # TODO: implement piecewise sizing

    def test_invest_mandatory_forces_investment(self, optimize):
        """Proves: mandatory=True forces investment even when it's not economical.

        ExpensiveBoiler: mandatory=True, fixed invest=1000€, per_size=1€/kW, eta=1.0.
        CheapBoiler: no invest, eta=0.5. Demand=[10,10].

        Without mandatory, CheapBoiler covers all: fuel=40, total=40.
        With mandatory=True, ExpensiveBoiler must be built: invest=1000+10, fuel=20, total=1030.
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
                    'ExpensiveBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=Sizing(
                            min_size=10,
                            max_size=100,
                            mandatory=True,
                            effects_fixed={'cost': 1000},
                            effects_per_size={'cost': 1},
                        ),
                    ),
                ),
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # mandatory=True forces ExpensiveBoiler to be built, size=10 (minimum needed)
        assert_allclose(result.sizes.sel(flow='ExpensiveBoiler(Heat)').item(), 10.0, rtol=1e-5)
        # invest=1000+10*1=1010, fuel from ExpensiveBoiler=20 (eta=1.0), total=1030
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 1030.0, rtol=1e-5)

    def test_invest_not_mandatory_skips_when_uneconomical(self, optimize):
        """Proves: mandatory=False (default) allows optimizer to skip investment
        when it's not economical.

        ExpensiveBoiler: mandatory=False, invest_cost=1000€, eta=1.0.
        CheapBoiler: no invest, eta=0.5. Demand=[10,10].

        Sensitivity: cost=40 here vs cost=1030 with mandatory=True.
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
                    'ExpensiveBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=Sizing(min_size=10, max_size=100, mandatory=False, effects_fixed={'cost': 1000}),
                    ),
                ),
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # mandatory=False allows skipping uneconomical investment
        assert_allclose(
            result.solution['flow--size_indicator'].sel(flow='ExpensiveBoiler(Heat)').item(), 0.0, atol=1e-5
        )
        # CheapBoiler covers all: fuel = 20/0.5 = 40
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)

    @pytest.mark.skip(reason='retirement effects not supported in fluxopt')
    def test_invest_effects_of_retirement(self, optimize):
        """Proves: effects_of_retirement adds a cost when NOT investing.

        Boiler with effects_of_retirement=500€. If not built, incur 500€ penalty.

        Sensitivity: Without effects_of_retirement, backup is cheaper (fuel=40 vs 120).
        With retirement=500, investing becomes cheaper. Cost=120.
        """
        raise NotImplementedError  # TODO: implement effects_of_retirement on Sizing

    @pytest.mark.skip(reason='retirement effects not supported in fluxopt')
    def test_invest_retirement_triggers_when_not_investing(self, optimize):
        """Proves: effects_of_retirement is incurred when investment is skipped.

        Boiler with invest_cost=1000, effects_of_retirement=50.
        Optimizer skips investment, pays retirement cost. cost=90.

        Sensitivity: Without effects_of_retirement, cost=40. With it, cost=90.
        """
        raise NotImplementedError  # TODO: implement effects_of_retirement on Sizing


class TestFlowInvestWithStatus:
    def test_invest_with_startup_cost(self, optimize):
        """Proves: Sizing and Status work together correctly.

        Boiler with investment sizing AND startup costs.
        Demand=[0,20,0,20]. Two startup events if boiler is used.

        Sensitivity: Without startup_cost, cost = invest + fuel.
        With startup_cost=50 * 2, cost increases by 100.
        """
        result = optimize(
            timesteps=ts(4),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
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
                            mandatory=False,
                            effects_fixed={'cost': 10},
                            effects_per_size={'cost': 1},
                        ),
                        status=Status(effects_per_startup={'cost': 50}),
                    ),
                ),
            ],
        )
        # size=20 (peak), invest=10+20=30, fuel=40, 2 startups=100
        # total = 30 + 40 + 100 = 170
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 20.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 170.0, rtol=1e-5)

    def test_invest_with_min_uptime(self, optimize):
        """Proves: Invested unit respects min_uptime constraint.

        InvestBoiler with sizing AND min_uptime=2. Once started, must stay on 2 hours.
        Backup available but expensive. Demand=[20,10,20].

        Sensitivity: The cost changes due to min_uptime forcing operation patterns.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 10, 20])),
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
                    'InvestBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        relative_minimum=0.1,
                        size=Sizing(min_size=0, max_size=100, mandatory=False, effects_per_size={'cost': 1}),
                        status=Status(min_uptime=2),
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
        # InvestBoiler is built (cheaper fuel @eta=1.0 vs Backup @eta=0.5)
        # size=20 (peak demand), invest=20
        # min_uptime=2: runs continuously t=0,1,2
        # fuel = 20 + 10 + 20 = 50
        # total = 20 (invest) + 50 (fuel) = 70
        assert_allclose(result.sizes.sel(flow='InvestBoiler(Heat)').item(), 20.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 70.0, rtol=1e-5)
        # Verify InvestBoiler runs all 3 hours due to min_uptime
        status = result.solution['flow--on'].sel(flow='InvestBoiler(Heat)').values
        assert_allclose(status, [1, 1, 1], atol=1e-5)

    @pytest.mark.skip(reason='active_hours not supported in fluxopt')
    def test_invest_with_active_hours_max(self, optimize):
        """Proves: Invested unit respects active_hours_max constraint.

        InvestBoiler (eta=1.0) with active_hours_max=2. Backup (eta=0.5).
        Demand=[10,10,10,10]. InvestBoiler can only run 2 of 4 hours.

        Sensitivity: Without limit, InvestBoiler runs all 4 hours → fuel=40.
        With active_hours_max=2, cost=61.
        """
        raise NotImplementedError  # TODO: implement active_hours_max on Status (#16)
