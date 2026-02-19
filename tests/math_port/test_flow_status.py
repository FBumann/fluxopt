"""Mathematical correctness tests for Flow status (on/off) variables."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port, Status

from .conftest import _waste, ts


class TestFlowStatus:
    def test_startup_cost(self, optimize):
        """Proves: effects_per_startup adds a fixed cost each time the unit transitions to on.

        Demand pattern [0,10,0,10,0] forces 2 start-up events.

        Sensitivity: Without startup costs, objective=40 (fuel only).
        With 100€/startup * 2 startups, objective=240.
        """
        result = optimize(
            timesteps=ts(5),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 10, 0, 10, 0])),
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
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior=[0],
                        status=Status(effects_per_startup={'cost': 100}),
                    ),
                ),
            ],
        )
        # fuel = (10+10)/0.5 = 40, startups = 2, cost = 40 + 200 = 240
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 240.0, rtol=1e-5)

    @pytest.mark.skip(reason='active_hours not supported in fluxopt')
    def test_active_hours_max(self, optimize):
        """Proves: active_hours_max limits the total number of on-hours for a unit.

        Cheap boiler limited to 1 hour; expensive backup.
        Sensitivity: Without limit, cost=40. With limit=1, cost=60.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(3)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 20, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat', flow_id='heat', size=100, status_parameters=fx.StatusParameters(active_hours_max=1)
                ),
            ),
            fx.Converter.boiler(
                'ExpensiveBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(bus='Heat', flow_id='heat', size=100),
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 60.0, rtol=1e-5)

    def test_min_uptime_forces_operation(self, optimize):
        """Proves: min_uptime forces a unit to stay on for at least N consecutive hours
        once started, even if cheaper to turn off earlier.

        Cheap boiler (eta=0.5) with min_uptime=2 and max_uptime=2.
        demand = [5, 10, 20, 18, 12]. Optimal: boiler on t=0,1 and t=3,4; backup at t=2.

        Sensitivity: The constraint forces status=[1,1,0,1,1].
        """
        result = optimize(
            timesteps=ts(5),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([5, 10, 20, 18, 12])),
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
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=100,
                        relative_minimum=0.01,
                        prior=[0],
                        status=Status(min_uptime=2, max_uptime=2),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.2,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # Boiler on t=0,1 and t=3,4. Off at t=2 → backup.
        # Boiler fuel: (5+10+18+12)/0.5 = 90. Backup fuel: 20/0.2 = 100. Total = 190.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 190.0, rtol=1e-5)
        assert_allclose(
            result.solution['flow--on'].sel(flow='Boiler(Heat)').values,
            [1, 1, 0, 1, 1],
            atol=1e-5,
        )

    def test_min_downtime_prevents_restart(self, optimize):
        """Proves: min_downtime prevents a unit from restarting before N consecutive
        off-hours have elapsed.

        Cheap boiler (eta=1.0, min_downtime=3) was on before the horizon.
        demand = [20, 0, 20, 0]. Must stay off for t=1,2,3 → cannot serve t=2.

        Sensitivity: Without min_downtime, boiler restarts at t=2 → cost=40.
        With min_downtime=3, backup needed at t=2 → cost=60.
        """
        result = optimize(
            timesteps=ts(4),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 0, 20, 0])),
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
                        prior=[20],
                        status=Status(min_downtime=3),
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
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        # Verify boiler off at t=2
        assert_allclose(result.solution['flow--on'].sel(flow='Boiler(Heat)').values[2], 0.0, atol=1e-5)

    def test_effects_per_active_hour(self, optimize):
        """Proves: effects_per_running_hour adds a cost for each hour a unit is on.

        Boiler (eta=1.0) with 50€/running_hour. Demand=[10,10]. Boiler is on both hours.

        Sensitivity: Without effects_per_running_hour, cost=20 (fuel only).
        With 50€/h * 2h, cost = 20 + 100 = 120.
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
                        size=100,
                        relative_minimum=0.1,
                        status=Status(effects_per_running_hour={'cost': 50}),
                    ),
                ),
            ],
        )
        # fuel=20, active_hour_cost=2*50=100, total=120
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 120.0, rtol=1e-5)

    @pytest.mark.skip(reason='active_hours not supported in fluxopt')
    def test_active_hours_min(self, optimize):
        """Proves: active_hours_min forces a unit to run for at least N hours total.

        Expensive boiler (eta=0.5, active_hours_min=2). Cheap backup (eta=1.0).
        Demand=[10,10]. Without floor, all from backup → cost=20.
        With active_hours_min=2, expensive boiler must run both hours.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'ExpBoiler',
                thermal_efficiency=0.5,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat', flow_id='heat', size=100, status_parameters=fx.StatusParameters(active_hours_min=2)
                ),
            ),
            fx.Converter.boiler(
                'CheapBoiler',
                thermal_efficiency=1.0,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(bus='Heat', flow_id='heat', size=100),
            ),
        )
        fs = optimize(fs)
        status = fs.solution['ExpBoiler(heat)|status'].values[:-1]
        assert_allclose(status, [1, 1], atol=1e-5)

    def test_max_downtime(self, optimize):
        """Proves: max_downtime forces a unit to restart after being off for N consecutive hours.

        Expensive boiler (eta=0.5, max_downtime=1, relative_minimum=0.5, size=20).
        Cheap backup (eta=1.0). Demand=[10,10,10,10].

        Sensitivity: Without max_downtime, all from CheapBoiler → cost=40.
        With max_downtime=1, ExpBoiler forced on ≥2 hours → cost > 40.
        """
        result = optimize(
            timesteps=ts(4),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10])),
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
                    'ExpBoiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(
                        bus='Heat',
                        size=20,
                        relative_minimum=0.5,
                        prior=[10],
                        status=Status(max_downtime=1),
                    ),
                ),
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # Verify max_downtime: no two consecutive off-hours
        status = result.solution['flow--on'].sel(flow='ExpBoiler(Heat)').values
        for i in range(len(status) - 1):
            assert not (status[i] < 0.5 and status[i + 1] < 0.5), f'Consecutive off at t={i},{i + 1}: status={status}'
        assert result.effect_totals.sel(effect='cost').item() > 40.0 + 1e-5

    @pytest.mark.skip(reason='startup_limit not supported in fluxopt')
    def test_startup_limit(self, optimize):
        """Proves: startup_limit caps the number of startup events per period.

        Boiler (eta=0.8, startup_limit=1). Demand=[10,0,10]. cost=32.5.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(3)
        fs.add(
            fx.Bus('Heat'),
            fx.Bus('Gas'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Heat', flow_id='heat', size=1, fixed_relative_profile=np.array([10, 0, 10]))],
            ),
            fx.Port('GasSrc', imports=[fx.Flow(bus='Gas', flow_id='gas', effects_per_flow_hour=1)]),
            fx.Converter.boiler(
                'Boiler',
                thermal_efficiency=0.8,
                fuel_flow=fx.Flow(bus='Gas', flow_id='fuel'),
                thermal_flow=fx.Flow(
                    bus='Heat',
                    flow_id='heat',
                    size=20,
                    relative_minimum=0.5,
                    previous_flow_rate=0,
                    status_parameters=fx.StatusParameters(startup_limit=1),
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
        assert_allclose(fs.solution['costs'].item(), 32.5, rtol=1e-5)

    def test_max_uptime_standalone(self, optimize):
        """Proves: max_uptime on a flow limits continuous operation.

        CheapBoiler (eta=1.0) with max_uptime=2.
        ExpensiveBackup (eta=0.5). Demand=[10]*5.
        Cheap boiler can run at most 2 consecutive hours.

        Sensitivity: Without max_uptime, all 5 hours cheap → cost=50.
        With max_uptime=2, backup covers 1 hour → cost=60.
        """
        result = optimize(
            timesteps=ts(5),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10, 10, 10])),
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
                        status=Status(max_uptime=2),
                    ),
                ),
                Converter.boiler(
                    'ExpensiveBackup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # Verify no more than 2 consecutive on-hours
        status = result.solution['flow--on'].sel(flow='CheapBoiler(Heat)').values
        max_consecutive = 0
        current = 0
        for s in status:
            if s > 0.5:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        assert max_consecutive <= 2, f'max_uptime violated: {status}'
        # Cheap: 4*10 = 40 fuel. Backup @1h: 10/0.5 = 20 fuel. Total = 60.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)


class TestPreviousFlowRate:
    def test_previous_flow_rate_scalar_on_forces_min_uptime(self, optimize):
        """Proves: prior=[X] with X>0 means unit was ON before t=0,
        and min_uptime carry-over forces it to stay on.

        Boiler with min_uptime=2, prior=[10] (was on for 1 hour before t=0).
        Must stay on at t=0 to complete 2-hour minimum uptime block.

        Sensitivity: With prior=[0] (was off), cost=0.
        With prior=[10] (was on), cost=10.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 0])),
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
                        prior=[10],
                        status=Status(min_uptime=2),
                    ),
                ),
            ],
        )
        # Forced ON at t=0 (relative_min=10), cost=10.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)

    def test_previous_flow_rate_scalar_off_no_carry_over(self, optimize):
        """Proves: prior=[0] means unit was OFF before t=0, so no min_uptime carry-over.

        Same setup but prior=[0]. Cost=0 here vs cost=10 with prior=[10].
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 0])),
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
                        status=Status(min_uptime=2),
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 0.0, rtol=1e-5)

    def test_previous_flow_rate_array_uptime_satisfied_vs_partial(self, optimize):
        """Proves: prior array length affects uptime carry-over calculation.

        prior=[10, 20] (2 hours ON), min_uptime=2 → satisfied, can turn off.
        Demand=[0, 0]. With satisfied uptime, can be off entirely (cost=0).
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 0])),
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
                        prior=[10, 20],
                        status=Status(min_uptime=2),
                    ),
                ),
            ],
        )
        # With 2h uptime history, min_uptime=2 is satisfied → can be off at t=0 → cost=0
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 0.0, rtol=1e-5)

    def test_previous_flow_rate_array_partial_uptime_forces_continuation(self, optimize):
        """Proves: prior array with partial uptime forces continuation.

        Boiler with min_uptime=3, prior=[0, 10] (off then on for 1 hour).
        Only 1 hour of uptime accumulated → needs 2 more hours at t=0,t=1.

        Sensitivity: With prior=[0] (was off), cost=0. With prior=[0, 10], cost=20.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 0, 0])),
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
                        prior=[0, 10],
                        status=Status(min_uptime=3),
                    ),
                ),
            ],
        )
        # prior=[0, 10]: consecutive uptime = 1 hour
        # min_uptime=3: needs 2 more hours → forced on at t=0, t=1 with relative_min=10
        # cost = 2 * 10 = 20
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

    def test_previous_flow_rate_array_min_downtime_carry_over(self, optimize):
        """Proves: prior array affects min_downtime carry-over.

        CheapBoiler with min_downtime=3, prior=[10, 0] (was on, then off for 1 hour).
        Only 1 hour of downtime accumulated → needs 2 more hours off at t=0,t=1.

        Sensitivity: Without carry-over, cost=60. With carry-over, cost=100.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([20, 20, 20])),
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
                        prior=[10, 0],
                        status=Status(min_downtime=3),
                    ),
                ),
                Converter.boiler(
                    'ExpensiveBoiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(bus='Gas', id='fuel'),
                    thermal_flow=Flow(bus='Heat', size=100),
                ),
            ],
        )
        # prior=[10, 0]: last is OFF, consecutive downtime = 1 hour
        # min_downtime=3: needs 2 more off hours → CheapBoiler off t=0,t=1
        # ExpensiveBoiler covers t=0,t=1: 2*20/0.5 = 80. CheapBoiler covers t=2: 20.
        # Total = 100
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_previous_flow_rate_array_longer_history(self, optimize):
        """Proves: longer prior arrays correctly track consecutive hours.

        Boiler with min_uptime=4, prior=[0, 10, 20, 30] (off, then on for 3 hours).
        3 hours uptime accumulated → needs 1 more hour at t=0.

        Sensitivity: With prior=[10, 20, 30, 40] (4 hours on), cost=0.
        With prior=[0, 10, 20, 30] (3 hours on), cost=10.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([0, 0])),
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
                        prior=[0, 10, 20, 30],
                        status=Status(min_uptime=4),
                    ),
                ),
            ],
        )
        # prior=[0, 10, 20, 30]: consecutive uptime from end = 3 hours
        # min_uptime=4: needs 1 more → forced on at t=0 with relative_min=10
        # cost = 10
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)
