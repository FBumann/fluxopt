"""Status (on/off) constraint tests.

Each test builds a small model with flows that have Status and verifies
semi-continuous behavior, startup costs, and running costs.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from conftest import assert_off_blocks, assert_on_blocks, ts, waste
from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port, Sizing, Status, optimize


class TestSemiContinuous:
    def test_flow_is_zero_or_within_bounds(self):
        """Status flow must be either 0 or in [min, max] * size.

        Source: size=100, rel_min=0.5, Status(), 1€/MWh.
        Backup: unsized, 10€/MWh.
        Demand: [30, 60, 0].

        t=0: demand=30. Source min=50, so cheaper to use source at 50 (cost 50)
             than backup at 30*10=300. Waste absorbs 20.
        t=1: demand=60. Source at 60 (cost 60).
        t=2: demand=0. Source off (cost 0), backup off.
        Total: 50 + 60 = 110.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[30, 60, 0])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.5,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
                waste('Heat'),
            ],
        )
        rates = result.flow_rate('Src(Heat)').values
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # t=2: flow should be off
        assert_allclose(on[2], 0.0, atol=1e-5)
        assert_allclose(rates[2], 0.0, atol=1e-5)

        # t=0, t=1: flow should be on and >= 50 (= 100 * 0.5)
        assert_allclose(on[0], 1.0, atol=1e-5)
        assert_allclose(on[1], 1.0, atol=1e-5)
        assert rates[0] >= 50.0 - 1e-5
        assert rates[1] >= 60.0 - 1e-5

    def test_status_avoids_tiny_output(self):
        """With Status, the solver cannot produce below minimum when on.

        Source: size=100, rel_min=0.4, Status(), 1€/MWh.
        Demand: [10, 80]. Backup: 0.5€/MWh.

        t=0: demand=10 < min=40. Cheaper to use backup (10*0.5=5) than
             source at 40 (40*1=40). Source off.
        t=1: demand=80. Source at 80 (cost 80), cheaper than backup (80*0.5=40)...
             actually backup is cheaper. So source stays off, backup covers both.
        Total with all-backup: 10*0.5 + 80*0.5 = 45.
        """
        result = optimize(
            ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 80])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.4,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 45.0, rtol=1e-5)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert_allclose(on, [0.0, 0.0], atol=1e-5)


class TestStartupCosts:
    def test_startup_cost_added_to_objective(self):
        """Startup cost is charged per event.

        Source: size=100, prior_rates=[0] (was off), effects_per_startup={'costs': 50}, 1€/MWh.
        Demand: [60, 60] (constant). No backup.

        Source runs both hours: 1 startup event at t=0 (was off).
        Operational: 60*1*2 = 120. Startup: 50. Total: 170.
        """
        result = optimize(
            ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[60, 60])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(effects_per_startup={'costs': 50}),
                            prior_rates=[0],
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 170.0, rtol=1e-5)

    def test_startup_cost_discourages_cycling(self):
        """High startup cost keeps unit running rather than cycling.

        Source: size=100, rel_min=0.3, prior_rates=[0] (was off),
                Status(effects_per_startup={'costs': 200}), 0.1€/MWh.
        Backup: 5€/MWh.
        Demand: [80, 0, 80].

        On all 3h: 1 startup=200 + (80+30+80)*0.1=219. Waste absorbs 30 at t=1.
        Cycling on/off/on: 2*200 + (80+80)*0.1=416.
        Stays on to avoid 2nd startup.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[80, 0, 80])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.3,
                            effects_per_flow_hour={'costs': 0.1},
                            status=Status(effects_per_startup={'costs': 200}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values

        # Source stays on all 3 hours to avoid 2nd startup
        assert_allclose(on, [1.0, 1.0, 1.0], atol=1e-5)
        # Only 1 startup event (at t=0)
        assert_allclose(np.sum(startup), 1.0, atol=1e-5)


class TestPrior:
    def test_prior_none_gives_free_initial(self):
        """Flow.prior=None with Status() leaves initial state free.

        Source: size=100, Status(effects_per_startup={'costs': 1000}), no prior.
        Demand: [50, 50]. No backup, so source must run.

        Solver is free to assume on at t=-1 (no startup cost) or off (startup cost).
        With high startup cost, solver prefers to assume it was already on.
        Expected cost: 0 (no startup) + 0 (no flow cost) = 0.
        """
        result = optimize(
            ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            status=Status(effects_per_startup={'costs': 1000}),
                        )
                    ],
                ),
            ],
        )
        # With free initial, solver avoids startup cost entirely
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values
        assert_allclose(np.sum(startup), 0.0, atol=1e-5)

    def test_prior_on_carries_uptime(self):
        """Prior with consecutive on-hours carries uptime into the horizon.

        Source: size=100, min_uptime=3h, prior_rates=[50, 60] (2h on already).
        Demand: [80, 0, 0].

        With 2h of prior uptime and min_uptime=3h, source must stay on for
        at least 1 more hour. After that it can turn off.
        t=0: must stay on (uptime=3h total). Flow=80.
        t=1: can turn off. Demand=0, cheaper to turn off.
        t=2: off.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[80, 0, 0])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=3),
                            prior_rates=[50, 60],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 0.5})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        # t=0: forced on by min_uptime continuation
        assert_allclose(on[0], 1.0, atol=1e-5)

    def test_prior_off_carries_downtime(self):
        """Prior with consecutive off-hours carries downtime into the horizon.

        Source: size=100, min_downtime=3h, prior_rates=[0, 0] (2h off already).
        Demand: [80, 80, 80].

        With 2h of prior downtime and min_downtime=3h, source must stay off
        for at least 1 more hour.
        t=0: must stay off (downtime=3h total). Backup covers.
        t=1: can turn on.
        t=2: on.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[80, 80, 80])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_downtime=3),
                            prior_rates=[0, 0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        # t=0: forced off by min_downtime continuation
        assert_allclose(on[0], 0.0, atol=1e-5)
        # t=1, t=2: can and should turn on (cheaper than backup)
        assert_allclose(on[1], 1.0, atol=1e-5)
        assert_allclose(on[2], 1.0, atol=1e-5)

    def test_running_cost_per_hour(self):
        """Running cost is charged per hour the unit is on.

        Source: size=100, Status(effects_per_running_hour={'costs': 10}), 1€/MWh.
        Demand: [50, 50].

        Operational: 50*1*2 = 100. Running: 10*1*2 = 20.
        Total: 100 + 20 = 120.
        """
        result = optimize(
            ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(effects_per_running_hour={'costs': 10}),
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 120.0, rtol=1e-5)


class TestStatusSizing:
    def test_semi_continuous_with_optimized_size(self):
        """Status + Sizing: semi-continuous behavior with optimized capacity.

        Src: Sizing(20, 200, mandatory=True), rel_min=0.5, Status(), 1€/MWh.
        Backup: 10€/MWh.
        Demand: [30, 80, 0].

        Solver must invest in size and respect semi-continuous bounds.
        t=0: demand=30. Source min = 0.5*S. If S=80, min=40 > 30 → source at 40,
             waste absorbs 10, cost=40. Cheaper than backup (30*10=300).
        t=1: demand=80. Source at 80, cost=80.
        t=2: demand=0. Source off, cost=0.
        Optimal size=80 (just enough for peak). Total operational: 40+80=120.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[30, 80, 0])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(20, 200, mandatory=True),
                            relative_minimum=0.5,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
                waste('Heat'),
            ],
        )
        rates = result.flow_rate('Src(Heat)').values
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        size = float(result.sizes.sel(flow='Src(Heat)').values)

        # Size must be at least 80 to cover peak demand
        assert size >= 80.0 - 1e-5

        # t=2: off
        assert_allclose(on[2], 0.0, atol=1e-5)
        assert_allclose(rates[2], 0.0, atol=1e-5)

        # t=0, t=1: on and respecting minimum
        assert_allclose(on[0], 1.0, atol=1e-5)
        assert_allclose(on[1], 1.0, atol=1e-5)
        assert rates[0] >= 0.5 * size - 1e-5
        assert rates[1] >= 80.0 - 1e-5

    def test_sizing_rel_min_forces_off_at_low_demand(self):
        """Sizing + Status + relative_minimum forces off when demand < min_load.

        Src: Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.5}),
             rel_min=0.5, Status(), 1€/MWh.
        Backup: 10€/MWh.
        Demand: [5, 50].

        Optimal size=50 (peak demand). min_load = 0.5*50 = 25 > demand[0]=5.
        t=0: Src must turn OFF (can't produce <25), Backup covers: 5*10=50.
        t=1: Src ON at 50, cost=50.
        invest=50*0.5=25 + operational=50 + backup=50 = 125.

        Without status: min_load=25 forces Src to produce 25 even when demand=5,
        requiring waste absorption. With status, cleaner: just turn off.
        """
        result = optimize(
            ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[5, 50])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.5}),
                            relative_minimum=0.5,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        assert_allclose(size, 50.0, rtol=1e-4)
        # t=0: off (demand < min_load)
        assert_allclose(on[0], 0.0, atol=1e-5)
        # t=1: on
        assert_allclose(on[1], 1.0, atol=1e-5)
        assert_allclose(result.objective, 125.0, rtol=1e-4)

    def test_optional_sizing_not_invested_means_off(self):
        """Optional Sizing + Status: if not invested, on=0 and flow rate=0.

        Src: Sizing(50, 100, mandatory=False, effects_fixed={'costs': 1000}),
             Status(), 1€/MWh. High fixed cost discourages investment.
        Backup: 2€/MWh.
        Demand: [10]*3.

        Without invest: backup cost = 10*2*3 = 60.
        With invest: fixed=1000 + operational ≈ 30 = 1030. Too expensive.
        Solver should NOT invest. indicator=0 → on=0 → flow rate=0.
        The on<=indicator constraint prevents spurious running/startup costs.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 3)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(50, 100, mandatory=False, effects_fixed={'costs': 1000}),
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 2})]),
            ],
        )
        indicator = float(result.solution['flow--size_indicator'].sel(flow='Src(Heat)').values)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        rates = result.flow_rate('Src(Heat)').values

        # Not invested → off → zero flow
        assert_allclose(indicator, 0.0, atol=1e-5)
        assert_allclose(on, [0, 0, 0], atol=1e-5)
        assert_allclose(rates, [0, 0, 0], atol=1e-5)
        # All from backup
        assert_allclose(result.objective, 60.0, rtol=1e-5)

    def test_sizing_with_startup_costs(self):
        """Sizing + Status + startup cost: invest decision includes startup penalty.

        Src: Sizing(0, 200, mandatory=True), Status(effects_per_startup={'costs': 100}),
             prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [50, 50, 50].

        Src runs all 3h: 1 startup(100) + operational(150) + invest_cost(0) = 250.
        All backup: 50*5*3 = 750. Src is cheaper despite startup cost.
        """
        result = optimize(
            ts(3),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[50] * 3)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 200, mandatory=True),
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(effects_per_startup={'costs': 100}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
            ],
        )
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values
        size = float(result.sizes.sel(flow='Src(Heat)').values)

        assert size >= 50.0 - 1e-5
        # Exactly 1 startup at t=0
        assert_allclose(np.sum(startup), 1.0, atol=1e-5)
        # Total: 100 (startup) + 150 (operational) = 250
        assert_allclose(result.objective, 250.0, rtol=1e-5)

    def test_sizing_with_min_uptime(self):
        """Sizing + Status + min_uptime: duration constraint with variable capacity.

        Src: Sizing(0, 200, mandatory=True), rel_min=0.3,
             Status(min_uptime=3), prior_rates=[0], 1€/MWh.
        Backup: 0.5€/MWh (cheaper).
        Demand: [80, 0, 0, 80].

        Backup is cheaper, but once Src turns on, min_uptime=3 forces it to
        stay on for 3 consecutive hours. If Src turns on at t=0 for demand,
        it must stay on through t=2 (even with zero demand), producing at
        least 0.3*S each hour. Waste absorbs excess at t=1,t=2.
        """
        result = optimize(
            ts(4),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[80, 0, 0, 80])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 200, mandatory=True),
                            relative_minimum=0.3,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=3),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 0.5})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        rates = result.flow_rate('Src(Heat)').values
        size = float(result.sizes.sel(flow='Src(Heat)').values)

        # Check on-blocks are ≥3h
        assert_on_blocks(on, min_length=3)

        # When on, flow rate >= rel_min * size
        for t in range(len(on)):
            if on[t] > 0.5:
                assert rates[t] >= 0.3 * size - 1e-5, f'Below minimum at t={t}: {rates[t]} < 0.3*{size}'


class TestMaxUptime:
    def test_max_uptime_forces_shutdown(self):
        """max_uptime=2 limits continuous operation to 2 consecutive hours.

        Src: size=100, Status(max_uptime=2), prior_rates=[0] (was off), 1€/MWh.
        Backup: 10€/MWh.
        Demand: [10, 10, 10, 10, 10].

        Without max_uptime: Src runs all 5h → cost=50.
        With max_uptime=2: Src runs at most 2 consecutive hours, then must
        shut down for ≥1h. Pattern like [on,on,off,on,on] → Src covers 4h,
        Backup covers 1h at 10€ → total=40+10=50... but the waste of backup
        hour makes it 4*10 + 1*10*10 = 140? No:
        Src 4h: 4*10*1 = 40. Backup 1h: 1*10*10 = 100. Total = 140.
        Without: 5*10*1 = 50. So cost > 50 proves the constraint works.
        """
        result = optimize(
            ts(5),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(max_uptime=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no more than 2 consecutive on-hours
        assert_on_blocks(on, max_length=2)

        # Src 4h: 4*10*1=40. Backup 1h: 1*10*10=100. Total=140.
        assert_allclose(result.objective, 140.0, rtol=1e-5)


class TestMaxDowntime:
    def test_max_downtime_forces_restart(self):
        """max_downtime=1 prevents staying off for more than 1 consecutive hour.

        Src: size=100, rel_min=0.5, Status(max_downtime=1), prior_rates=[10] (was on),
             10€/MWh (expensive).
        Backup: 1€/MWh (cheap).
        Demand: [10, 10, 10, 10].

        Without max_downtime: all from cheap Backup → cost=40.
        With max_downtime=1: Src can be off at most 1 consecutive hour. Since
        it was previously on, it can turn off but must restart within 1h.
        This forces Src on for ≥2 of 4 hours → cost > 40.
        """
        result = optimize(
            ts(4),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 4)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.5,
                            effects_per_flow_hour={'costs': 10},
                            status=Status(max_downtime=1),
                            prior_rates=[10],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no two consecutive off-hours
        assert_off_blocks(on, max_length=1, skip_leading=False)

        # Pattern [on,off,on,off]: Src min=50 when on, 2*50*10=1000, Backup 2*10*1=20. Total=1020.
        assert_allclose(result.objective, 1020.0, rtol=1e-5)


class TestDurationCombinations:
    def test_min_and_max_uptime_forces_exact_blocks(self):
        """min_uptime=2 + max_uptime=2 forces operation in exact 2-hour blocks.

        Src: size=100, Status(min_uptime=2, max_uptime=2), prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [5, 10, 20, 18, 12].

        With min=max=2h blocks, best pattern is [on,on,off,on,on]:
        Src covers t=0,1,3,4; Backup covers t=2.
        Src cost: (5+10+18+12)*1 = 45. Backup cost: 20*5 = 100. Total = 145.
        """
        result = optimize(
            ts(5),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[5, 10, 20, 18, 12])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.01,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=2, max_uptime=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert_allclose(on, [1, 1, 0, 1, 1], atol=1e-5)
        assert_allclose(result.objective, 145.0, rtol=1e-5)

    def test_min_uptime_with_min_downtime_block_pattern(self):
        """min_uptime=2 + min_downtime=2 forces on/off blocks of ≥2 hours each.

        Src: size=100, rel_min=0.1, Status(min_uptime=2, min_downtime=2),
             prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [20]*6.

        Must run in ≥2h blocks, off in ≥2h blocks. From prior off, stays off
        ≥2h then on ≥2h. Patterns like [off,off,on,on,on,on] or
        [off,off,on,on,off,off]. Cheapest: maximize Src hours.
        """
        result = optimize(
            ts(6),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[20] * 6)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=2, min_downtime=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify on-blocks are ≥2h
        assert_on_blocks(on, min_length=2)

        # Verify off-blocks within horizon are ≥2h (first block may be carry-over)
        assert_off_blocks(on, min_length=2)

        # Pattern [off,on,on,on,on,on]: Src 5h*20=100, Backup 1h*20*5=100. Total=200.
        assert_allclose(result.objective, 200.0, rtol=1e-5)

    def test_max_uptime_with_prior_carry_over(self):
        """Prior uptime reduces remaining allowed on-time at start of horizon.

        Src: size=100, Status(max_uptime=3), prior_rates=[50, 50] (2h on already),
             1€/MWh.
        Backup: 10€/MWh.
        Demand: [10]*5.

        With 2h prior uptime and max_uptime=3, Src can run at most 1 more
        hour before forced shutdown. Then can restart for up to 3h.
        """
        result = optimize(
            ts(5),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(max_uptime=3),
                            prior_rates=[50, 50],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # t=0 should be on (continuing from prior), then forced off by max_uptime=3
        assert_allclose(on[0], 1.0, atol=1e-5)

        # Verify no run of >3 consecutive on-hours within horizon
        assert_on_blocks(on, max_length=3)

        # Pattern [on,off,on,on,on]: Src 4h*10=40, Backup 1h*10*10=100. Total=140.
        assert_allclose(result.objective, 140.0, rtol=1e-5)

    def test_max_uptime_with_startup_costs(self):
        """max_uptime forces shutdowns which incur startup costs on restart.

        Src: size=100, Status(max_uptime=2, effects_per_startup={'costs': 50}),
             prior_rates=[0], 1€/MWh.
        Backup: 10€/MWh.
        Demand: [10]*5.

        max_uptime=2 forces at least 1 shutdown in 5h. Restarting costs 50€
        each time. Pattern [on,on,off,on,on] = 2 startups = 100€ startup +
        40€ operational + 100€ backup = 240€.
        Without max_uptime: 1 startup = 50 + 50 operational = 100€.
        """
        result = optimize(
            ts(5),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(max_uptime=2, effects_per_startup={'costs': 50}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values

        # Verify max_uptime constraint
        assert_on_blocks(on, max_length=2)

        # Exactly 2 startups (initial + restart after forced shutdown)
        assert_allclose(np.sum(startup), 2.0, atol=1e-5)

        # 2*50 startups + 4h*10 Src + 1h*10*10 Backup = 240
        assert_allclose(result.objective, 240.0, rtol=1e-5)

    def test_max_downtime_with_prior_carry_over(self):
        """Prior downtime reduces remaining allowed off-time at start of horizon.

        Src: size=100, rel_min=0.5, Status(max_downtime=2),
             prior_rates=[0, 0] (2h off already), 10€/MWh (expensive).
        Backup: 1€/MWh (cheap).
        Demand: [10]*4.

        With 2h prior downtime and max_downtime=2, Src must restart
        immediately at t=0 (can't stay off any longer).
        """
        result = optimize(
            ts(4),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 4)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.5,
                            effects_per_flow_hour={'costs': 10},
                            status=Status(max_downtime=2),
                            prior_rates=[0, 0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # With 2h prior off and max_downtime=2, must turn on at t=0
        assert_allclose(on[0], 1.0, atol=1e-5)

    def test_min_uptime_with_half_hour_timesteps(self):
        """Duration constraints work correctly with sub-hourly timesteps.

        Src: size=100, Status(min_uptime=2), prior_rates=[0], 1€/MWh.
        Backup: 0.5€/MWh (cheaper).
        8 timesteps of 30min each (4 hours total).
        Demand: [0,0,0,0, 80,80, 0,0] (demand only at t=4,5 → hours 2-3).

        min_uptime=2h = 4 timesteps at dt=0.5h. Once Src turns on it must
        stay on for 4 consecutive timesteps. Backup is cheaper, so Src only
        runs if forced. Demand at t=4,5 needs coverage. With min_uptime=4ts,
        turning on at t=4 means on through t=7 (or t=4-7). Src runs 4 slots
        at 1€/MWh, costing (80+80+0+0)*0.5*1 = 80. But also Backup may cover
        the demand slots cheaper. Since Backup is 0.5€/MWh: 80*0.5*0.5 +
        80*0.5*0.5 = 40. So all-backup = 40.

        But if Src turns on at all (forced by some mechanism), it must stay
        on 4 timesteps. Let's instead make Src cheaper but with startup cost
        to make it interesting.

        Revised: Src=1€/MWh, Backup=5€/MWh. Demand=[80]*8.
        Src must run for demand. min_uptime=2h → once on, stays on ≥4 slots.
        Src runs all 8 slots: cost = 80*0.5*8*1 = 320.
        Verify on-blocks are ≥4 timesteps (=2h).
        """
        # 30-minute timesteps: 8 slots = 4 hours
        ts = [datetime(2020, 1, 1, h, m) for h in range(4) for m in (0, 30)]
        result = optimize(
            ts,
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[80] * 8)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify all on-blocks are ≥4 timesteps (= 2h at dt=0.5h)
        assert_on_blocks(on, min_length=4)

    def test_max_uptime_with_half_hour_timesteps(self):
        """max_uptime enforced correctly with 30-minute timesteps.

        Src: size=100, Status(max_uptime=1), prior_rates=[0], 1€/MWh.
        Backup: 10€/MWh.
        6 timesteps of 30min (3 hours total).
        Demand: [10]*6.

        max_uptime=1h = 2 timesteps at dt=0.5h. Src can run at most 2
        consecutive slots before forced shutdown. Pattern like
        [on,on,off,on,on,off] → Src covers 4 slots, Backup covers 2.
        """
        ts = [datetime(2020, 1, 1, h, m) for h in range(3) for m in (0, 30)]
        result = optimize(
            ts,
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10] * 6)]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=100,
                            relative_minimum=0.1,
                            effects_per_flow_hour={'costs': 1},
                            status=Status(max_uptime=1),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no on-block exceeds 2 timesteps (= 1h at dt=0.5h)
        assert_on_blocks(on, max_length=2)

        # Pattern [on,on,off,on,on,off]: Src 4*10*0.5=20, Backup 2*10*10*0.5=100. Total=120.
        assert_allclose(result.objective, 120.0, rtol=1e-5)
