"""Status (on/off) constraint tests.

Each test builds a small model with flows that have Status and verifies
semi-continuous behavior, startup costs, and running costs.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from conftest import assert_off_blocks, assert_on_blocks, ts, waste
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Status, optimize

_heat = [Carrier(id='Heat')]


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
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[30, 60, 0])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
                waste('Heat'),
            ],
        )
        assert_allclose(result.objective, 110.0, atol=1e-5)

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
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 80])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.4,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 45.0, rtol=1e-5)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert_allclose(on, [0.0, 0.0], atol=1e-5)


class TestStartupCosts:
    def test_startup_cost_added_to_objective(self):
        """Startup cost is charged per event.

        Source: size=100, prior_rates=[0] (was off), effects_per_startup={'cost': 50}, 1€/MWh.
        Demand: [60, 60] (constant). No backup.

        Source runs both hours: 1 startup event at t=0 (was off).
        Operational: 60*1*2 = 120. Startup: 50. Total: 170.
        """
        result = optimize(
            ts(2),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[60, 60])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(effects_per_startup={'cost': 50}),
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
                Status(effects_per_startup={'cost': 200}), 0.1€/MWh.
        Backup: 5€/MWh.
        Demand: [80, 0, 80].

        On all 3h: 1 startup=200 + (80+30+80)*0.1=219. Waste absorbs 30 at t=1.
        Cycling on/off/on: 2*200 + (80+80)*0.1=416.
        Stays on to avoid 2nd startup.
        """
        result = optimize(
            ts(3),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[80, 0, 80])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.3,
                            effects_per_flow_hour={'cost': 0.1},
                            status=Status(effects_per_startup={'cost': 200}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
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

        Source: size=100, Status(effects_per_startup={'cost': 1000}), no prior.
        Demand: [50, 50]. No backup, so source must run.

        Solver is free to assume on at t=-1 (no startup cost) or off (startup cost).
        With high startup cost, solver prefers to assume it was already on.
        Expected cost: 0 (no startup) + 0 (no flow cost) = 0.
        """
        result = optimize(
            ts(2),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            status=Status(effects_per_startup={'cost': 1000}),
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

        Source: size=100, uptime_min=3h, prior_rates=[50, 60] (2h on already).
        Demand: [80, 0, 0].

        With 2h of prior uptime and uptime_min=3h, source must stay on for
        at least 1 more hour. After that it can turn off.
        t=0: must stay on (uptime=3h total). Flow=80.
        t=1: can turn off. Demand=0, cheaper to turn off.
        t=2: off.
        """
        result = optimize(
            ts(3),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[80, 0, 0])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_min=3),
                            prior_rates=[50, 60],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 0.5})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        # t=0: forced on by uptime_min continuation
        assert_allclose(on[0], 1.0, atol=1e-5)

    def test_prior_off_carries_downtime(self):
        """Prior with consecutive off-hours carries downtime into the horizon.

        Source: size=100, downtime_min=3h, prior_rates=[0, 0] (2h off already).
        Demand: [80, 80, 80].

        With 2h of prior downtime and downtime_min=3h, source must stay off
        for at least 1 more hour.
        t=0: must stay off (downtime=3h total). Backup covers.
        t=1: can turn on.
        t=2: on.
        """
        result = optimize(
            ts(3),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[80, 80, 80])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(downtime_min=3),
                            prior_rates=[0, 0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        # t=0: forced off by downtime_min continuation
        assert_allclose(on[0], 0.0, atol=1e-5)
        # t=1, t=2: can and should turn on (cheaper than backup)
        assert_allclose(on[1], 1.0, atol=1e-5)
        assert_allclose(on[2], 1.0, atol=1e-5)

    def test_running_cost_per_hour(self):
        """Running cost is charged per hour the unit is on.

        Source: size=100, Status(effects_per_running_hour={'cost': 10}), 1€/MWh.
        Demand: [50, 50].

        Operational: 50*1*2 = 100. Running: 10*1*2 = 20.
        Total: 100 + 20 = 120.
        """
        result = optimize(
            ts(2),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(effects_per_running_hour={'cost': 10}),
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
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[30, 80, 0])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=20, size_max=200, mandatory=True),
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
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
        """Sizing + Status + relative_rate_min forces off when demand < min_load.

        Src: Sizing(0, 100, mandatory=True, effects_per_size={'cost': 0.5}),
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
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 50])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=0, size_max=100, mandatory=True, effects_per_size={'cost': 0.5}),
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
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

        Src: Sizing(50, 100, mandatory=False, effects_fixed={'cost': 1000}),
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
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 3)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=50, size_max=100, mandatory=False, effects_fixed={'cost': 1000}),
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 2})]),
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

        Src: Sizing(0, 200, mandatory=True), Status(effects_per_startup={'cost': 100}),
             prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [50, 50, 50].

        Src runs all 3h: 1 startup(100) + operational(150) + invest_cost(0) = 250.
        All backup: 50*5*3 = 750. Src is cheaper despite startup cost.
        """
        result = optimize(
            ts(3),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[50] * 3)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=0, size_max=200, mandatory=True),
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(effects_per_startup={'cost': 100}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values
        size = float(result.sizes.sel(flow='Src(Heat)').values)

        assert size >= 50.0 - 1e-5
        # Exactly 1 startup at t=0
        assert_allclose(np.sum(startup), 1.0, atol=1e-5)
        # Total: 100 (startup) + 150 (operational) = 250
        assert_allclose(result.objective, 250.0, rtol=1e-5)

    def test_sizing_with_uptime_min(self):
        """Sizing + Status + uptime_min: duration constraint with variable capacity.

        Src: Sizing(0, 200, mandatory=True), rel_min=0.3,
             Status(uptime_min=3), prior_rates=[0], 1€/MWh.
        Backup: 0.5€/MWh (cheaper).
        Demand: [80, 0, 0, 80].

        Backup is cheaper, but once Src turns on, uptime_min=3 forces it to
        stay on for 3 consecutive hours. If Src turns on at t=0 for demand,
        it must stay on through t=2 (even with zero demand), producing at
        least 0.3*S each hour. Waste absorbs excess at t=1,t=2.
        """
        result = optimize(
            ts(4),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[80, 0, 0, 80])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=0, size_max=200, mandatory=True),
                            relative_rate_min=0.3,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_min=3),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 0.5})]),
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
    def test_uptime_max_forces_shutdown(self):
        """uptime_max=2 limits continuous operation to 2 consecutive hours.

        Src: size=100, Status(uptime_max=2), prior_rates=[0] (was off), 1€/MWh.
        Backup: 10€/MWh.
        Demand: [10, 10, 10, 10, 10].

        Without uptime_max: Src runs all 5h → cost=50.
        With uptime_max=2: Src runs at most 2 consecutive hours, then must
        shut down for ≥1h. Pattern like [on,on,off,on,on] → Src covers 4h,
        Backup covers 1h at 10€ → total=40+10=50... but the waste of backup
        hour makes it 4*10 + 1*10*10 = 140? No:
        Src 4h: 4*10*1 = 40. Backup 1h: 1*10*10 = 100. Total = 140.
        Without: 5*10*1 = 50. So cost > 50 proves the constraint works.
        """
        result = optimize(
            ts(5),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_max=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no more than 2 consecutive on-hours
        assert_on_blocks(on, max_length=2)

        # Src 4h: 4*10*1=40. Backup 1h: 1*10*10=100. Total=140.
        assert_allclose(result.objective, 140.0, rtol=1e-5)

    def test_shutdown_feasible_on_irregular_grid(self):
        """A prior-on unit can shut down when timestep durations vary.

        Grid: [00:00, 04:00, 05:00] → dt=[4, 4, 1]. Src: size=10,
        rel_min=0.5, prior_rates=[10] (on, 4h prior uptime), 1€/MWh.
        Demand: [10, 10, 0] — step 2 has no demand and no waste, so Src
        must shut off there. Src serves steps 0-1: 10*(4+4) = 80 MWh → 80.

        Sensitivity: the duration recursion must add the entered step's dt
        (dt[t+1]). With the departed step's dt and the longest-episode
        Big-M (M = 9+4 = 13), the forced uptime D[1] = 12 would require
        D[2] >= 12 + 4 - 13 = 3 while off caps D[2] at 0 — infeasible.
        """
        result = optimize(
            [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 4), datetime(2024, 1, 1, 5)],
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10, 0])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=10,
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_max=100),
                            prior_rates=[10],
                        )
                    ],
                ),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert list(on) == [1, 1, 0]
        assert_allclose(result.objective, 80.0, rtol=1e-5)


class TestMaxDowntime:
    def test_downtime_max_forces_restart(self):
        """downtime_max=1 prevents staying off for more than 1 consecutive hour.

        Src: size=100, rel_min=0.5, Status(downtime_max=1), prior_rates=[10] (was on),
             10€/MWh (expensive).
        Backup: 1€/MWh (cheap).
        Demand: [10, 10, 10, 10].

        Without downtime_max: all from cheap Backup → cost=40.
        With downtime_max=1: Src can be off at most 1 consecutive hour. Since
        it was previously on, it can turn off but must restart within 1h.
        This forces Src on for ≥2 of 4 hours → cost > 40.
        """
        result = optimize(
            ts(4),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 4)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 10},
                            status=Status(downtime_max=1),
                            prior_rates=[10],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no two consecutive off-hours
        assert_off_blocks(on, max_length=1, skip_leading=False)

        # Pattern [on,off,on,off]: Src min=50 when on, 2*50*10=1000, Backup 2*10*1=20. Total=1020.
        assert_allclose(result.objective, 1020.0, rtol=1e-5)


class TestDurationCombinations:
    def test_min_and_uptime_max_forces_exact_blocks(self):
        """uptime_min=2 + uptime_max=2 forces operation in exact 2-hour blocks.

        Src: size=100, Status(uptime_min=2, uptime_max=2), prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [5, 10, 20, 18, 12].

        With min=max=2h blocks, best pattern is [on,on,off,on,on]:
        Src covers t=0,1,3,4; Backup covers t=2.
        Src cost: (5+10+18+12)*1 = 45. Backup cost: 20*5 = 100. Total = 145.
        """
        result = optimize(
            ts(5),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 10, 20, 18, 12])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.01,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_min=2, uptime_max=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert_allclose(on, [1, 1, 0, 1, 1], atol=1e-5)
        assert_allclose(result.objective, 145.0, rtol=1e-5)

    def test_uptime_min_with_downtime_min_block_pattern(self):
        """uptime_min=2 + downtime_min=2 forces on/off blocks of ≥2 hours each.

        Src: size=100, rel_min=0.1, Status(uptime_min=2, downtime_min=2),
             prior_rates=[0], 1€/MWh.
        Backup: 5€/MWh.
        Demand: [20]*6.

        Must run in ≥2h blocks, off in ≥2h blocks. From prior off, stays off
        ≥2h then on ≥2h. Patterns like [off,off,on,on,on,on] or
        [off,off,on,on,off,off]. Cheapest: maximize Src hours.
        """
        result = optimize(
            ts(6),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[20] * 6)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_min=2, downtime_min=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
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

    def test_uptime_max_with_prior_carry_over(self):
        """Prior uptime reduces remaining allowed on-time at start of horizon.

        Src: size=100, Status(uptime_max=3), prior_rates=[50, 50] (2h on already),
             1€/MWh.
        Backup: 10€/MWh.
        Demand: [10]*5.

        With 2h prior uptime and uptime_max=3, Src can run at most 1 more
        hour before forced shutdown. Then can restart for up to 3h.
        """
        result = optimize(
            ts(5),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_max=3),
                            prior_rates=[50, 50],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # t=0 should be on (continuing from prior), then forced off by uptime_max=3
        assert_allclose(on[0], 1.0, atol=1e-5)

        # Verify no run of >3 consecutive on-hours within horizon
        assert_on_blocks(on, max_length=3)

        # Pattern [on,off,on,on,on]: Src 4h*10=40, Backup 1h*10*10=100. Total=140.
        assert_allclose(result.objective, 140.0, rtol=1e-5)

    def test_uptime_max_with_startup_costs(self):
        """uptime_max forces shutdowns which incur startup costs on restart.

        Src: size=100, Status(uptime_max=2, effects_per_startup={'cost': 50}),
             prior_rates=[0], 1€/MWh.
        Backup: 10€/MWh.
        Demand: [10]*5.

        uptime_max=2 forces at least 1 shutdown in 5h. Restarting costs 50€
        each time. Pattern [on,on,off,on,on] = 2 startups = 100€ startup +
        40€ operational + 100€ backup = 240€.
        Without uptime_max: 1 startup = 50 + 50 operational = 100€.
        """
        result = optimize(
            ts(5),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 5)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_max=2, effects_per_startup={'cost': 50}),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        startup = result.solution['flow--startup'].sel(flow='Src(Heat)').values

        # Verify uptime_max constraint
        assert_on_blocks(on, max_length=2)

        # Exactly 2 startups (initial + restart after forced shutdown)
        assert_allclose(np.sum(startup), 2.0, atol=1e-5)

        # 2*50 startups + 4h*10 Src + 1h*10*10 Backup = 240
        assert_allclose(result.objective, 240.0, rtol=1e-5)

    def test_downtime_max_with_prior_carry_over(self):
        """Prior downtime reduces remaining allowed off-time at start of horizon.

        Src: size=100, rel_min=0.5, Status(downtime_max=2),
             prior_rates=[0, 0] (2h off already), 10€/MWh (expensive).
        Backup: 1€/MWh (cheap).
        Demand: [10]*4.

        With 2h prior downtime and downtime_max=2, Src must restart
        immediately at t=0 (can't stay off any longer).
        """
        result = optimize(
            ts(4),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 4)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 10},
                            status=Status(downtime_max=2),
                            prior_rates=[0, 0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
                waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # With 2h prior off and downtime_max=2, must turn on at t=0
        assert_allclose(on[0], 1.0, atol=1e-5)

    def test_uptime_min_with_half_hour_timesteps(self):
        """Duration constraints work correctly with sub-hourly timesteps.

        Src: size=100, Status(uptime_min=2), prior_rates=[0], 1€/MWh.
        Backup: 0.5€/MWh (cheaper).
        8 timesteps of 30min each (4 hours total).
        Demand: [0,0,0,0, 80,80, 0,0] (demand only at t=4,5 → hours 2-3).

        uptime_min=2h = 4 timesteps at dt=0.5h. Once Src turns on it must
        stay on for 4 consecutive timesteps. Backup is cheaper, so Src only
        runs if forced. Demand at t=4,5 needs coverage. With uptime_min=4ts,
        turning on at t=4 means on through t=7 (or t=4-7). Src runs 4 slots
        at 1€/MWh, costing (80+80+0+0)*0.5*1 = 80. But also Backup may cover
        the demand slots cheaper. Since Backup is 0.5€/MWh: 80*0.5*0.5 +
        80*0.5*0.5 = 40. So all-backup = 40.

        But if Src turns on at all (forced by some mechanism), it must stay
        on 4 timesteps. Let's instead make Src cheaper but with startup cost
        to make it interesting.

        Revised: Src=1€/MWh, Backup=5€/MWh. Demand=[80]*8.
        Src must run for demand. uptime_min=2h → once on, stays on ≥4 slots.
        Src runs all 8 slots: cost = 80*0.5*8*1 = 320.
        Verify on-blocks are ≥4 timesteps (=2h).
        """
        # 30-minute timesteps: 8 slots = 4 hours
        half_hour_ts = [datetime(2020, 1, 1, h, m) for h in range(4) for m in (0, 30)]
        result = optimize(
            half_hour_ts,
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[80] * 8)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_min=2),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify all on-blocks are ≥4 timesteps (= 2h at dt=0.5h)
        assert_on_blocks(on, min_length=4)

    def test_uptime_max_with_half_hour_timesteps(self):
        """uptime_max enforced correctly with 30-minute timesteps.

        Src: size=100, Status(uptime_max=1), prior_rates=[0], 1€/MWh.
        Backup: 10€/MWh.
        6 timesteps of 30min (3 hours total).
        Demand: [10]*6.

        uptime_max=1h = 2 timesteps at dt=0.5h. Src can run at most 2
        consecutive slots before forced shutdown. Pattern like
        [on,on,off,on,on,off] → Src covers 4 slots, Backup covers 2.
        """
        half_hour_ts = [datetime(2020, 1, 1, h, m) for h in range(3) for m in (0, 30)]
        result = optimize(
            half_hour_ts,
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10] * 6)]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.1,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(uptime_max=1),
                            prior_rates=[0],
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10})]),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values

        # Verify no on-block exceeds 2 timesteps (= 1h at dt=0.5h)
        assert_on_blocks(on, max_length=2)

        # Pattern [on,on,off,on,on,off]: Src 4*10*0.5=20, Backup 2*10*10*0.5=100. Total=120.
        assert_allclose(result.objective, 120.0, rtol=1e-5)


class TestStatusBuildGating:
    def test_built_size_below_one_can_run(self):
        """A built unit with size < 1 must be allowed to switch on.

        Src: Sizing(0.2, 0.5, mandatory=True), rel_min=0.5, Status(), 1€/MWh.
        Backup: 100€/MWh. Demand: [0.3, 0.4].

        Optimal: size=0.4, Src covers both steps (cost 0.3+0.4=0.7 with
        waste absorbing the t=0 excess of min-load if any; min at size 0.4
        is 0.2 <= 0.3, so exact). The former ``on <= size`` constraint
        forced on=0 for any size < 1, pushing everything to the backup.
        """
        result = optimize(
            ts(2),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[0.3, 0.4])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=0.2, size_max=0.5, mandatory=True),
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 100})]),
                waste('Heat'),
            ],
        )
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert size < 1.0  # the interesting regime
        assert_allclose(on, [1.0, 1.0], atol=1e-5)
        assert_allclose(result.objective, 0.7, atol=1e-5)

    def test_min_positive_size_gates_on(self):
        """With size_min > 0, staying off entirely avoids building (optional sizing).

        Src: Sizing(50, 100, mandatory=False, effects_per_size={'cost': 10}),
        rel_min=0.5, Status(), 1€/MWh. Backup: 2€/MWh. Demand: [5, 5].

        Building 50 costs 500 + operation; backup covers 10 MWh for 20.
        Optimal: never build, Src stays off both steps.
        """
        result = optimize(
            ts(2),
            carriers=_heat,
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5])]),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=50, size_max=100, mandatory=False, effects_per_size={'cost': 10}),
                            relative_rate_min=0.5,
                            effects_per_flow_hour={'cost': 1},
                            status=Status(),
                        )
                    ],
                ),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 2})]),
                waste('Heat'),
            ],
        )
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        assert_allclose(size, 0.0, atol=1e-5)
        assert_allclose(on, [0.0, 0.0], atol=1e-5)
        assert_allclose(result.objective, 20.0, atol=1e-5)


class TestStatusSizingProfileRejected:
    def test_element_level_rejection(self):
        """Profile + Status + Sizing has no formulation and fails at authoring."""
        import pytest

        with pytest.raises(ValueError, match='fixed_relative_profile is not supported together with'):
            Flow(
                carrier='Heat',
                size=Sizing(size_min=10, size_max=100),
                fixed_relative_profile=[0.5, 0.8],
                relative_rate_min=0.1,
                status=Status(),
            )


class TestPriorNonUniformDtWarning:
    def test_warns_when_prior_rates_meet_irregular_grid(self):
        """Non-uniform dt + prior_rates warns about the assumed prior-step duration."""
        import pytest

        from fluxopt import ModelData

        with pytest.warns(UserWarning, match='pre-horizon status durations assume'):
            ModelData.build(
                [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 3)],
                carriers=_heat,
                effects=[Effect(id='cost')],
                ports=[
                    Port(
                        id='Src',
                        imports=[
                            Flow(
                                carrier='Heat',
                                size=10,
                                relative_rate_min=0.5,
                                status=Status(uptime_min=2.0),
                                prior_rates=[5.0, 5.0],
                            )
                        ],
                    ),
                ],
            )
