"""Status (on/off) constraint tests.

Each test builds a small model with flows that have Status and verifies
semi-continuous behavior, startup costs, and running costs.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port, Sizing, Status, solve


def _ts(n: int) -> list[datetime]:
    """Create n hourly timesteps starting 2020-01-01."""
    return [datetime(2020, 1, 1, h) for h in range(n)]


def _waste(bus: str) -> Port:
    """Free-disposal port: absorbs excess on *bus* at zero cost."""
    return Port('waste', exports=[Flow(bus=bus)])


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
        result = solve(
            _ts(3),
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
                _waste('Heat'),
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
        result = solve(
            _ts(2),
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

        Source: size=100, prior=[0] (was off), effects_per_startup={'costs': 50}, 1€/MWh.
        Demand: [60, 60] (constant). No backup.

        Source runs both hours: 1 startup event at t=0 (was off).
        Operational: 60*1*2 = 120. Startup: 50. Total: 170.
        """
        result = solve(
            _ts(2),
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
                            effects_per_flow_hour={'costs': 1},
                            status=Status(effects_per_startup={'costs': 50}),
                            prior=[0],
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 170.0, rtol=1e-5)

    def test_startup_cost_discourages_cycling(self):
        """High startup cost keeps unit running rather than cycling.

        Source: size=100, rel_min=0.3, prior=[0] (was off),
                Status(effects_per_startup={'costs': 200}), 0.1€/MWh.
        Backup: 5€/MWh.
        Demand: [80, 0, 80].

        On all 3h: 1 startup=200 + (80+30+80)*0.1=219. Waste absorbs 30 at t=1.
        Cycling on/off/on: 2*200 + (80+80)*0.1=416.
        Stays on to avoid 2nd startup.
        """
        result = solve(
            _ts(3),
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
                            prior=[0],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
                _waste('Heat'),
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
        result = solve(
            _ts(2),
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

        Source: size=100, min_uptime=3h, prior=[50, 60] (2h on already).
        Demand: [80, 0, 0].

        With 2h of prior uptime and min_uptime=3h, source must stay on for
        at least 1 more hour. After that it can turn off.
        t=0: must stay on (uptime=3h total). Flow=80.
        t=1: can turn off. Demand=0, cheaper to turn off.
        t=2: off.
        """
        result = solve(
            _ts(3),
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
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_uptime=3),
                            prior=[50, 60],
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 0.5})]),
                _waste('Heat'),
            ],
        )
        on = result.solution['flow--on'].sel(flow='Src(Heat)').values
        # t=0: forced on by min_uptime continuation
        assert_allclose(on[0], 1.0, atol=1e-5)

    def test_prior_off_carries_downtime(self):
        """Prior with consecutive off-hours carries downtime into the horizon.

        Source: size=100, min_downtime=3h, prior=[0, 0] (2h off already).
        Demand: [80, 80, 80].

        With 2h of prior downtime and min_downtime=3h, source must stay off
        for at least 1 more hour.
        t=0: must stay off (downtime=3h total). Backup covers.
        t=1: can turn on.
        t=2: on.
        """
        result = solve(
            _ts(3),
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
                            effects_per_flow_hour={'costs': 1},
                            status=Status(min_downtime=3),
                            prior=[0, 0],
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

        Operational: 50*1*2 = 100. Running: 10*1*2 = 20. Startup: 10*1 = 10 (1 event).
        Total: 100 + 20 = 120.
        """
        result = solve(
            _ts(2),
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
        result = solve(
            _ts(3),
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
                _waste('Heat'),
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
