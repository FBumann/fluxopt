"""Sizing (investment) optimization tests.

Each test builds a small model with investable flows or storages and asserts
that the solver chooses the correct capacity.
"""

from __future__ import annotations

from datetime import datetime

from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port, Sizing, Storage, optimize


def _ts(n: int) -> list[datetime]:
    """Create n hourly timesteps starting 2020-01-01."""
    return [datetime(2020, 1, 1, h) for h in range(n)]


def _waste(bus: str) -> Port:
    """Free-disposal port: absorbs excess on *bus* at zero cost."""
    return Port('waste', exports=[Flow(bus=bus)])


class TestFlowSizing:
    def test_mandatory_continuous_sizing(self):
        """Source with Sizing(10, 200, mandatory=True), demand=50.

        The solver must invest at least 50 MW to meet demand.
        Operational cost = 50 * 1€/MWh * 2h = 100.
        """
        result = optimize(
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
                            size=Sizing(10, 200, mandatory=True),
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        assert size >= 50.0 - 1e-5

    def test_optional_sizing_with_fixed_cost(self):
        """Optional invest with fixed cost.

        Src: Sizing(10, 100, effects_fixed={'costs': 200}), 1€/MWh operational.
        Backup: 5€/MWh, unsized.
        Demand = 50/ts * 2ts.

        Without invest: backup cost = 50 * 5 * 2 = 500.
        With invest: fixed=200 + operational=50*1*2=100 → total=300. Cheaper → invest.
        """
        result = optimize(
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
                            size=Sizing(10, 100, mandatory=False, effects_fixed={'costs': 200}),
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
            ],
        )
        assert_allclose(result.objective, 300.0, rtol=1e-5)
        indicator = float(result.solution['flow--size_indicator'].sel(flow='Src(Heat)').values)
        assert_allclose(indicator, 1.0, atol=1e-5)

    def test_fixed_size_binary_invest(self):
        """Binary invest at fixed size: Sizing(80, 80, effects_fixed={'costs': 10}).

        Demand=50. Src operational=1€/MWh. Backup=2€/MWh.
        With invest: 10 + 50*1*2 = 110.
        Without invest: 50*2*2 = 200.
        Should invest → size=80, cost=110.
        """
        result = optimize(
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
                            size=Sizing(80, 80, mandatory=False, effects_fixed={'costs': 10}),
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
                Port('Backup', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 2})]),
            ],
        )
        assert_allclose(result.objective, 110.0, rtol=1e-5)
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        assert_allclose(size, 80.0, rtol=1e-5)

    def test_per_size_effects(self):
        """Per-size investment cost.

        Sizing(0, 200, mandatory=True, effects_per_size={'costs': 5}).
        Demand=50/ts * 2ts. Operational=1€/MWh.
        Total = 5*size + 50*1*2. Solver minimizes → size=50.
        Total = 5*50 + 100 = 350.
        """
        result = optimize(
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
                            size=Sizing(0, 200, mandatory=True, effects_per_size={'costs': 5}),
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 350.0, rtol=1e-5)
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        assert_allclose(size, 50.0, rtol=1e-5)

    def test_investable_flow_with_fixed_profile(self):
        """Investable flow with fixed_relative_profile: P = profile * size.

        Profile=[0.5, 1.0], demand=[25, 50]. Operational=1€/MWh.
        Need size=50 (from t1: 50/1.0). t0: 0.5*50=25. Total cost=75.
        """
        result = optimize(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[25, 50])]),
                Port(
                    'Src',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 200, mandatory=True),
                            fixed_relative_profile=[0.5, 1.0],
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 75.0, rtol=1e-5)
        size = float(result.sizes.sel(flow='Src(Heat)').values)
        assert_allclose(size, 50.0, rtol=1e-5)

    def test_multiple_investable_merit_order(self):
        """Two investable sources with per-size cost, merit order picks cheapest.

        Cheap: Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.01}), 1€/MWh.
        Expensive: Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.01}), 5€/MWh.
        Demand=60. Cheap covers all → size=60, operational=120, invest=0.6 → total=120.6.
        Per-size cost incentivizes minimal size.
        """
        result = optimize(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[60, 60])]),
                Port(
                    'Cheap',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.01}),
                            effects_per_flow_hour={'costs': 1},
                        )
                    ],
                ),
                Port(
                    'Expensive',
                    imports=[
                        Flow(
                            bus='Heat',
                            size=Sizing(0, 100, mandatory=True, effects_per_size={'costs': 0.01}),
                            effects_per_flow_hour={'costs': 5},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 120.6, rtol=1e-4)
        cheap_size = float(result.sizes.sel(flow='Cheap(Heat)').values)
        expensive_size = float(result.sizes.sel(flow='Expensive(Heat)').values)
        assert_allclose(cheap_size, 60.0, rtol=1e-4)
        assert_allclose(expensive_size, 0.0, atol=1e-4)


class TestStorageSizing:
    def test_storage_capacity_sizing(self):
        """Storage capacity sizing with price arbitrage.

        Price=[1, 10, 1]. Demand=[0, 50, 0]. Storage capacity=Sizing(0, 500, mandatory=True).
        Optimal: charge 50 at t=0 @1€, discharge at t=1. Cost=50.
        Capacity ≥ 50.
        """
        result = optimize(
            _ts(3),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 50, 0])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 10, 1]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=100),
                    discharging=Flow(bus='Elec', size=100),
                    capacity=Sizing(0, 500, mandatory=True),
                    prior_level=0.0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-5)
        cap = float(result.storage_capacities.sel(storage='Battery').values)
        assert cap >= 50.0 - 1e-5
