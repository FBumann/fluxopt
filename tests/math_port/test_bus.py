"""Mathematical correctness tests for bus balance & dispatch."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port

from .conftest import ts


class TestBusBalance:
    def test_merit_order_dispatch(self, optimize):
        """Proves: Bus balance forces total supply = demand, and the optimizer
        dispatches sources in merit order (cheapest first, up to capacity).

        Src1: 1€/kWh, max 20. Src2: 2€/kWh, max 20. Demand=30 per timestep.
        Optimal: Src1=20, Src2=10.

        Sensitivity: If bus balance allowed oversupply, Src2 could be zero → cost=40.
        If merit order were wrong (Src2 first), cost=100. Only correct bus balance
        with merit order yields cost=80 and the exact flow split [20,10].
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port(
                    'Src1',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 1}, size=20),
                    ],
                ),
                Port(
                    'Src2',
                    imports=[
                        Flow(bus='Heat', effects_per_flow_hour={'cost': 2}, size=20),
                    ],
                ),
            ],
        )
        # Src1 at max 20 @1€, Src2 covers remaining 10 @2€
        # cost = 2*(20*1 + 10*2) = 80
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        # Verify individual flows to confirm dispatch split
        src1 = result.flow_rate('Src1(Heat)').values
        src2 = result.flow_rate('Src2(Heat)').values
        assert_allclose(src1, [20, 20], rtol=1e-5)
        assert_allclose(src2, [10, 10], rtol=1e-5)

    @pytest.mark.skip(reason='imbalance_penalty not supported in fluxopt')
    def test_imbalance_penalty(self, optimize):
        """Proves: imbalance_penalty_per_flow_hour creates a 'Penalty' effect that
        charges for any mismatch between supply and demand on a bus.

        Source fixed at 20, demand=10 → 10 excess per timestep, penalty=100€/kWh.

        Sensitivity: Without the penalty mechanism, objective=40 (fuel only).
        With penalty, objective=2040 (fuel 40 + penalty 2000). The penalty is
        tracked in a separate 'Penalty' effect, not in 'costs'.
        """
        raise NotImplementedError  # TODO: rewrite for fluxopt API (see tests/math/test_bus_penalty.py)

    @pytest.mark.skip(reason='prevent_simultaneous not supported in fluxopt')
    def test_prevent_simultaneous_flow_rates(self, optimize):
        """Proves: prevent_simultaneous_flow_rates on a Source prevents multiple outputs
        from being active at the same time, forcing sequential operation.

        Source with 2 outputs to 2 buses. Both buses have demand=10 each timestep.
        Output1: 1€/kWh, Output2: 1€/kWh. Without exclusion, both active → cost=40.
        With exclusion, only one output per timestep → must use expensive backup (5€/kWh)
        for the other bus.

        Sensitivity: Without prevent_simultaneous, cost=40. With it, cost=2*(10+50)=120.
        """
        raise NotImplementedError  # TODO: implement prevent_simultaneous_flow_rates
