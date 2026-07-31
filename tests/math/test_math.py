"""Mathematical correctness tests ported from flixopt/tests/test_math.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.
Sensitivity comments explain what would break if the feature were disabled.

API mapping (flixopt -> fluxopt):
    fx.Source('name', outputs=[...])     -> Port('name', imports=[...])
    fx.Sink('name', inputs=[...])        -> Port('name', exports=[...])
    fx.Flow('label', bus=..., ...)       -> Flow(carrier, ...)
    effects_per_flow_hour=<scalar>       -> effects_per_flow_hour={'cost': <scalar>}
    capacity_in_flow_hours=X             -> capacity=X
    initial_charge_state='equals_final'  -> cyclic=True
    imbalance_penalty_per_flow_hour=0    -> waste Port (absorbs excess at zero cost)
"""

from __future__ import annotations

import pytest
from conftest import ts, waste
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, Port, Sizing, Status, Storage, optimize

# ---------------------------------------------------------------------------
# Bus balance & dispatch
# ---------------------------------------------------------------------------


class TestBusBalance:
    def test_merit_order_dispatch(self):
        """Src1: 1EUR/kWh, max 20. Src2: 2EUR/kWh, max 20. Demand=30/ts.
        Optimal: Src1=20, Src2=10.

        Sensitivity: Without merit order, cost could be 100 (Src2 first).
        Only correct bus balance with merit order yields cost=80.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[30, 30])]),
                Port(id='Src1', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1}, size=20)]),
                Port(id='Src2', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 2}, size=20)]),
            ],
        )
        assert_allclose(result.objective, 80.0, rtol=1e-5)
        src1 = result.flow_rate('Src1(Heat)').values
        src2 = result.flow_rate('Src2(Heat)').values
        assert_allclose(src1, [20, 20], rtol=1e-5)
        assert_allclose(src2, [10, 10], rtol=1e-5)


# ---------------------------------------------------------------------------
# Conversion & efficiency
# ---------------------------------------------------------------------------


class TestConversionEfficiency:
    def test_boiler_efficiency(self):
        """Boiler eta=0.8, demand=[10,20,10]. fuel = 40/0.8 = 50.

        Sensitivity: If eta ignored (1.0), cost=40 instead of 50.
        """

        fuel = Flow(carrier='Gas')
        thermal = Flow(carrier='Heat')
        result = optimize(
            ts(3),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 20, 10])]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[Converter.boiler('Boiler', 0.8, fuel, thermal)],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-5)

    def test_variable_efficiency(self):
        """Boiler eta=[0.5, 1.0], demand=[10,10]. fuel = 10/0.5 + 10/1.0 = 30.

        Sensitivity: Scalar mean (0.75) -> 26.67. Only per-timestep yields 30.
        """

        fuel = Flow(carrier='Gas')
        thermal = Flow(carrier='Heat')
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[Converter.boiler('Boiler', [0.5, 1.0], fuel, thermal)],
        )
        assert_allclose(result.objective, 30.0, rtol=1e-5)

    def test_chp_dual_output(self):
        """CHP eta_th=0.5, eta_el=0.4. demand=50 heat/ts. Elec sold at -2EUR/kWh.
        fuel=50/0.5=100, elec=100*0.4=40. cost/ts = 100*1 - 40*2 = 20. total=40.

        Sensitivity: If eta_el broken, cost=200. If eta_th wrong (1.0), cost=-60.
        """

        fuel = Flow(carrier='Gas')
        thermal = Flow(carrier='Heat')
        electrical = Flow(carrier='Elec')
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat'), Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='HeatDemand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port(id='ElecGrid', exports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': -2})]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[Converter.chp('CHP', 0.4, 0.5, fuel, electrical, thermal)],
        )
        assert_allclose(result.objective, 40.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Effects & objective
# ---------------------------------------------------------------------------


class TestEffects:
    def test_effects_per_flow_hour(self):
        """costs=2EUR/kWh, CO2=0.5kg/kWh. Total flow=30.
        costs=60, CO2=15.

        Sensitivity: If only one effect applied, the other would be wrong.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 20])]),
                Port(id='HeatSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 2, 'CO2': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 60.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 15.0, rtol=1e-5)

    def test_effect_maximum(self):
        """CO2 capped at 15. Dirty: 1EUR+1kgCO2/kWh. Clean: 10EUR+0kgCO2.
        Demand=20. Split: 15 Dirty + 5 Clean -> cost=65.

        Sensitivity: Without CO2 cap, all Dirty -> cost=20.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', total_max=15)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 10, 'CO2': 0})]),
            ],
        )
        assert_allclose(result.objective, 65.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 15.0, rtol=1e-5)

    def test_effect_minimum(self):
        """CO2 floor at 25. Dirty: 1EUR+1kgCO2. Demand=20. Must overproduce.
        Dirty=25 (5 excess absorbed by waste port). cost=25.

        Sensitivity: Without minimum, Dirty=20 -> cost=20.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', total_min=25)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 0})]),
                waste('Heat'),
            ],
        )
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 25.0, rtol=1e-5)
        assert_allclose(result.objective, 25.0, rtol=1e-5)

    def test_effect_maximum_temporal(self):
        """CO2 total_max=12 (= maximum_temporal when no periodic effects).
        Dirty: 1EUR+1kgCO2. Clean: 5EUR+0kgCO2. Demand=[10,10].
        Dirty=12, Clean=8 -> cost=52.

        Sensitivity: Without cap, all Dirty -> cost=20.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', total_max=12)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5, 'CO2': 0})]),
            ],
        )
        assert_allclose(result.objective, 52.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 12.0, rtol=1e-5)

    def test_effect_minimum_temporal(self):
        """CO2 total_min=25 (= minimum_temporal). Dirty: 1EUR+1kgCO2.
        Demand=[10,10]. Dirty >=25 -> 5 excess. cost=25.

        Sensitivity: Without floor, Dirty=20 -> cost=20.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', total_min=25)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                waste('Heat'),
            ],
        )
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 25.0, rtol=1e-5)
        assert_allclose(result.objective, 25.0, rtol=1e-5)

    def test_effect_periodic_max(self):
        """CO2 periodic_max=8 caps each period independently.

        2 periods (weights=1), demand=10 per ts. Per-period: Dirty<=8 (CO2 cap),
        Clean>=12. cost per period = 8+60 = 68. Objective = 2*68 = 136.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_max=8)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5, 'CO2': 0})]),
            ],
            periods=[2020, 2025],
            period_weights=[1, 1],
        )
        # Each period: total demand 20, Dirty<=8, Clean=12. cost = 8 + 60 = 68 per period.
        assert_allclose(result.objective, 136.0, rtol=1e-5)

    def test_effect_periodic_min(self):
        """CO2 periodic_min=15 forces minimum emission each period.

        2 periods (weights=1), demand=5 per ts. Each period needs >=15 CO2.
        Dirty >= 15 per period, demand = 10 per period -> 5 excess to waste.
        cost = 15 per period, total = 30.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_min=15)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                waste('Heat'),
            ],
            periods=[2020, 2025],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 30.0, rtol=1e-5)

    def test_effect_periodic_max_per_period_values(self):
        """periodic_max accepts per-period values: [8, 12] caps periods differently.

        2 periods (weights=1), demand=10 per ts (20 per period).
        Period 2020: Dirty<=8, Clean=12 -> cost 8 + 60 = 68.
        Period 2025: Dirty<=12, Clean=8 -> cost 12 + 40 = 52.
        Objective = 68 + 52 = 120.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_max=[8, 12])],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5, 'CO2': 0})]),
            ],
            periods=[2020, 2025],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 120.0, rtol=1e-5)

    def test_effect_periodic_bound_array_requires_periods(self):
        """Per-period bound values without a period axis fail loudly at build."""
        with pytest.raises(ValueError):
            optimize(
                ts(2),
                carriers=[Carrier(id='Heat')],
                effects=[Effect(id='cost'), Effect(id='CO2', periodic_max=[8, 12])],
                objective='cost',
                ports=[
                    Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                    Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                ],
            )

    def test_effect_maximum_multi_period_weighted(self):
        """maximum bound across multi-period uses period_weights for the weighted sum.

        2 periods, weights=[1,1], Effect.total_max=20. 3 timesteps x demand=5 per period.
        Total CO2 cap = 1*co2[0] + 1*co2[1] <= 20.
        Per-period demand = 15 → all Dirty would give CO2=15 per period, sum=30.
        Capped at 20: Dirty=20 total, Clean=10. cost = 20*1 + 10*5 = 70.
        """

        result = optimize(
            ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', total_max=20)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5, 5])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5, 'CO2': 0})]),
            ],
            periods=[2020, 2025],
            period_weights=[1, 1],
        )
        co2 = float(result.effect_totals.sel(effect='CO2').sum().item())
        # Total CO2 (weighted by [1,1]) <= 20.
        assert co2 <= 20 + 1e-5
        assert_allclose(result.objective, 70.0, rtol=1e-5)

    @pytest.mark.parametrize(
        ('chain_effects', 'lump_effect'),
        [
            pytest.param([Effect(id='co2')], 'co2', id='direct'),
            pytest.param([Effect(id='co2', contribution_from={'pe': 0.2}), Effect(id='pe')], 'pe', id='transitive'),
        ],
    )
    def test_effect_time_varying_contribution_into_lump_bearing_raises(self, chain_effects, lump_effect):
        """Time-varying contribution_from is rejected when the source effect has lump contributions.

        Sizing on the source creates the lump contribution — directly on co2, or
        on pe with a scalar chain co2 <- pe (the check follows chains).
        """
        source = Flow(
            carrier='Heat',
            effects_per_flow_hour={lump_effect: 1},
            size=Sizing(size_min=10, size_max=10, mandatory=True, effects_per_size={lump_effect: 1.0}),
        )
        with pytest.raises(ValueError, match='ill-defined'):
            optimize(
                ts(2),
                carriers=[Carrier(id='Heat')],
                effects=[Effect(id='cost', contribution_from={'co2': [1.0, 2.0]}), *chain_effects],
                objective='cost',
                ports=[
                    Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5])]),
                    Port(id='Source', imports=[source]),
                ],
            )

    def test_effect_time_varying_contribution_without_lump(self):
        """Time-varying contribution_from is fine when the source effect is purely temporal.

        co2 = [5, 5] per timestep, factor = [1, 2] -> cost = 5*1 + 5*2 = 15.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost', contribution_from={'co2': [1.0, 2.0]}),
                Effect(id='co2'),
            ],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5])]),
                Port(id='Source', imports=[Flow(carrier='Heat', effects_per_flow_hour={'co2': 1})]),
            ],
        )
        assert_allclose(result.objective, 15.0, rtol=1e-5)

    def test_effect_constant_contribution_does_not_warn(self):
        """A constant contribution_from must not trip the time-varying warning (mean != value in float)."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            optimize(
                ts(3),
                carriers=[Carrier(id='Heat')],
                effects=[
                    Effect(id='cost', contribution_from={'co2': 0.045}),
                    Effect(id='co2'),
                ],
                objective='cost',
                ports=[
                    Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[5, 5, 5])]),
                    Port(
                        id='Source',
                        imports=[
                            Flow(
                                carrier='Heat',
                                effects_per_flow_hour={'co2': 1},
                                size=Sizing(size_min=10, size_max=10, mandatory=True, effects_per_size={'co2': 1.0}),
                            ),
                        ],
                    ),
                ],
            )
        msgs = [str(w.message) for w in caught]
        assert not any('averaged over time' in m for m in msgs), f'Unexpected warning: {msgs}'


# ---------------------------------------------------------------------------
# Flow constraints
# ---------------------------------------------------------------------------


class TestFlowConstraints:
    def test_relative_rate_min(self):
        """Boiler (size=100, relative_rate_min=0.4). When on, must produce >=40.
        Demand=30 -> excess absorbed by waste. fuel = 40/1.0 = 40. cost=80.

        Sensitivity: Without relative_rate_min, boiler=30 -> cost=60.
        """

        fuel = Flow(carrier='Gas')
        thermal = Flow(carrier='Heat', size=100, relative_rate_min=0.4)
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[30, 30])]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
                waste('Heat'),
            ],
            converters=[Converter.boiler('Boiler', 1.0, fuel, thermal)],
        )
        assert_allclose(result.objective, 80.0, rtol=1e-5)
        flow = result.flow_rate('Boiler(Heat)').values
        assert all(f >= 40.0 - 1e-5 for f in flow), f'Flow below relative_rate_min: {flow}'

    def test_relative_rate_max(self):
        """Source (size=100, relative_rate_max=0.5). Max output=50.
        Demand=60 -> CheapSrc=50, ExpensiveSrc=10. cost=2*(50*1+10*5)=200.

        Sensitivity: Without relative_rate_max, all from CheapSrc -> cost=120.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[60, 60])]),
                Port(
                    id='CheapSrc',
                    imports=[Flow(carrier='Heat', size=100, relative_rate_max=0.5, effects_per_flow_hour={'cost': 1})],
                ),
                Port(id='ExpensiveSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        assert_allclose(result.objective, 200.0, rtol=1e-5)
        flow = result.flow_rate('CheapSrc(Heat)').values
        assert all(f <= 50.0 + 1e-5 for f in flow), f'Flow above relative_rate_max: {flow}'

    def test_flow_hours_max_per_period(self):
        """flow_hours_max bounds each period independently.

        2 periods (weights=1), demand=[10,10] per ts. CheapSrc flow_hours_max=15.
        Each period: Cheap=15, Expensive=5 -> cost = 15 + 25 = 40 per period.
        Objective = 80.

        Sensitivity: A whole-horizon bound of 15 would force Cheap<=15 across
        both periods (cost >= 115); per-period allows 15 in each.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(
                    id='CheapSrc', imports=[Flow(carrier='Heat', flow_hours_max=15, effects_per_flow_hour={'cost': 1})]
                ),
                Port(id='ExpensiveSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
            periods=[2020, 2025],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 80.0, rtol=1e-5)
        cheap = result.flow_rate('CheapSrc(Heat)')
        for p in (2020, 2025):
            per_period = float(cheap.where(cheap.time_period == p).sum().values)
            assert per_period <= 15.0 + 1e-5, f'CheapSrc above flow_hours_max in period {p}: {per_period}'

    def test_flow_hours_max_respects_timestep_duration(self):
        """flow_hours_max is an energy bound: rates are weighted by dt.

        2 timesteps of 4 h, demand 10 MW -> 80 MWh. CheapSrc flow_hours_max=40
        caps cheap energy at 40 MWh; ExpensiveSrc covers the rest.
        cost = 40*1 + 40*5 = 240.

        Sensitivity: summing rates without dt reads 20 "flow hours" for the
        full cheap supply, the cap never binds, and the objective drops to 80.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(
                    id='CheapSrc', imports=[Flow(carrier='Heat', flow_hours_max=40, effects_per_flow_hour={'cost': 1})]
                ),
                Port(id='ExpensiveSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
            dt=4.0,
        )
        assert_allclose(result.objective, 240.0, rtol=1e-5)

    def test_flow_hours_min_respects_timestep_duration(self):
        """flow_hours_min forces minimum energy, not a minimum rate sum.

        2 timesteps of 4 h, demand 10 MW -> 80 MWh. ExpensiveSrc
        flow_hours_min=40 forces 40 MWh expensive; cheap covers the rest.
        cost = 40*1 + 40*5 = 240.

        Sensitivity: without dt the floor reads as 40 rate-units = 160 MWh,
        exceeding total demand — the model turns infeasible.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='CheapSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
                Port(
                    id='ExpensiveSrc',
                    imports=[Flow(carrier='Heat', flow_hours_min=40, effects_per_flow_hour={'cost': 5})],
                ),
            ],
            dt=4.0,
        )
        assert_allclose(result.objective, 240.0, rtol=1e-5)

    def test_load_factor_requires_size(self):
        """load_factor bounds on an unsized flow fail loudly at element level."""
        with pytest.raises(ValueError, match='load_factor'):
            Flow(carrier='Heat', load_factor_max=0.5)

    def test_ramp_up_limits_increase(self):
        """ramp_up_per_hour caps the rate increase between timesteps.

        CheapSrc (size=100, ramp_up=0.2 -> max +20/h), cost 1. Expensive cost 5.
        Demand=[10,50]. Cheap: t0=10, t1<=30 -> Expensive covers 20 at t1.
        cost = 10 + 30 + 20*5 = 140.

        Sensitivity: Without ramp_up, Cheap covers all -> cost=60.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 50])]),
                Port(
                    id='CheapSrc',
                    imports=[Flow(carrier='Heat', size=100, ramp_up_per_hour=0.2, effects_per_flow_hour={'cost': 1})],
                ),
                Port(id='ExpensiveSrc', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        assert_allclose(result.objective, 140.0, rtol=1e-5)
        cheap = result.flow_rate('CheapSrc(Heat)').values
        assert cheap[1] - cheap[0] <= 20.0 + 1e-5, f'Ramp-up violated: {cheap}'

    def test_ramp_down_limits_decrease(self):
        """ramp_down_per_hour caps the rate decrease between timesteps.

        Src (size=100, ramp_down=0.2 -> max -20/h), cost 1. Demand=[50,10].
        Src: t0=50, t1 >= 30 -> excess 20 absorbed by waste.
        cost = 50 + 30 = 80.

        Sensitivity: Without ramp_down, Src follows demand -> cost=60.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[50, 10])]),
                Port(
                    id='Src',
                    imports=[Flow(carrier='Heat', size=100, ramp_down_per_hour=0.2, effects_per_flow_hour={'cost': 1})],
                ),
                waste('Heat'),
            ],
        )
        assert_allclose(result.objective, 80.0, rtol=1e-5)
        src = result.flow_rate('Src(Heat)').values
        assert src[0] - src[1] <= 20.0 + 1e-5, f'Ramp-down violated: {src}'

    def test_ramp_requires_size(self):
        """Ramp limits on an unsized flow fail loudly at element level."""
        with pytest.raises(ValueError, match='ramp'):
            Flow(carrier='Heat', ramp_up_per_hour=0.2)

    def test_ramp_with_status_allows_startup(self):
        """The ramp does not bind across a startup: the unit may jump to its
        semi-continuous minimum even when the ramp is tighter.

        Unit (size=100, relative_rate_min=0.4, ramp_up=0.2, startup cost 5).
        Demand=[0,40,50]: off at t0, starts at t1 jumping to 40 (> ramp +20),
        then ramps 40->50 (delta 10 <= 20). cost = 90 + 5 = 95.
        """
        result = optimize(
            ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[0, 40, 50])]),
                Port(
                    id='Unit',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.4,
                            ramp_up_per_hour=0.2,
                            status=Status(effects_per_startup={'cost': 5}),
                            effects_per_flow_hour={'cost': 1},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 95.0, rtol=1e-5)

    def test_ramp_binds_between_running_steps(self):
        """The startup relaxation cannot be abused: between two running steps
        the ramp binds (startup/shutdown are pinned to actual transitions).

        Unit (size=100, relative_rate_min=0.4, ramp_up=0.2, startup FREE).
        Demand=[0,40,70]: starts at t1 -> 40; t1->t2 capped at +20 -> Unit=60,
        Backup covers 10 at cost 5. cost = 40 + 60 + 50 = 150.

        Sensitivity: If a spurious free startup at t2 could relax the ramp,
        Unit would cover 70 -> cost = 110.
        """
        result = optimize(
            ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[0, 40, 70])]),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
                Port(
                    id='Unit',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=100,
                            relative_rate_min=0.4,
                            ramp_up_per_hour=0.2,
                            status=Status(),
                            effects_per_flow_hour={'cost': 1},
                        )
                    ],
                ),
            ],
        )
        assert_allclose(result.objective, 150.0, rtol=1e-5)
        unit = result.flow_rate('Unit(Heat)').values
        assert unit[2] - unit[1] <= 20.0 + 1e-5, f'Ramp violated between running steps: {unit}'

    def test_ramp_with_component_status(self):
        """Ramps on storage flows relax via the component transition binaries.

        Battery (Storage.status, charge ramp_up=0.5): builds and solves with
        the cstatus-relaxed ramp constraint. Charge 5 cheap at t0, discharge
        at t1 -> cost 5.
        """
        result = optimize(
            ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 5])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 10]})]),
            ],
            storages=[
                Storage(
                    id='bat',
                    charging=Flow(carrier='Elec', size=10, ramp_up_per_hour=0.5),
                    discharging=Flow(carrier='Elec', size=10),
                    capacity=10,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        assert_allclose(result.objective, 5.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorage:
    def test_storage_shift_saves_money(self):
        """Price=[10,1,10]. Demand=[0,0,20]. Storage buys at t=1 @1EUR.
        cost=20.

        Sensitivity: Without storage, buy at t=2 @10EUR -> cost=200.
        """

        result = optimize(
            ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 0, 20])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [10, 1, 10]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=100),
                    discharging=Flow(carrier='Elec', size=100),
                    capacity=100,
                    prior_level=0.0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 20.0, rtol=1e-5)

    def test_storage_losses(self):
        """10% loss/h. Charge 100 at t=0, available after 1h = 90.
        Demand=[0,90]. Grid price=[1,1000]. cost=100.

        Sensitivity: Without losses, charge only 90 -> cost=90.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 90])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 1000]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=200,
                    prior_level=0.0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0.1,
                ),
            ],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self):
        """eta_c=0.9, eta_d=0.8. Need 72 out -> stored=72/0.8=90, charge=90/0.9=100.
        Demand=[0,72]. Price=[1,1000]. cost=100.

        Sensitivity: eta_c broken -> cost=90. eta_d broken -> cost=80.
        Both broken -> cost=72. Only both correct yields 100.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 72])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 1000]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=200,
                    prior_level=0.0,
                    cyclic=False,
                    eta_charge=0.9,
                    eta_discharge=0.8,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self):
        """Capacity=100, max SOC=0.5 -> 50 usable. Demand=[0,60].
        Price=[1,100]. Store 50 cheap, buy 10 expensive. cost=1050.

        Sensitivity: Without SOC bound, store 60 -> cost=60.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 60])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 100]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=100,
                    prior_level=0.0,
                    cyclic=False,
                    relative_level_max=0.5,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 1050.0, rtol=1e-5)

    def test_storage_cyclic_level(self):
        """Cyclic: final level = initial level. Price=[1,100]. Demand=[0,50].
        Must charge 50 at t=0 @1EUR and discharge at t=1. cost=50.

        Sensitivity: Without cyclic, start full (free energy) -> cost=0.
        """

        result = optimize(
            ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 50])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 100]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=100,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-5)

    def test_storage_relative_level_min(self):
        """Capacity=100, prior_level=50, min level=0.3 (->30 abs).
        Price=[1,100,1]. Demand=[0,80,0]. Charge 50 @t0 -> level=100.
        Discharge 70 @t1 -> level=30 (min). Grid covers 10 @100EUR.
        cost=50+1000=1050.

        Sensitivity: Without min level, discharge all -> no grid -> cost=50 less.
        """

        result = optimize(
            ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=[0, 80, 0])]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': [1, 100, 1]})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=100,
                    prior_level=50.0,
                    cyclic=False,
                    relative_level_min=0.3,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 1050.0, rtol=1e-5)
