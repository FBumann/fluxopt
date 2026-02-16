"""Mathematical correctness tests ported from flixopt/tests/test_math.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.
Sensitivity comments explain what would break if the feature were disabled.

API mapping (flixopt → fluxopt):
    fx.Source('name', outputs=[...])     → Port('name', imports=[...])
    fx.Sink('name', inputs=[...])        → Port('name', exports=[...])
    fx.Flow('label', bus=..., ...)       → Flow(bus=..., ...)
    effects_per_flow_hour=<scalar>       → effects_per_flow_hour={'costs': <scalar>}
    capacity_in_flow_hours=X             → capacity=X
    initial_charge_state='equals_final'  → initial_charge_state='cyclic'
    imbalance_penalty_per_flow_hour=0    → waste Port (absorbs excess at zero cost)
"""

from __future__ import annotations

from datetime import datetime

from numpy.testing import assert_allclose

from fluxopt import Bus, Converter, Effect, Flow, Port, Storage, solve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(n: int) -> list[datetime]:
    """Create n hourly timesteps starting 2020-01-01."""
    return [datetime(2020, 1, 1, h) for h in range(n)]


def _waste(bus: str) -> Port:
    """Free-disposal port: absorbs excess on *bus* at zero cost."""
    return Port('waste', exports=[Flow(bus=bus)])


# ---------------------------------------------------------------------------
# Bus balance & dispatch
# ---------------------------------------------------------------------------


class TestBusBalance:
    def test_merit_order_dispatch(self):
        """Src1: 1€/kWh, max 20. Src2: 2€/kWh, max 20. Demand=30/ts.
        Optimal: Src1=20, Src2=10.

        Sensitivity: Without merit order, cost could be 100 (Src2 first).
        Only correct bus balance with merit order yields cost=80.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[30, 30])]),
                Port('Src1', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1}, size=20)]),
                Port('Src2', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 2}, size=20)]),
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
        fuel = Flow(bus='Gas')
        thermal = Flow(bus='Heat')
        result = solve(
            _ts(3),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 20, 10])]),
                Port('GasSrc', imports=[Flow(bus='Gas', effects_per_flow_hour={'costs': 1})]),
            ],
            converters=[Converter.boiler('Boiler', 0.8, fuel, thermal)],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-5)

    def test_variable_efficiency(self):
        """Boiler eta=[0.5, 1.0], demand=[10,10]. fuel = 10/0.5 + 10/1.0 = 30.

        Sensitivity: Scalar mean (0.75) → 26.67. Only per-timestep yields 30.
        """
        fuel = Flow(bus='Gas')
        thermal = Flow(bus='Heat')
        result = solve(
            _ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port('GasSrc', imports=[Flow(bus='Gas', effects_per_flow_hour={'costs': 1})]),
            ],
            converters=[Converter.boiler('Boiler', [0.5, 1.0], fuel, thermal)],
        )
        assert_allclose(result.objective, 30.0, rtol=1e-5)

    def test_chp_dual_output(self):
        """CHP eta_th=0.5, eta_el=0.4. demand=50 heat/ts. Elec sold at -2€/kWh.
        fuel=50/0.5=100, elec=100*0.4=40. cost/ts = 100*1 - 40*2 = 20. total=40.

        Sensitivity: If eta_el broken, cost=200. If eta_th wrong (1.0), cost=-60.
        """
        fuel = Flow(bus='Gas')
        thermal = Flow(bus='Heat')
        electrical = Flow(bus='Elec')
        result = solve(
            _ts(2),
            buses=[Bus('Heat'), Bus('Elec'), Bus('Gas')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('HeatDemand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[50, 50])]),
                Port('ElecGrid', exports=[Flow(bus='Elec', effects_per_flow_hour={'costs': -2})]),
                Port('GasSrc', imports=[Flow(bus='Gas', effects_per_flow_hour={'costs': 1})]),
            ],
            converters=[Converter.chp('CHP', 0.4, 0.5, fuel, electrical, thermal)],
        )
        assert_allclose(result.objective, 40.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Effects & objective
# ---------------------------------------------------------------------------


class TestEffects:
    def test_effects_per_flow_hour(self):
        """costs=2€/kWh, CO2=0.5kg/kWh. Total flow=30.
        costs=60, CO2=15.

        Sensitivity: If only one effect applied, the other would be wrong.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2')],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 20])]),
                Port('HeatSrc', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 2, 'CO2': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 60.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 15.0, rtol=1e-5)

    def test_effect_maximum_total(self):
        """CO2 capped at 15. Dirty: 1€+1kgCO2/kWh. Clean: 10€+0kgCO2.
        Demand=20. Split: 15 Dirty + 5 Clean → cost=65.

        Sensitivity: Without CO2 cap, all Dirty → cost=20.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', maximum_total=15)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                Port('Clean', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 10, 'CO2': 0})]),
            ],
        )
        assert_allclose(result.objective, 65.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 15.0, rtol=1e-5)

    def test_effect_minimum_total(self):
        """CO2 floor at 25. Dirty: 1€+1kgCO2. Demand=20. Must overproduce.
        Dirty=25 (5 excess absorbed by waste port). cost=25.

        Sensitivity: Without minimum_total, Dirty=20 → cost=20.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', minimum_total=25)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                Port('Clean', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 0})]),
                _waste('Heat'),
            ],
        )
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 25.0, rtol=1e-5)
        assert_allclose(result.objective, 25.0, rtol=1e-5)

    def test_effect_maximum_per_hour(self):
        """CO2 max_per_hour=8. Dirty: 1€+1kgCO2. Clean: 5€+0kgCO2.
        Demand=[15,5]. Dirty capped at 8/ts → Dirty=[8,5], Clean=[7,0].
        cost = 13*1 + 7*5 = 48.

        Sensitivity: Without max_per_hour, all Dirty → cost=20.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', maximum_per_hour=8)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[15, 5])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                Port('Clean', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5, 'CO2': 0})]),
            ],
        )
        assert_allclose(result.objective, 48.0, rtol=1e-5)

    def test_effect_minimum_per_hour(self):
        """CO2 min_per_hour=10. Dirty: 1€+1kgCO2. Demand=[5,5].
        Must produce ≥10 CO2/ts → Dirty ≥10/ts. Excess absorbed by waste.
        cost=20.

        Sensitivity: Without min_per_hour, Dirty=5/ts → cost=10.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', minimum_per_hour=10)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[5, 5])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                _waste('Heat'),
            ],
        )
        assert_allclose(result.objective, 20.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 20.0, rtol=1e-5)

    def test_effect_maximum_temporal(self):
        """CO2 maximum_total=12 (≡ maximum_temporal when no periodic effects).
        Dirty: 1€+1kgCO2. Clean: 5€+0kgCO2. Demand=[10,10].
        Dirty=12, Clean=8 → cost=52.

        Sensitivity: Without cap, all Dirty → cost=20.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', maximum_total=12)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                Port('Clean', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5, 'CO2': 0})]),
            ],
        )
        assert_allclose(result.objective, 52.0, rtol=1e-5)
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 12.0, rtol=1e-5)

    def test_effect_minimum_temporal(self):
        """CO2 minimum_total=25 (≡ minimum_temporal). Dirty: 1€+1kgCO2.
        Demand=[10,10]. Dirty ≥25 → 5 excess. cost=25.

        Sensitivity: Without floor, Dirty=20 → cost=20.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True), Effect('CO2', minimum_total=25)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port('Dirty', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 1, 'CO2': 1})]),
                _waste('Heat'),
            ],
        )
        co2 = float(result.effect_totals.sel(effect='CO2').values)
        assert_allclose(co2, 25.0, rtol=1e-5)
        assert_allclose(result.objective, 25.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Flow constraints
# ---------------------------------------------------------------------------


class TestFlowConstraints:
    def test_relative_minimum(self):
        """Boiler (size=100, relative_minimum=0.4). When on, must produce ≥40.
        Demand=30 → excess absorbed by waste. fuel = 40/1.0 = 40. cost=80.

        Sensitivity: Without relative_minimum, boiler=30 → cost=60.
        """
        fuel = Flow(bus='Gas')
        thermal = Flow(bus='Heat', size=100, relative_minimum=0.4)
        result = solve(
            _ts(2),
            buses=[Bus('Heat'), Bus('Gas')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[30, 30])]),
                Port('GasSrc', imports=[Flow(bus='Gas', effects_per_flow_hour={'costs': 1})]),
                _waste('Heat'),
            ],
            converters=[Converter.boiler('Boiler', 1.0, fuel, thermal)],
        )
        assert_allclose(result.objective, 80.0, rtol=1e-5)
        flow = result.flow_rate('Boiler(Heat)').values
        assert all(f >= 40.0 - 1e-5 for f in flow), f'Flow below relative_minimum: {flow}'

    def test_relative_maximum(self):
        """Source (size=100, relative_maximum=0.5). Max output=50.
        Demand=60 → CheapSrc=50, ExpensiveSrc=10. cost=2*(50*1+10*5)=200.

        Sensitivity: Without relative_maximum, all from CheapSrc → cost=120.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Heat')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=[60, 60])]),
                Port(
                    'CheapSrc',
                    imports=[Flow(bus='Heat', size=100, relative_maximum=0.5, effects_per_flow_hour={'costs': 1})],
                ),
                Port('ExpensiveSrc', imports=[Flow(bus='Heat', effects_per_flow_hour={'costs': 5})]),
            ],
        )
        assert_allclose(result.objective, 200.0, rtol=1e-5)
        flow = result.flow_rate('CheapSrc(Heat)').values
        assert all(f <= 50.0 + 1e-5 for f in flow), f'Flow above relative_maximum: {flow}'


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestStorage:
    def test_storage_shift_saves_money(self):
        """Price=[10,1,10]. Demand=[0,0,20]. Storage buys at t=1 @1€.
        cost=20.

        Sensitivity: Without storage, buy at t=2 @10€ → cost=200.
        """
        result = solve(
            _ts(3),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 0, 20])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [10, 1, 10]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=100),
                    discharging=Flow(bus='Elec', size=100),
                    capacity=100,
                    initial_charge_state=0.0,
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

        Sensitivity: Without losses, charge only 90 → cost=90.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 90])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 1000]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=200,
                    initial_charge_state=0.0,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0.1,
                ),
            ],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self):
        """eta_c=0.9, eta_d=0.8. Need 72 out → stored=72/0.8=90, charge=90/0.9=100.
        Demand=[0,72]. Price=[1,1000]. cost=100.

        Sensitivity: eta_c broken → cost=90. eta_d broken → cost=80.
        Both broken → cost=72. Only both correct yields 100.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 72])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 1000]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=200,
                    initial_charge_state=0.0,
                    eta_charge=0.9,
                    eta_discharge=0.8,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self):
        """Capacity=100, max SOC=0.5 → 50 usable. Demand=[0,60].
        Price=[1,100]. Store 50 cheap, buy 10 expensive. cost=1050.

        Sensitivity: Without SOC bound, store 60 → cost=60.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 60])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 100]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=100,
                    initial_charge_state=0.0,
                    relative_maximum_charge_state=0.5,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 1050.0, rtol=1e-5)

    def test_storage_cyclic_charge_state(self):
        """Cyclic: final SOC = initial SOC. Price=[1,100]. Demand=[0,50].
        Must charge 50 at t=0 @1€ and discharge at t=1. cost=50.

        Sensitivity: Without cyclic, start full (free energy) → cost=0.
        """
        result = solve(
            _ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 50])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 100]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=100,
                    initial_charge_state='cyclic',
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-5)

    def test_storage_relative_minimum_charge_state(self):
        """Capacity=100, initial=50, min SOC=0.3 (→30 abs).
        Price=[1,100,1]. Demand=[0,80,0]. Charge 50 @t0 → SOC=100.
        Discharge 70 @t1 → SOC=30 (min). Grid covers 10 @100€.
        cost=50+1000=1050.

        Sensitivity: Without min SOC, discharge all → no grid → cost=50 less.
        """
        result = solve(
            _ts(3),
            buses=[Bus('Elec')],
            effects=[Effect('costs', is_objective=True)],
            ports=[
                Port('Demand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=[0, 80, 0])]),
                Port('Grid', imports=[Flow(bus='Elec', effects_per_flow_hour={'costs': [1, 100, 1]})]),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=100,
                    initial_charge_state=0.5,
                    relative_minimum_charge_state=0.3,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.objective, 1050.0, rtol=1e-5)
