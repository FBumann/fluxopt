from __future__ import annotations

import pytest
import xarray as xr

from fluxopt import Bus, Effect, Flow, Port, Sizing, Status, Storage, optimize
from fluxopt.components import Converter


class TestSumToTotal:
    def test_single_source(self, timesteps_3):
        """Single source gets 100% of effects."""
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_two_sources_sum_to_total(self, timesteps_3):
        """Two sources' contributions sum to the solver total."""
        cheap = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02})
        expensive = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port('cheap_src', imports=[cheap]),
                Port('exp_src', imports=[expensive]),
                Port('demand', exports=[sink]),
            ],
        )

        contrib = result.effect_contributions()
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_per_timestep_sum_to_effect_per_timestep(self, timesteps_3):
        """Operational contributions summed over flows match effect_per_timestep."""
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        op_sum = contrib['operational'].sel(effect='cost').sum('flow')
        ept = result.effects_per_timestep.sel(effect='cost')
        xr.testing.assert_allclose(op_sum, ept)


class TestProportionalSplit:
    def test_cheap_gets_all_demand(self, timesteps_3):
        """With fixed demand, cheaper source serves everything → gets all cost."""
        cheap = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02})
        expensive = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port('cheap_src', imports=[cheap]),
                Port('exp_src', imports=[expensive]),
                Port('demand', exports=[sink]),
            ],
        )

        contrib = result.effect_contributions()
        cheap_cost = float(contrib['total'].sel(flow='cheap_src(elec)', effect='cost').values)
        exp_cost = float(contrib['total'].sel(flow='exp_src(elec)', effect='cost').values)
        demand_total = 50 + 80 + 60
        assert cheap_cost == pytest.approx(demand_total * 0.02, abs=1e-6)
        assert exp_cost == pytest.approx(0.0, abs=1e-6)


class TestCrossEffects:
    def test_carbon_tax_attributed_to_emitter(self, timesteps_3):
        """Carbon tax cost is attributed to the CO2-emitting flow."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        total_energy = sum(demand)
        direct_cost = total_energy * 0.04
        co2_total = total_energy * 0.5
        co2_cost = co2_total * 50
        expected_cost = direct_cost + co2_cost

        grid_cost = float(contrib['total'].sel(flow='grid(elec)', effect='cost').values)
        assert grid_cost == pytest.approx(expected_cost, abs=1e-6)

        # Sum matches solver
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_cross_effect_two_emitters(self, timesteps_3):
        """Carbon tax is split proportionally between two emitting sources."""
        dirty = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
        clean = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        co2_limit = 100.0
        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', maximum_total=co2_limit),
            ],
            ports=[
                Port('dirty_src', imports=[dirty]),
                Port('clean_src', imports=[clean]),
                Port('demand', exports=[sink]),
            ],
        )

        contrib = result.effect_contributions()
        # Clean source has zero CO2 → zero carbon tax contribution
        clean_co2 = float(contrib['operational'].sel(flow='clean_src(elec)', effect='co2').sum('time').values)
        assert clean_co2 == pytest.approx(0.0, abs=1e-6)

        # Totals still match
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_transitive_cross_effects(self, timesteps_3):
        """PE → CO2 → cost chain: contributions propagate transitively."""
        demand = [50.0, 80.0, 60.0]
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'pe': 2.0})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg', contribution_from={'pe': 0.3}),
                Effect('pe', unit='kWh'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        total_energy = sum(demand)
        pe_total = total_energy * 2.0
        co2_total = pe_total * 0.3
        cost_total = co2_total * 50

        grid_cost = float(contrib['total'].sel(flow='grid(elec)', effect='cost').values)
        assert grid_cost == pytest.approx(cost_total, abs=1e-6)


class TestSizing:
    def test_sizing_investment_on_correct_flow(self, timesteps_3):
        """Investment costs appear on the flow that has sizing."""
        source = Flow(
            bus='elec',
            size=Sizing(min_size=50, max_size=200, mandatory=True, effects_per_size={'cost': 100}),
            effects_per_flow_hour={'cost': 0.04},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        grid_inv = float(contrib['investment'].sel(flow='grid(elec)', effect='cost').values)
        demand_inv = float(contrib['investment'].sel(flow='demand(elec)', effect='cost').values)
        assert grid_inv > 0
        assert demand_inv == pytest.approx(0.0, abs=1e-6)

        # Sum matches solver
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_sizing_cross_effect_investment(self, timesteps_3):
        """Sizing CO2 priced into cost via contribution_from."""
        source = Flow(
            bus='elec',
            size=Sizing(min_size=50, max_size=200, mandatory=True, effects_per_size={'co2': 10}),
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

        # Grid flow gets the investment cost (including cross-effect from CO2)
        grid_inv_cost = float(contrib['investment'].sel(flow='grid(elec)', effect='cost').values)
        invest_size = float(result.sizes.sel(flow='grid(elec)').values)
        invest_co2 = invest_size * 10
        expected_inv_cost = invest_co2 * 50
        assert grid_inv_cost == pytest.approx(expected_inv_cost, abs=1e-6)


class TestStatus:
    def test_running_costs_on_correct_flow(self, timesteps_3):
        """Running costs appear on the flow that has status."""
        source = Flow(
            bus='elec',
            size=100,
            relative_minimum=0.3,
            effects_per_flow_hour={'cost': 0.04},
            status=Status(effects_per_running_hour={'cost': 5.0}),
        )
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

        # Grid has running costs, demand does not
        grid_cost = float(contrib['total'].sel(flow='grid(elec)', effect='cost').values)
        demand_cost = float(contrib['total'].sel(flow='demand(elec)', effect='cost').values)
        assert grid_cost > 0
        assert demand_cost == pytest.approx(0.0, abs=1e-6)


class TestConverter:
    def test_converter_contributions(self, timesteps_3):
        """Converter flows are attributed to their respective flows."""
        fuel = Flow(bus='gas', size=200, effects_per_flow_hour={'cost': 0.03})
        heat = Flow(bus='heat', size=200)
        gas_supply = Flow(bus='gas', size=200)
        heat_sink = Flow(bus='heat', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('gas_grid', imports=[gas_supply]), Port('demand', exports=[heat_sink])],
            converters=[Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat)],
        )

        contrib = result.effect_contributions()
        # Boiler fuel flow has cost
        boiler_fuel_cost = float(contrib['total'].sel(flow='boiler(gas)', effect='cost').values)
        assert boiler_fuel_cost > 0
        # Boiler heat flow has no direct cost
        boiler_heat_cost = float(contrib['total'].sel(flow='boiler(heat)', effect='cost').values)
        assert boiler_heat_cost == pytest.approx(0.0, abs=1e-6)

        # Total matches
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('flow').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)


class TestStorage:
    def test_storage_sizing_costs(self, timesteps_4):
        """Storage sizing investment costs appear in storage_investment."""
        charge = Flow(bus='elec')
        discharge = Flow(bus='elec')
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02, 0.08]})
        sink = Flow(bus='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_4,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
            storages=[
                Storage(
                    'battery',
                    charging=charge,
                    discharging=discharge,
                    capacity=Sizing(min_size=10, max_size=100, mandatory=True, effects_per_size={'cost': 50}),
                )
            ],
        )

        contrib = result.effect_contributions()
        assert 'storage_investment' in contrib
        bat_inv = float(contrib['storage_investment'].sel(storage='battery', effect='cost').values)
        assert bat_inv > 0

        # Full total: flow totals + storage investment = solver total
        flow_total = float(contrib['total'].sel(effect='cost').sum('flow').values)
        stor_inv_total = float(contrib['storage_investment'].sel(effect='cost').sum('storage').values)
        solver_total = float(result.effect_totals.sel(effect='cost').values)
        assert flow_total + stor_inv_total == pytest.approx(solver_total, abs=1e-6)


class TestEdgeCases:
    def test_no_cost_flows(self, timesteps_3):
        """With no effects_per_flow_hour, contributions are zero."""
        source = Flow(bus='elec', size=100)
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        assert float(contrib['total'].sum().values) == pytest.approx(0.0, abs=1e-6)

    def test_data_required(self, timesteps_3):
        """effect_contributions raises when data is missing."""
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        result.data = None
        with pytest.raises(ValueError, match='ModelData is required'):
            result.effect_contributions()

    def test_multiple_effects_sum_to_total(self, timesteps_3):
        """Multiple effects tracked simultaneously all sum correctly."""
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True), Effect('co2', unit='kg')],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        contrib = result.effect_contributions()
        for eff in ['cost', 'co2']:
            total_from_contrib = float(contrib['total'].sel(effect=eff).sum('flow').values)
            total_from_solver = float(result.effect_totals.sel(effect=eff).values)
            assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_caching(self, timesteps_3):
        """Repeated calls return the same cached object."""
        source = Flow(bus='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )

        first = result.effect_contributions()
        second = result.effect_contributions()
        assert first is second
