from __future__ import annotations

import pytest
import xarray as xr
from conftest import ts

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Status, Storage, optimize
from fluxopt.components import Converter


class TestSumToTotal:
    def test_single_source(self):
        """Single source gets 100% of effects."""

        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_two_sources_sum_to_total(self):
        """Two sources' contributions sum to the solver total."""

        cheap = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.02})
        expensive = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.10})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='cheap_src', imports=[cheap]),
                Port(id='exp_src', imports=[expensive]),
                Port(id='demand', exports=[sink]),
            ],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_per_timestep_temporal_matches_hand_computed(self):
        """Reconstructed per-timestep effect values match rate x coefficient.

        Demand profile [0.5, 0.8, 0.6] x size 100 -> rates [50, 80, 60];
        cost coefficient 0.04/flow-hour -> per-timestep cost [2.0, 3.2, 2.4].
        """
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        ept = result.effects_temporal.sel(effect='cost')
        assert list(ept.values.round(6)) == [2.0, 3.2, 2.4]
        contrib = result.stats.effect_contributions
        temporal_sum = contrib['temporal'].sel(effect='cost').sum('contributor')
        xr.testing.assert_allclose(temporal_sum, ept)


class TestProportionalSplit:
    def test_cheap_gets_all_demand(self):
        """With fixed demand, cheaper source serves everything -> gets all cost."""

        cheap = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.02})
        expensive = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.10})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='cheap_src', imports=[cheap]),
                Port(id='exp_src', imports=[expensive]),
                Port(id='demand', exports=[sink]),
            ],
        )

        contrib = result.stats.effect_contributions
        cheap_cost = float(contrib['total'].sel(contributor='cheap_src(elec)', effect='cost').values)
        exp_cost = float(contrib['total'].sel(contributor='exp_src(elec)', effect='cost').values)
        demand_total = 50 + 80 + 60
        assert cheap_cost == pytest.approx(demand_total * 0.02, abs=1e-6)
        assert exp_cost == pytest.approx(0.0, abs=1e-6)


class TestCrossEffects:
    def test_carbon_tax_attributed_to_emitter(self):
        """Carbon tax cost is attributed to the CO2-emitting flow."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_energy = sum(demand)
        direct_cost = total_energy * 0.04
        co2_total = total_energy * 0.5
        co2_cost = co2_total * 50
        expected_cost = direct_cost + co2_cost

        grid_cost = float(contrib['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_cost == pytest.approx(expected_cost, abs=1e-6)

        # Sum matches solver
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_cross_effect_two_emitters(self):
        """Carbon tax is split proportionally between two emitting sources."""

        dirty = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
        clean = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        co2_limit = 100.0
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', total_max=co2_limit),
            ],
            objective='cost',
            ports=[
                Port(id='dirty_src', imports=[dirty]),
                Port(id='clean_src', imports=[clean]),
                Port(id='demand', exports=[sink]),
            ],
        )

        contrib = result.stats.effect_contributions
        # Clean source has zero CO2 -> zero carbon tax contribution
        clean_co2 = float(contrib['temporal'].sel(contributor='clean_src(elec)', effect='co2').sum('time').values)
        assert clean_co2 == pytest.approx(0.0, abs=1e-6)

        # Totals still match
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_transitive_cross_effects(self):
        """PE -> CO2 -> cost chain: contributions propagate transitively."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'pe': 2.0})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg', contribution_from={'pe': 0.3}),
                Effect(id='pe', unit='kWh'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_energy = sum(demand)
        pe_total = total_energy * 2.0
        co2_total = pe_total * 0.3
        cost_total = co2_total * 50

        grid_cost = float(contrib['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_cost == pytest.approx(cost_total, abs=1e-6)


class TestSizing:
    def test_sizing_investment_on_correct_flow(self):
        """Investment costs appear on the flow that has sizing."""

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=50, size_max=200, mandatory=True, effects_per_size={'cost': 100}),
            effects_per_flow_hour={'cost': 0.04},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        grid_inv = float(contrib['lump'].sel(contributor='grid(elec)', effect='cost').values)
        demand_inv = float(contrib['lump'].sel(contributor='demand(elec)', effect='cost').values)
        # size=50 (min to meet demand) * effects_per_size=100
        expected_inv = 50 * 100
        assert grid_inv == pytest.approx(expected_inv, abs=1e-6)
        assert demand_inv == pytest.approx(0.0, abs=1e-6)

        # Sum matches solver
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

    def test_optional_sizing_fixed_costs(self):
        """Optional sizing with fixed costs uses binary indicator in contributions."""

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=0, size_max=200, mandatory=False, effects_fixed={'cost': 1000}),
            effects_per_flow_hour={'cost': 0.04},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

        # Grid has investment cost from fixed costs
        grid_periodic = float(contrib['lump'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_periodic == pytest.approx(1000, abs=1e-6)

    def test_sizing_cross_effect_investment(self):
        """Sizing CO2 priced into cost via contribution_from."""

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=50, size_max=200, mandatory=True, effects_per_size={'co2': 10}),
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)

        # Grid flow gets the investment cost (including cross-effect from CO2)
        grid_inv_cost = float(contrib['lump'].sel(contributor='grid(elec)', effect='cost').values)
        invest_size = float(result.sizes.sel(flow='grid(elec)').values)
        invest_co2 = invest_size * 10
        expected_inv_cost = invest_co2 * 50
        assert grid_inv_cost == pytest.approx(expected_inv_cost, abs=1e-6)


class TestStatus:
    def test_running_costs_on_correct_flow(self):
        """Running costs appear on the flow that has status."""

        source = Flow(
            carrier='elec',
            size=100,
            relative_rate_min=0.3,
            effects_per_flow_hour={'cost': 0.04},
            status=Status(effects_per_running_hour={'cost': 5.0}),
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        total_from_solver = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

        # Grid has running costs, demand does not
        grid_cost = float(contrib['total'].sel(contributor='grid(elec)', effect='cost').values)
        demand_cost = float(contrib['total'].sel(contributor='demand(elec)', effect='cost').values)
        # operational: (50+80+60)*0.04=7.6, running: 5.0*3h=15.0
        expected_grid_cost = (50 + 80 + 60) * 0.04 + 5.0 * 3
        assert grid_cost == pytest.approx(expected_grid_cost, abs=1e-6)
        assert demand_cost == pytest.approx(0.0, abs=1e-6)


class TestConverter:
    def test_converter_contributions(self):
        """Converter flows are attributed to their respective flows."""

        fuel = Flow(carrier='gas', size=200, effects_per_flow_hour={'cost': 0.03})
        heat_flow = Flow(carrier='heat', size=200)
        gas_supply = Flow(carrier='gas', size=200)
        heat_sink = Flow(carrier='heat', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='gas_grid', imports=[gas_supply]), Port(id='demand', exports=[heat_sink])],
            converters=[Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat_flow)],
        )

        contrib = result.stats.effect_contributions
        # Boiler fuel flow has cost
        boiler_fuel_cost = float(contrib['total'].sel(contributor='boiler(gas)', effect='cost').values)
        assert boiler_fuel_cost == pytest.approx(sum([50, 80, 60]) / 0.9 * 0.03, abs=1e-5)
        # Boiler heat flow has no direct cost
        boiler_heat_cost = float(contrib['total'].sel(contributor='boiler(heat)', effect='cost').values)
        assert boiler_heat_cost == pytest.approx(0.0, abs=1e-6)

        # Total matches
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        assert total_from_contrib == pytest.approx(float(result.effect_totals.sel(effect='cost').values), abs=1e-6)


class TestStorage:
    def test_storage_sizing_costs(self):
        """Storage sizing investment costs appear on contributor dim."""

        charge = Flow(carrier='elec')
        discharge = Flow(carrier='elec')
        source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02, 0.08]})
        sink = Flow(carrier='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
            storages=[
                Storage(
                    id='battery',
                    charging=charge,
                    discharging=discharge,
                    capacity=Sizing(size_min=10, size_max=100, mandatory=True, effects_per_size={'cost': 50}),
                )
            ],
        )

        contrib = result.stats.effect_contributions
        # Storage appears as a contributor in periodic
        bat_inv = float(contrib['lump'].sel(contributor='battery', effect='cost').values)
        bat_capacity = float(result.storage_capacities.sel(storage='battery').values)
        assert bat_inv == pytest.approx(bat_capacity * 50, abs=1e-5)

        # Total (summed over all contributors) matches solver
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        solver_total = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(solver_total, abs=1e-6)

    def test_storage_sizing_cross_effect(self):
        """Storage sizing CO2 priced into cost via contribution_from."""

        charge = Flow(carrier='elec')
        discharge = Flow(carrier='elec')
        source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04, 'co2': 0.1})
        sink = Flow(carrier='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
            storages=[
                Storage(
                    id='battery',
                    charging=charge,
                    discharging=discharge,
                    capacity=Sizing(size_min=10, size_max=100, mandatory=True, effects_per_size={'co2': 5}),
                )
            ],
        )

        contrib = result.stats.effect_contributions
        total_from_contrib = float(contrib['total'].sel(effect='cost').sum('contributor').values)
        solver_total = float(result.effect_totals.sel(effect='cost').values)
        assert total_from_contrib == pytest.approx(solver_total, abs=1e-6)

        # Storage has CO2 investment cost that gets priced into cost via cross-effect
        bat_periodic_cost = float(contrib['lump'].sel(contributor='battery', effect='cost').values)
        bat_capacity = float(result.storage_capacities.sel(storage='battery').values)
        expected_co2_inv = bat_capacity * 5
        expected_cost_inv = expected_co2_inv * 50
        assert bat_periodic_cost == pytest.approx(expected_cost_inv, abs=1e-6)


class TestEdgeCases:
    def test_no_cost_flows(self):
        """With no effects_per_flow_hour, contributions are zero."""

        source = Flow(carrier='elec', size=100)
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        assert float(contrib['total'].sum().values) == pytest.approx(0.0, abs=1e-6)

    def test_multiple_effects_sum_to_total(self):
        """Multiple effects tracked simultaneously all sum correctly."""

        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost'), Effect(id='co2', unit='kg')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        contrib = result.stats.effect_contributions
        for eff in ['cost', 'co2']:
            total_from_contrib = float(contrib['total'].sel(effect=eff).sum('contributor').values)
            total_from_solver = float(result.effect_totals.sel(effect=eff).values)
            assert total_from_contrib == pytest.approx(total_from_solver, abs=1e-6)

    def test_caching(self):
        """Stats accessor and its properties are cached."""

        source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        assert result.stats is result.stats
        assert result.stats.effect_contributions is result.stats.effect_contributions
        assert result.stats.effect_contributions_direct is result.stats.effect_contributions_direct


class TestDirectContributions:
    """Direct contributions skip Leontief — each contributor only shows what it
    directly emits, independent of ``contribution_from`` chains."""

    def test_direct_equals_with_cross_when_no_contribution_from(self):
        """Without contribution_from, direct and with-cross views match."""

        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost'), Effect(id='co2', unit='kg')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        with_cross = result.stats.effect_contributions
        direct = result.stats.effect_contributions_direct
        xr.testing.assert_allclose(with_cross['temporal'], direct['temporal'])
        xr.testing.assert_allclose(with_cross['lump'], direct['lump'])
        xr.testing.assert_allclose(with_cross['total'], direct['total'])

    def test_direct_and_with_cross_differ_when_contribution_from_present(self):
        """Sanity invariant: when contribution_from is set, the two views must
        disagree on at least one (contributor, effect) — otherwise the direct
        accessor is silently returning the with-cross result (or vice versa)."""

        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        with_cross = result.stats.effect_contributions
        direct = result.stats.effect_contributions_direct
        assert not direct['total'].equals(with_cross['total'])
        assert not direct['temporal'].equals(with_cross['temporal'])

    def test_direct_strips_carbon_tax_propagation(self):
        """Direct cost = direct flow cost only, ignoring CO₂→cost cross-effect."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        direct = result.stats.effect_contributions_direct
        with_cross = result.stats.effect_contributions

        total_energy = sum(demand)
        # Direct view: grid pays only its own per-flow-hour cost (no CO2 markup)
        grid_direct_cost = float(direct['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_direct_cost == pytest.approx(total_energy * 0.04, abs=1e-6)

        # With-cross view: grid pays direct cost + CO2 priced in at 50/kg
        grid_xc_cost = float(with_cross['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_xc_cost == pytest.approx(total_energy * 0.04 + total_energy * 0.5 * 50, abs=1e-6)

        # CO2 attribution itself doesn't change between the two views
        # (CO2 is a leaf with no contribution_from — Leontief is identity for it)
        xr.testing.assert_allclose(direct['total'].sel(effect='co2'), with_cross['total'].sel(effect='co2'))

    def test_direct_sum_equals_raw_emission_total(self):
        """Direct co2 contributions sum to the raw integrated emissions
        (no cross-effects mean physical totals are unchanged)."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        direct = result.stats.effect_contributions_direct
        co2_direct = float(direct['total'].sel(effect='co2').sum('contributor').values)
        assert co2_direct == pytest.approx(sum(demand) * 0.5, abs=1e-6)

    def test_direct_drops_transitive_propagation(self):
        """PE → CO₂ → cost: direct view shows zero direct cost from grid,
        full chain only appears in the with-cross view."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'pe': 2.0})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg', contribution_from={'pe': 0.3}),
                Effect(id='pe', unit='kWh'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        direct = result.stats.effect_contributions_direct
        with_cross = result.stats.effect_contributions

        total_energy = sum(demand)
        # Direct: grid only directly emits PE — no direct co2, no direct cost
        grid_direct_pe = float(direct['total'].sel(contributor='grid(elec)', effect='pe').values)
        grid_direct_co2 = float(direct['total'].sel(contributor='grid(elec)', effect='co2').values)
        grid_direct_cost = float(direct['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_direct_pe == pytest.approx(total_energy * 2.0, abs=1e-6)
        assert grid_direct_co2 == pytest.approx(0.0, abs=1e-6)
        assert grid_direct_cost == pytest.approx(0.0, abs=1e-6)

        # With-cross: chain propagates pe→co2→cost
        grid_xc_co2 = float(with_cross['total'].sel(contributor='grid(elec)', effect='co2').values)
        grid_xc_cost = float(with_cross['total'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_xc_co2 == pytest.approx(total_energy * 2.0 * 0.3, abs=1e-6)
        assert grid_xc_cost == pytest.approx(total_energy * 2.0 * 0.3 * 50, abs=1e-6)

    def test_direct_does_not_validate_against_solver_total(self):
        """Direct totals can differ from solver totals (which include cross-effects).
        Sanity: direct cost total is less than solver cost total when CO₂→cost is set."""

        demand = [50.0, 80.0, 60.0]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        direct = result.stats.effect_contributions_direct
        direct_cost_total = float(direct['total'].sel(effect='cost').sum('contributor').values)
        solver_cost_total = float(result.effect_totals.sel(effect='cost').values)
        # Strict inequality: there's a non-zero CO₂→cost contribution
        assert direct_cost_total < solver_cost_total
        assert direct_cost_total == pytest.approx(sum(demand) * 0.04, abs=1e-6)

    def test_direct_lump_strips_sizing_cross_effect(self):
        """Sizing CO₂ stays as CO₂ in direct view, not priced into cost."""

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=50, size_max=200, mandatory=True, effects_per_size={'co2': 10}),
            effects_per_flow_hour={'cost': 0.04},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        direct = result.stats.effect_contributions_direct
        invest_size = float(result.sizes.sel(flow='grid(elec)').values)

        grid_direct_co2_lump = float(direct['lump'].sel(contributor='grid(elec)', effect='co2').values)
        grid_direct_cost_lump = float(direct['lump'].sel(contributor='grid(elec)', effect='cost').values)
        assert grid_direct_co2_lump == pytest.approx(invest_size * 10, abs=1e-6)
        # No direct sizing cost — only CO₂ → cost via cross-effect, which direct strips
        assert grid_direct_cost_lump == pytest.approx(0.0, abs=1e-6)


class TestValidateAgainstSolver:
    """The validation helper raises when per-contributor totals don't sum to solver totals."""

    def test_raises_on_mismatch(self):
        from fluxopt.contributions import validate_against_solver

        total = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=['contributor', 'effect'],
            coords={'contributor': ['a', 'b'], 'effect': ['cost', 'co2']},
        )
        solution = xr.Dataset(
            {
                'effect--total': xr.DataArray([100.0, 200.0], dims=['effect'], coords={'effect': ['cost', 'co2']}),
            }
        )
        with pytest.raises(ValueError, match='Effect contributions do not sum to solver totals'):
            validate_against_solver(total, solution)

    def test_passes_on_exact_match(self):
        from fluxopt.contributions import validate_against_solver

        total = xr.DataArray(
            [[1.0, 2.0], [3.0, 4.0]],
            dims=['contributor', 'effect'],
            coords={'contributor': ['a', 'b'], 'effect': ['cost', 'co2']},
        )
        solution = xr.Dataset(
            {
                'effect--total': xr.DataArray([4.0, 6.0], dims=['effect'], coords={'effect': ['cost', 'co2']}),
            }
        )
        validate_against_solver(total, solution)  # no exception


class TestBreakdownIsReadOffTheModel:
    """The breakdown comes from the program's own expressions, not a recompute."""

    def _result(self):
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        return optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost', contribution_from={'co2': 50}), Effect(id='co2', unit='kg')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

    def test_the_solve_records_every_named_quantity(self):
        """A named expression is evaluated at a solution, so the solve is where it can be had."""
        result = self._result()
        assert 'contribution_flow_hour' in result.expressions
        assert 'effect_lump' in result.expressions
        # and it is reachable by name, not only through the breakdown
        assert result.expression('effect_temporal').sum() != 0

    def test_a_result_without_them_says_so_rather_than_guessing(self):
        """Re-deriving from the solution alone would answer with today's logic."""
        import dataclasses

        import xarray as xr

        stripped = dataclasses.replace(self._result(), expressions=xr.Dataset())
        with pytest.raises(ValueError, match='cannot re-derive'):
            _ = stripped.stats.effect_contributions
        with pytest.raises(KeyError, match='effect_lump'):
            stripped.expression('effect_lump')

    def test_the_two_views_differ_only_by_the_fold(self):
        """`direct` is `(I - C) . charged`, so it strips exactly the priced-in share."""
        result = self._result()
        charged = result.stats.effect_contributions
        direct = result.stats.effect_contributions_direct
        # cost receives co2 priced at 50, so direct must be the smaller of the two
        assert float(direct['total'].sel(effect='cost').sum()) < float(charged['total'].sel(effect='cost').sum())
        # co2 emits into nothing, so both views agree on it
        xr.testing.assert_allclose(direct['total'].sel(effect='co2'), charged['total'].sel(effect='co2'))
