from __future__ import annotations

import pytest

from fluxopt import Bus, Effect, Flow, LinearConverter, Sink, Source, solve


class TestBoiler:
    def test_gas_equals_heat_over_efficiency(self, timesteps_3):
        """Boiler: gas_rate = heat_rate / eta."""
        eta = 0.9
        heat_demand = [50.0, 80.0, 60.0]

        demand_flow = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        gas_flow = Flow('grid(gas)', bus='gas', size=200, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow('boiler(gas)', bus='gas', size=200)
        heat = Flow('boiler(heat)', bus='heat', size=100)

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_flow]),
                Sink('demand', inputs=[demand_flow]),
                LinearConverter.boiler('boiler', eta, fuel, heat),
            ],
        )

        gas_rates = result.flow_rate('boiler(gas)')['value'].to_list()
        for gas, h in zip(gas_rates, heat_demand, strict=False):
            assert gas == pytest.approx(h / eta, abs=1e-6)

    def test_cost_with_boiler(self, timesteps_3):
        """Total cost = sum(gas_rate * cost * dt)."""
        eta = 0.9

        demand_flow = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        gas_flow = Flow('grid(gas)', bus='gas', size=200, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow('boiler(gas)', bus='gas', size=200)
        heat = Flow('boiler(heat)', bus='heat', size=100)

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_flow]),
                Sink('demand', inputs=[demand_flow]),
                LinearConverter.boiler('boiler', eta, fuel, heat),
            ],
        )

        expected = (50 / eta + 80 / eta + 60 / eta) * 0.04
        assert result.objective_value == pytest.approx(expected, abs=1e-6)


class TestCHP:
    def test_chp_conversion(self, timesteps_3):
        """CHP: fuel * eta_el = elec, fuel * eta_th = heat."""
        eta_el, eta_th = 0.3, 0.5

        fuel_flow = Flow('chp(gas)', bus='gas', size=200)
        elec_flow = Flow('chp(elec)', bus='elec', size=100)
        heat_flow = Flow('chp(heat)', bus='heat', size=100)

        gas_source = Flow('grid(gas)', bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})
        elec_demand = Flow('demand(elec)', bus='elec', size=100, fixed_relative_profile=[0.3, 0.3, 0.3])
        heat_demand = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('gas'), Bus('elec'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_source]),
                Sink('elec_demand', inputs=[elec_demand]),
                Sink('heat_demand', inputs=[heat_demand]),
                LinearConverter.chp('chp', eta_el, eta_th, fuel_flow, elec_flow, heat_flow),
            ],
        )

        gas_rates = result.flow_rate('chp(gas)')['value'].to_list()
        elec_rates = result.flow_rate('chp(elec)')['value'].to_list()
        heat_rates = result.flow_rate('chp(heat)')['value'].to_list()

        for gas, elec, heat in zip(gas_rates, elec_rates, heat_rates, strict=False):
            assert elec == pytest.approx(gas * eta_el, abs=1e-6)
            assert heat == pytest.approx(gas * eta_th, abs=1e-6)
