from __future__ import annotations

import polars as pl
import pytest
from conftest import ts

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    ModelData,
    Port,
    Storage,
    optimize,
)
from fluxopt.math import solve


class TestEndToEnd:
    def test_full_system(self):
        """Full system: gas source -> boiler -> heat bus <- demand, with cost tracking."""

        eta = 0.9
        heat_demand = [40.0, 70.0, 50.0, 60.0]

        demand_flow = Flow(carrier='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
        gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow(carrier='gas', size=300)
        heat_flow = Flow(carrier='heat', size=200)

        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='grid', imports=[gas_source]),
                Port(id='demand', exports=[demand_flow]),
            ],
            converters=[Converter.boiler('boiler', eta, fuel, heat_flow)],
        )

        # Verify gas = heat / eta
        gas_rates = result.flow_rate('boiler(gas)').values
        for gas_rate, hd in zip(gas_rates, heat_demand, strict=False):
            assert gas_rate == pytest.approx(hd / eta, abs=1e-6)

        # Verify cost
        total_gas = sum(h / eta for h in heat_demand)
        expected_cost = total_gas * 0.04
        assert result.objective == pytest.approx(expected_cost, abs=1e-6)

    def test_boiler_plus_storage(self):
        """Boiler + thermal storage: store heat in cheap hours."""

        eta = 0.9
        gas_prices = [0.02, 0.08, 0.02, 0.08]

        demand_flow = Flow(carrier='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])
        gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': gas_prices})
        fuel = Flow(carrier='gas', size=300)
        heat_out = Flow(carrier='heat', size=200)

        charge_flow = Flow(carrier='heat', size=100)
        discharge_flow = Flow(carrier='heat', size=100)
        storage = Storage(id='heat_store', charging=charge_flow, discharging=discharge_flow, capacity=200.0)

        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='grid', imports=[gas_source]),
                Port(id='demand', exports=[demand_flow]),
            ],
            converters=[Converter.boiler('boiler', eta, fuel, heat_out)],
            storages=[storage],
        )

        # Verify the optimizer uses more gas in cheap hours
        gas_rates = result.flow_rate('grid(gas)').values
        assert gas_rates[0] > gas_rates[1]  # More gas bought in cheap hour

    def test_modified_data(self):
        """Build data, modify bounds, solve -- verify modified result."""

        sink_flow = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        source_flow = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source_flow]), Port(id='demand', exports=[sink_flow])],
        )

        # Change demand from 0.5 to 0.7 (relative); absolute = 0.7 * 100 = 70
        data.flows.fixed_profile = data.flows.fixed_profile.with_columns(
            pl.when(pl.col('flow') == 'demand(elec)').then(0.7).otherwise(pl.col('value')).alias('value')
        )

        result = solve(data, 'cost')

        source_rates = result.flow_rate('grid(elec)').values
        for rate in source_rates:
            assert rate == pytest.approx(70.0, abs=1e-6)

    def test_result_accessors(self):
        """Test Result accessor methods."""

        sink_flow = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source_flow]), Port(id='demand', exports=[sink_flow])],
        )

        # flow_rate accessor
        sr = result.flow_rate('grid(elec)')
        assert 'time' in sr.dims
        assert len(sr) == 3

        # effect_totals DataArray
        assert 'effect' in result.effect_totals.dims

        # effects_temporal
        assert 'effect' in result.effects_temporal.dims
        assert 'time' in result.effects_temporal.dims

        # effects_lump
        assert 'effect' in result.effects_lump.dims

    def test_int_timesteps(self):
        """Smoke test: int timesteps work end-to-end."""

        timesteps = [0, 1, 2, 3]

        demand_flow = Flow(carrier='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
        gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow(carrier='gas', size=300)
        heat_flow = Flow(carrier='heat', size=200)

        result = optimize(
            timesteps=timesteps,
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='grid', imports=[gas_source]),
                Port(id='demand', exports=[demand_flow]),
            ],
            converters=[Converter.boiler('boiler', 0.9, fuel, heat_flow)],
        )

        assert result.objective == pytest.approx(sum([40, 70, 50, 60]) / 0.9 * 0.04, abs=1e-6)
        sr = result.flow_rate('boiler(gas)')
        assert sr.dims == ('time',)
        assert len(sr) == 4
