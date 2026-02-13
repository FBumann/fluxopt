from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from fluxopt import (
    Bus,
    Effect,
    Flow,
    LinearConverter,
    Sink,
    Source,
    Storage,
    build_model_data,
    solve,
)
from fluxopt.model import FlowSystemModel


class TestEndToEnd:
    def test_full_system(self):
        """Full system: gas source → boiler → heat bus ← demand, with cost tracking."""
        timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
        eta = 0.9
        heat_demand = [40.0, 70.0, 50.0, 60.0]

        demand_flow = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
        gas_source = Flow('grid(gas)', bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow('boiler(gas)', bus='gas', size=300)
        heat = Flow('boiler(heat)', bus='heat', size=200)

        result = solve(
            timesteps=timesteps,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_source]),
                Sink('demand', inputs=[demand_flow]),
                LinearConverter.boiler('boiler', eta, fuel, heat),
            ],
        )

        # Verify gas = heat / eta
        for i, hd in enumerate(heat_demand):
            gas_rate = result.flow_rate('boiler(gas)')['value'][i]
            assert gas_rate == pytest.approx(hd / eta, abs=1e-6)

        # Verify cost
        total_gas = sum(h / eta for h in heat_demand)
        expected_cost = total_gas * 0.04
        assert result.objective_value == pytest.approx(expected_cost, abs=1e-6)

    def test_boiler_plus_storage(self):
        """Boiler + thermal storage: store heat in cheap hours."""
        timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
        eta = 0.9
        gas_prices = [0.02, 0.08, 0.02, 0.08]

        demand_flow = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])
        gas_source = Flow('grid(gas)', bus='gas', size=500, effects_per_flow_hour={'cost': gas_prices})
        fuel = Flow('boiler(gas)', bus='gas', size=300)
        heat_out = Flow('boiler(heat)', bus='heat', size=200)

        charge_flow = Flow('store(charge)', bus='heat', size=100)
        discharge_flow = Flow('store(discharge)', bus='heat', size=100)
        storage = Storage('heat_store', charging=charge_flow, discharging=discharge_flow, capacity=200.0)

        result = solve(
            timesteps=timesteps,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_source]),
                Sink('demand', inputs=[demand_flow]),
                LinearConverter.boiler('boiler', eta, fuel, heat_out),
            ],
            storages=[storage],
        )

        # Verify the optimizer uses more gas in cheap hours
        gas_t0 = result.flow_rate('grid(gas)')['value'][0]
        gas_t1 = result.flow_rate('grid(gas)')['value'][1]
        assert gas_t0 > gas_t1  # More gas bought in cheap hour

    def test_modified_data(self, timesteps_3):
        """Build data, modify bounds, solve — verify modified result."""
        sink_flow = Flow('sink', bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        source_flow = Flow('source', bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        data = build_model_data(
            timesteps_3,
            [Bus('elec')],
            [Effect('cost', is_objective=True)],
            [Source('grid', outputs=[source_flow]), Sink('demand', inputs=[sink_flow])],
        )

        # Change demand from 50 to 70 by modifying fixed values
        data.flows.fixed = data.flows.fixed.with_columns(pl.lit(70.0).alias('value'))

        model = FlowSystemModel(data)
        model.build()
        result = model.solve()

        source_rates = result.flow_rate('source')['value'].to_list()
        for rate in source_rates:
            assert rate == pytest.approx(70.0, abs=1e-6)

    def test_result_accessors(self, timesteps_3):
        """Test SolvedModel accessor methods."""
        sink_flow = Flow('sink', bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow('source', bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            components=[Source('grid', outputs=[source_flow]), Sink('demand', inputs=[sink_flow])],
        )

        # flow_rate accessor
        sr = result.flow_rate('source')
        assert set(sr.columns) == {'time', 'value'}
        assert len(sr) == 3

        # effects DataFrame
        assert 'effect' in result.effects.columns
        assert 'total' in result.effects.columns

        # effects_per_timestep
        assert 'effect' in result.effects_per_timestep.columns
        assert 'time' in result.effects_per_timestep.columns

    def test_int_timesteps(self):
        """Smoke test: int timesteps work end-to-end."""
        timesteps = [0, 1, 2, 3]

        demand_flow = Flow('demand(heat)', bus='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
        gas_source = Flow('grid(gas)', bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})
        fuel = Flow('boiler(gas)', bus='gas', size=300)
        heat = Flow('boiler(heat)', bus='heat', size=200)

        result = solve(
            timesteps=timesteps,
            buses=[Bus('gas'), Bus('heat')],
            effects=[Effect('cost', is_objective=True)],
            components=[
                Source('grid', outputs=[gas_source]),
                Sink('demand', inputs=[demand_flow]),
                LinearConverter.boiler('boiler', 0.9, fuel, heat),
            ],
        )

        assert result.objective_value > 0
        sr = result.flow_rate('boiler(gas)')
        assert sr['time'].dtype == pl.Int64
        assert len(sr) == 4
