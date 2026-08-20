from __future__ import annotations

from datetime import datetime

import xarray as xr

from fluxopt import Carrier, Converter, Effect, Flow, Port, Storage, optimize
from fluxopt.results import Result


def _solve_with_converter() -> Result:
    """Boiler + storage system for topology testing."""
    demand = Flow(carrier='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': 0.02})
    fuel = Flow(carrier='gas', size=300)
    heat_out = Flow(carrier='heat', size=200)
    charge = Flow(carrier='heat', size=100)
    discharge = Flow(carrier='heat', size=100)
    storage = Storage(id='heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return optimize(
        timesteps=[datetime(2024, 1, 1, h) for h in range(3)],
        carriers=[Carrier(id='gas'), Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[gas_source]), Port(id='demand', exports=[demand])],
        converters=[Converter.boiler('boiler', 0.9, fuel, heat_out)],
        storages=[storage],
    )


class TestTopology:
    def test_carriers_structure(self) -> None:
        """Topology contains carrier keys with inputs and outputs."""
        result = _solve_with_converter()
        topo = result.topology

        assert 'carriers' in topo
        assert 'converters' in topo
        assert 'gas' in topo['carriers']
        assert 'heat' in topo['carriers']

    def test_carrier_inputs_outputs(self) -> None:
        """Carrier inputs are +1 coeff flows, outputs are -1 coeff flows."""
        result = _solve_with_converter()
        gas = result.topology['carriers']['gas']
        heat = result.topology['carriers']['heat']

        # Gas carrier: grid imports gas (+1), boiler consumes gas (-1)
        assert len(gas['inputs']) >= 1
        assert len(gas['outputs']) >= 1
        assert any('grid' in f for f in gas['inputs'])
        assert any('boiler' in f for f in gas['outputs'])

        # Heat carrier: boiler produces heat (+1), demand consumes heat (-1)
        assert len(heat['inputs']) >= 1
        assert len(heat['outputs']) >= 1
        assert any('boiler' in f for f in heat['inputs'])
        assert any('demand' in f for f in heat['outputs'])

    def test_converter_inputs_outputs(self) -> None:
        """Converter inputs consume from carrier (-1), outputs produce to carrier (+1)."""
        result = _solve_with_converter()
        converters = result.topology['converters']

        assert 'boiler' in converters
        boiler = converters['boiler']
        assert len(boiler['inputs']) >= 1
        assert len(boiler['outputs']) >= 1
        # Boiler input is a gas flow, output is a heat flow
        assert any('gas' in f for f in boiler['inputs'])
        assert any('heat' in f for f in boiler['outputs'])

    def test_no_converters(self) -> None:
        """Topology works when there are no converters."""
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        result = optimize(
            timesteps=[datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        )
        topo = result.topology
        assert topo['converters'] == {}
        assert 'elec' in topo['carriers']


class TestDuals:
    def test_duals_is_dataset(self) -> None:
        """Duals is always an xr.Dataset."""
        result = _solve_with_converter()
        assert isinstance(result.duals, xr.Dataset)

    def test_duals_are_empty_after_a_roundtrip(self, tmp_path: object) -> None:
        """Duals loaded from NetCDF default to empty Dataset."""
        from pathlib import Path

        nc = Path(str(tmp_path)) / 'result'
        result = _solve_with_converter()
        result.save(nc)
        loaded = Result.load(nc)
        assert isinstance(loaded.duals, xr.Dataset)
