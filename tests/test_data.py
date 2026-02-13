from __future__ import annotations

import polars as pl
import pytest

from fluxopt import Bus, Converter, Effect, Flow, Port, Sizing, Storage, build_model_data


class TestFlowsTable:
    def test_bounds_with_size(self, timesteps_3):
        flow = Flow(bus='b', size=100, relative_minimum=0.2, relative_maximum=0.8)
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
        )
        bounds = data.flows.relative_bounds.filter(pl.col('flow') == 'src(b)')
        assert bounds['rel_lb'].to_list() == [0.2, 0.2, 0.2]
        assert bounds['rel_ub'].to_list() == [0.8, 0.8, 0.8]
        sizes = data.flows.sizes.filter(pl.col('flow') == 'src(b)')
        assert sizes['size'].to_list() == [100.0]

    def test_fixed_profile(self, timesteps_3):
        flow = Flow(bus='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('sink', exports=[flow])],
        )
        fixed = data.flows.fixed.filter(pl.col('flow') == 'sink(b)')
        assert fixed['value'].to_list() == [0.5, 0.8, 0.6]


class TestBusesTable:
    def test_coefficients(self, timesteps_3):
        out_flow = Flow(bus='b', size=100)
        in_flow = Flow(bus='b', size=100)
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[out_flow]), Port('sink', exports=[in_flow])],
        )
        coeffs = data.buses.flow_coefficients
        out_coeff = coeffs.filter(pl.col('flow') == 'src(b)')['coeff'][0]
        in_coeff = coeffs.filter(pl.col('flow') == 'sink(b)')['coeff'][0]
        assert out_coeff == 1.0  # output to bus
        assert in_coeff == -1.0  # input from bus


class TestConvertersTable:
    def test_scalar_factors(self, timesteps_3):
        fuel = Flow(bus='gas', size=200)
        heat = Flow(bus='heat', size=100)
        boiler = Converter.boiler('boiler', 0.9, fuel, heat)
        data = build_model_data(
            timesteps_3,
            [Bus('gas'), Bus('heat')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[Flow(bus='gas', size=200)])],
            converters=[boiler],
        )
        coeffs = data.converters.flow_coefficients
        fuel_coeff = coeffs.filter(pl.col('flow') == 'boiler(gas)')['coeff'].unique().to_list()
        heat_coeff = coeffs.filter(pl.col('flow') == 'boiler(heat)')['coeff'].unique().to_list()
        assert fuel_coeff == [0.9]
        assert heat_coeff == [-1.0]


class TestEffectsTable:
    def test_flow_coefficients(self, timesteps_3):
        flow = Flow(bus='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[flow])],
        )
        coeffs = data.flows.effect_coefficients
        assert len(coeffs) == 3  # one per timestep
        assert coeffs['coeff'].unique().to_list() == [0.04]

    def test_objective_effect(self, timesteps_3):
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True), Effect('co2')],
            ports=[Port('src', imports=[Flow(bus='b', size=100)])],
        )
        assert data.effects.objective_effect == 'cost'


class TestFlowsTableSizing:
    def test_sizing_flow_has_null_size(self, timesteps_3):
        """A Sizing flow stores size=null in sizes DataFrame."""
        flow = Flow(bus='b', size=Sizing(min_size=10, max_size=200))
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
        )
        sizes = data.flows.sizes.filter(pl.col('flow') == 'src(b)')
        assert sizes['size'][0] is None

    def test_sizing_params_populated(self, timesteps_3):
        """Sizing parameters are decomposed into sizing_params DataFrame."""
        flow = Flow(bus='b', size=Sizing(min_size=10, max_size=200, mandatory=True))
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
        )
        sp = data.flows.sizing_params
        assert len(sp) == 1
        row = sp.row(0, named=True)
        assert row['flow'] == 'src(b)'
        assert row['min_size'] == 10.0
        assert row['max_size'] == 200.0
        assert row['mandatory'] is True

    def test_sizing_effects(self, timesteps_3):
        """Sizing effect dicts are merged into sizing_effects DataFrame."""
        flow = Flow(
            bus='b',
            size=Sizing(
                min_size=0,
                max_size=100,
                effects_per_size={'cost': 50.0},
                effects_of_size={'cost': 1000.0, 'co2': 5.0},
            ),
        )
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True), Effect('co2')],
            ports=[Port('src', imports=[flow])],
        )
        se = data.flows.sizing_effects.sort('effect')
        assert len(se) == 2
        cost_row = se.filter(pl.col('effect') == 'cost').row(0, named=True)
        assert cost_row['per_size'] == 50.0
        assert cost_row['of_size'] == 1000.0
        co2_row = se.filter(pl.col('effect') == 'co2').row(0, named=True)
        assert co2_row['per_size'] == 0.0
        assert co2_row['of_size'] == 5.0

    def test_mixed_sizing_and_fixed(self, timesteps_3):
        """Mix of Sizing and fixed-size flows: only Sizing has sizing_params."""
        sizable = Flow(bus='b', id='sizable', size=Sizing(min_size=0, max_size=100))
        fixed = Flow(bus='b', id='fixed', size=50.0)
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[sizable, fixed])],
        )
        assert len(data.flows.sizing_params) == 1
        assert data.flows.sizing_params['flow'][0] == 'src(sizable)'
        fixed_size = data.flows.sizes.filter(pl.col('flow') == 'src(fixed)')['size'][0]
        assert fixed_size == 50.0

    def test_non_sizing_has_empty_sizing_params(self, timesteps_3):
        """Flows without Sizing have empty sizing DataFrames."""
        flow = Flow(bus='b', size=100)
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
        )
        assert len(data.flows.sizing_params) == 0
        assert len(data.flows.sizing_effects) == 0

    def test_unknown_sizing_effect_rejected(self, timesteps_3):
        """Referencing a nonexistent effect in Sizing raises ValueError."""
        flow = Flow(bus='b', size=Sizing(min_size=0, max_size=100, effects_per_size={'bogus': 1.0}))
        with pytest.raises(ValueError, match='unknown effect'):
            build_model_data(
                timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
            )


class TestStoragesTableSizing:
    def test_sizable_capacity_is_null(self, timesteps_3):
        """Storage with Sizing capacity stores capacity=null in params."""
        charge = Flow(bus='b', size=50)
        discharge = Flow(bus='b', size=50)
        stor = Storage('bat', charging=charge, discharging=discharge, capacity=Sizing(min_size=50, max_size=200))
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[Flow(bus='b', size=200)])],
            storages=[stor],
        )
        cap = data.storages.params.filter(pl.col('storage') == 'bat')['capacity'][0]
        assert cap is None
        sp = data.storages.sizing_params
        assert len(sp) == 1
        assert sp['min_size'][0] == 50.0
        assert sp['max_size'][0] == 200.0

    def test_fixed_capacity_unchanged(self, timesteps_3):
        """Storage with scalar capacity keeps it in params, no sizing_params."""
        charge = Flow(bus='b', size=50)
        discharge = Flow(bus='b', size=50)
        stor = Storage('bat', charging=charge, discharging=discharge, capacity=100.0)
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[Flow(bus='b', size=200)])],
            storages=[stor],
        )
        cap = data.storages.params.filter(pl.col('storage') == 'bat')['capacity'][0]
        assert cap == 100.0
        assert len(data.storages.sizing_params) == 0

    def test_cs_bounds_are_relative(self, timesteps_3):
        """cs_bounds stores relative values (not multiplied by capacity)."""
        charge = Flow(bus='b', size=50)
        discharge = Flow(bus='b', size=50)
        stor = Storage(
            'bat',
            charging=charge,
            discharging=discharge,
            capacity=100.0,
            relative_minimum_charge_state=0.1,
            relative_maximum_charge_state=0.9,
        )
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[Flow(bus='b', size=200)])],
            storages=[stor],
        )
        cs = data.storages.cs_bounds.filter(pl.col('storage') == 'bat')
        assert cs['rel_cs_lb'].unique().to_list() == [0.1]
        assert cs['rel_cs_ub'].unique().to_list() == [0.9]
