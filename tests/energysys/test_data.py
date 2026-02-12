from __future__ import annotations

import polars as pl

from energysys import Bus, Effect, Flow, Sink, Source, build_model_data


class TestFlowsTable:
    def test_bounds_with_size(self, timesteps_3):
        flow = Flow('f1', bus='b', size=100, relative_minimum=0.2, relative_maximum=0.8)
        Source('src', outputs=[flow])
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], [Source('src', outputs=[flow])]
        )
        # Re-collect since Source.__post_init__ mutates flow
        bounds = data.flows.bounds.filter(pl.col('flow') == 'f1')
        assert bounds['lb'].to_list() == [20.0, 20.0, 20.0]
        assert bounds['ub'].to_list() == [80.0, 80.0, 80.0]

    def test_fixed_profile(self, timesteps_3):
        flow = Flow('f1', bus='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            [Sink('sink', inputs=[flow])],
        )
        fixed = data.flows.fixed.filter(pl.col('flow') == 'f1')
        assert fixed['value'].to_list() == [50.0, 80.0, 60.0]


class TestBusesTable:
    def test_coefficients(self, timesteps_3):
        out_flow = Flow('out', bus='b', size=100)
        in_flow = Flow('in', bus='b', size=100)
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            [Source('src', outputs=[out_flow]), Sink('sink', inputs=[in_flow])],
        )
        coeffs = data.buses.flow_coefficients
        out_coeff = coeffs.filter(pl.col('flow') == 'out')['coeff'][0]
        in_coeff = coeffs.filter(pl.col('flow') == 'in')['coeff'][0]
        assert out_coeff == 1.0  # output to bus
        assert in_coeff == -1.0  # input from bus


class TestConvertersTable:
    def test_scalar_factors(self, timesteps_3):
        from energysys import LinearConverter

        fuel = Flow('fuel', bus='gas', size=200)
        heat = Flow('heat', bus='heat', size=100)
        boiler = LinearConverter.boiler('boiler', 0.9, fuel, heat)
        data = build_model_data(
            timesteps_3,
            [Bus('gas'), Bus('heat')],
            [Effect('cost', is_objective=True)],
            [boiler, Source('src', outputs=[Flow('g', bus='gas', size=200)])],
        )
        coeffs = data.converters.flow_coefficients
        fuel_coeff = coeffs.filter(pl.col('flow') == 'fuel')['coeff'].unique().to_list()
        heat_coeff = coeffs.filter(pl.col('flow') == 'heat')['coeff'].unique().to_list()
        assert fuel_coeff == [0.9]
        assert heat_coeff == [-1.0]


class TestEffectsTable:
    def test_flow_coefficients(self, timesteps_3):
        flow = Flow('f', bus='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            [Source('src', outputs=[flow])],
        )
        coeffs = data.effects.flow_coefficients
        assert len(coeffs) == 3  # one per timestep
        assert coeffs['coeff'].unique().to_list() == [0.04]

    def test_objective_effect(self, timesteps_3):
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True), Effect('co2')],
            [Source('src', outputs=[Flow('f', bus='b', size=100)])],
        )
        assert data.effects.objective_effect == 'cost'
