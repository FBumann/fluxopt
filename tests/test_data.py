from __future__ import annotations

from fluxopt import Bus, Converter, Effect, Flow, Port, build_model_data


class TestFlowsTable:
    def test_bounds_with_size(self, timesteps_3):
        flow = Flow(bus='b', size=100, relative_minimum=0.2, relative_maximum=0.8)
        data = build_model_data(
            timesteps_3, [Bus('b')], [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])]
        )
        ds = data.flows
        lb = ds['rel_lb'].sel(flow='src(b)').values
        ub = ds['rel_ub'].sel(flow='src(b)').values
        assert list(lb) == [0.2, 0.2, 0.2]
        assert list(ub) == [0.8, 0.8, 0.8]
        assert float(ds['size'].sel(flow='src(b)').values) == 100.0

    def test_fixed_profile(self, timesteps_3):
        flow = Flow(bus='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('sink', exports=[flow])],
        )
        fixed = data.flows['fixed_profile'].sel(flow='sink(b)').values
        assert list(fixed) == [0.5, 0.8, 0.6]


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
        coeffs = data.buses['flow_coeff']
        out_coeff = float(coeffs.sel(bus='b', flow='src(b)').values)
        in_coeff = float(coeffs.sel(bus='b', flow='sink(b)').values)
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
        ds = data.converters
        fuel_coeff = float(
            ds['flow_coeff'].sel(converter='boiler', eq_idx=0, flow='boiler(gas)', time=data.time[0]).values
        )
        heat_coeff = float(
            ds['flow_coeff'].sel(converter='boiler', eq_idx=0, flow='boiler(heat)', time=data.time[0]).values
        )
        assert fuel_coeff == 0.9
        assert heat_coeff == -1.0


class TestEffectsTable:
    def test_flow_coefficients(self, timesteps_3):
        flow = Flow(bus='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[flow])],
        )
        coeff = data.flows['effect_coeff'].sel(flow='src(b)', effect='cost')
        assert all(v == 0.04 for v in coeff.values)

    def test_objective_effect(self, timesteps_3):
        data = build_model_data(
            timesteps_3,
            [Bus('b')],
            [Effect('cost', is_objective=True), Effect('co2')],
            ports=[Port('src', imports=[Flow(bus='b', size=100)])],
        )
        assert data.effects.attrs['objective_effect'] == 'cost'
