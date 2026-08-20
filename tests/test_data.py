from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from conftest import ts
from pydantic import ValidationError

from fluxopt import (
    Carrier,
    Converter,
    Dims,
    Effect,
    Flow,
    FlowSystem,
    ModelData,
    Port,
    ProfileRef,
    Storage,
    optimize,
)


class TestFlowsTable:
    def test_bounds_with_size(self):
        flow = Flow(carrier='b', size=100, relative_rate_min=0.2, relative_rate_max=0.8)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        ds = data.flows
        lb = ds.rel_lb.sel(flow='src(b)').values
        ub = ds.rel_ub.sel(flow='src(b)').values
        assert list(lb) == [0.2, 0.2, 0.2]
        assert list(ub) == [0.8, 0.8, 0.8]
        assert float(ds.size.sel(flow='src(b)').values) == 100.0
        assert str(ds.bound_type.sel(flow='src(b)').values) == 'bounded'

    def test_fixed_profile(self):
        flow = Flow(carrier='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='sink', exports=[flow])],
        )
        fixed = data.flows.fixed_profile.sel(flow='sink(b)').values
        assert list(fixed) == [0.5, 0.8, 0.6]
        assert str(data.flows.bound_type.sel(flow='sink(b)').values) == 'profile'

    def test_unsized_flow(self):
        flow = Flow(carrier='b')
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        assert str(data.flows.bound_type.sel(flow='src(b)').values) == 'unsized'


class TestCarriersData:
    def test_coefficients(self):
        out_flow = Flow(carrier='b', size=100)
        in_flow = Flow(carrier='b', size=100)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[out_flow]), Port(id='sink', exports=[in_flow])],
        )
        cd = data.carriers
        assert float(cd.sign.sel(flow='src(b)')) == 1.0  # output to carrier
        assert float(cd.sign.sel(flow='sink(b)')) == -1.0  # input from carrier
        assert {str(c) for c in cd.carrier_of.values} == {'b'}

    def test_metadata(self):
        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='blue', description='Electricity')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        assert str(data.carriers.unit.sel(carrier='elec').values) == 'kWh'
        assert str(data.carriers.color.sel(carrier='elec').values) == 'blue'
        assert str(data.carriers.description.sel(carrier='elec').values) == 'Electricity'

    def test_from_dataset_roundtrip(self):
        from fluxopt.model_data import CarriersData

        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='red', description='Power')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        ds = data.carriers.to_dataset()
        loaded = CarriersData.from_dataset(ds)
        assert str(loaded.unit.sel(carrier='elec').values) == 'kWh'
        assert str(loaded.color.sel(carrier='elec').values) == 'red'
        assert str(loaded.description.sel(carrier='elec').values) == 'Power'


class TestConvertersTable:
    def test_scalar_factors(self):
        fuel = Flow(carrier='gas', size=200)
        heat_flow = Flow(carrier='heat', size=100)
        boiler = Converter.boiler('boiler', 0.9, fuel, heat_flow)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='gas', size=200)])],
            converters=[boiler],
        )
        ds = data.converters
        assert ds is not None
        fuel_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(gas)', time=data.dims.time[0]).values
        )
        heat_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(heat)', time=data.dims.time[0]).values
        )
        assert fuel_coeff == 0.9
        assert heat_coeff == -1.0


class TestEffectsTable:
    def test_flow_coefficients(self):
        flow = Flow(carrier='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        fds = data.flows
        # One row per (flow, effect) the flow actually charges — not a dense
        # product over every flow and every effect.
        assert list(fds.effect_pair_flow.values) == ['src(b)']
        assert list(fds.effect_pair_effect.values) == ['cost']
        assert all(v == 0.04 for v in fds.effect_pair_coeff.isel(effect_pair=0).values)


class TestFlowNodeId:
    def test_node_included_in_default_short_id(self):
        """Flow with node set auto-generates carrier:node short_id."""
        f = Flow(carrier='heat', node='A')
        assert f.short_id == 'heat:A'

    def test_node_without_node_uses_carrier(self):
        """Flow without node uses carrier as short_id."""
        f = Flow(carrier='heat')
        assert f.short_id == 'heat'


class TestStorageValidation:
    def test_mismatched_carriers_raises(self):
        """Storage with different charging/discharging carriers raises ValueError."""
        with pytest.raises(ValueError, match='charging carrier'):
            Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='heat'))

    def test_same_short_id_resolves_to_charge_discharge(self):
        """Colliding short_ids resolve to charge/discharge in the qualified ids only."""
        s = Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='elec'))
        assert s.charging.short_id == 'elec'  # declaration untouched
        assert s.discharging.short_id == 'elec'
        assert s._charging_id == 'bat(charge)'
        assert s._discharging_id == 'bat(discharge)'

    def test_distinct_short_ids_preserved(self):
        """Storage with explicit different short_ids keeps them in qualified id."""
        s = Storage(
            id='bat', charging=Flow(carrier='elec', short_id='in'), discharging=Flow(carrier='elec', short_id='out')
        )
        assert s._charging_id == 'bat(in)'
        assert s._discharging_id == 'bat(out)'


class TestFlowQualification:
    def test_declarations_are_never_mutated(self):
        """Placing a flow in a component leaves the flow object untouched."""
        f = Flow(carrier='elec')
        Port(id='grid', imports=[f])
        assert f.short_id == 'elec'
        assert not hasattr(f, 'id')

    def test_port_qualified_flows_carry_signs(self):
        buy, sell = Flow(carrier='elec', short_id='buy'), Flow(carrier='elec', short_id='sell')
        port = Port(id='grid', imports=[buy], exports=[sell])
        assert [(bf.id, bf.sign) for bf in port._qualified_flows()] == [
            ('grid(buy)', 1),
            ('grid(sell)', -1),
        ]

    def test_flow_reused_across_components_gets_two_entries(self):
        """One flow declaration placed in two components yields two dataset columns."""
        f = Flow(carrier='b', size=100)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[f]), Port(id='sink', exports=[f])],
        )
        assert list(data.flows.size.coords['flow'].values) == ['src(b)', 'sink(b)']

    def test_port_duplicate_short_ids_raise_at_construction(self):
        with pytest.raises(ValueError, match=r"Port 'grid': duplicate flow short_id\(s\) \['elec'\]"):
            Port(id='grid', imports=[Flow(carrier='elec')], exports=[Flow(carrier='elec')])

    def test_converter_duplicate_short_ids_raise_at_construction(self):
        with pytest.raises(ValueError, match=r"Converter 'c': duplicate flow short_id\(s\) \['gas'\]"):
            Converter(
                id='c',
                inputs=[Flow(carrier='gas'), Flow(carrier='gas')],
                outputs=[Flow(carrier='heat')],
                conversion_factors=[{'gas': 0.9, 'heat': -1}],
            )


class TestConverterValidation:
    def test_unknown_short_id_in_conversion_factors_raises(self):
        with pytest.raises(ValueError, match=r"unknown flow short_ids \['gas'\]"):
            Converter(
                id='boiler',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat')],
                conversion_factors=[{'gas': 0.9, 'Heat': -1}],
            )

    def test_unknown_short_id_reports_equation_index(self):
        with pytest.raises(ValueError, match=r'conversion_factors\[1\]'):
            Converter(
                id='chp',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat'), Flow(carrier='Elec')],
                conversion_factors=[
                    {'Gas': 0.5, 'Heat': -1},
                    {'Gas': 0.4, 'Electricity': -1},
                ],
            )

    def test_known_short_ids_pass(self):
        conv = Converter(
            id='boiler',
            inputs=[Flow(carrier='Gas')],
            outputs=[Flow(carrier='Heat')],
            conversion_factors=[{'Gas': 0.9, 'Heat': -1}],
        )
        assert conv.conversion_factors[0]['Gas'] == 0.9


class TestCarrierValidation:
    def test_undeclared_carrier_raises(self):
        """Flow referencing an undeclared carrier raises ValueError."""
        with pytest.raises(ValueError, match='undeclared carrier'):
            optimize(
                timesteps=ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                objective='cost',
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_undeclared_carrier_in_model_data_build(self):
        """ModelData.build rejects flows with undeclared carriers."""
        with pytest.raises(ValueError, match=r"undeclared carrier\(s\) \['elec'\]"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_duplicate_carrier_raises(self):
        """Duplicate carrier declarations raise ValueError."""
        with pytest.raises(ValueError, match='Duplicate carrier id'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='elec'), Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_flow_node_on_nodeless_carrier_raises(self):
        """Flow with node on a carrier without nodes raises ValueError."""
        with pytest.raises(ValueError, match='has no nodes'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat')],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='A', size=100)])],
            )

    def test_flow_node_not_in_carrier_nodes_raises(self):
        """Flow with node not declared on carrier raises ValueError."""
        with pytest.raises(ValueError, match="node='C'"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat', nodes=['A', 'B'])],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='C', size=100)])],
            )


class TestCarrierBalance:
    def test_carrier_balance_property(self):
        """StatsAccessor.carrier_balance returns each flow's signed contribution."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='src', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='sink', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        balance = result.stats.carrier_balance
        assert 'flow' in balance.dims
        # The carrier rides as a coordinate on the flow axis, not as an axis
        assert 'carrier' in balance.coords
        assert 'carrier' not in balance.dims
        # Source produces, sink consumes — grouped by carrier they cancel
        for val in balance.groupby('carrier').sum().sel(carrier='elec').values:
            assert val == pytest.approx(0.0, abs=1e-6)


class TestMultiNodeCarrier:
    def test_independent_node_balance(self):
        """Two flows on the same carrier but different nodes get independent balance equations."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        # Source A matches sink A demand (50 MW)
        rate_a = result.flow_rate('src_a(heat:A)').values
        for val in rate_a:
            assert val == pytest.approx(50.0, abs=1e-4)

        # Source B matches sink B demand (80 MW)
        rate_b = result.flow_rate('src_b(heat:B)').values
        for val in rate_b:
            assert val == pytest.approx(80.0, abs=1e-4)

    def test_node_in_carrier_dim_id(self):
        """Carrier dimension coordinates contain 'heat:A' and 'heat:B'."""
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        carrier_ids = list(data.carriers.unit.coords['carrier'].values)
        assert 'heat:A' in carrier_ids
        assert 'heat:B' in carrier_ids
        assert len(carrier_ids) == 2


class TestDimsValidation:
    def test_mismatched_dim_raises(self):
        """Dims rejects arrays that are not 1D with dims=('time',)."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        bad_dt = xr.DataArray([1.0, 1.0], dims=['other'])
        with pytest.raises(ValueError, match='must be 1D'):
            Dims(time=time, dt=bad_dt, weights=time)

    def test_mismatched_coords_raises(self):
        """Dims rejects arrays with different time coordinates."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        dt = xr.DataArray([1.0, 1.0], dims=['time'], coords={'time': [0, 1]})
        bad_weights = xr.DataArray(np.ones(3), dims=['time'], coords={'time': [0, 1, 2]})
        with pytest.raises(ValueError, match='does not match'):
            Dims(time=time, dt=dt, weights=bad_weights)


class TestContributionsAreDeclared:
    def test_the_program_names_every_contribution_the_ledger_sums(self):
        """The breakdown and the ledger are one declaration, so they must agree.

        `effect_temporal` and `effect_lump` sum exactly the expressions
        `contributions.py` reads back; a contribution added to one and not the
        other would attribute a cost nobody is charged, or charge one nobody
        is attributed.
        """
        import lpspec

        from fluxopt.contributions import LUMP, TEMPORAL
        from fluxopt.math import PROGRAM

        program = lpspec.load_model(PROGRAM)
        declared = set(TEMPORAL) | set(LUMP)
        assert declared <= set(program.expressions), 'a contribution is read that the program does not declare'
        summed = program.expressions['effect_temporal'].expression + program.expressions['effect_lump'].expression
        for name in declared:
            assert name in summed, f'{name} is read back but the ledger never sums it'


class TestStorageRanges:
    """Physical ranges are refused where the value is written."""

    def _storage(self, **kwargs):
        return Storage(id='b', charging=Flow(carrier='e'), discharging=Flow(carrier='e'), **kwargs)

    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'capacity': -5}, 'capacity is negative'),
            ({'eta_charge': 0}, r'eta_charge must be in \(0.0, 1.0\]'),
            ({'eta_discharge': 1.5}, r'eta_discharge must be in \(0.0, 1.0\]'),
            ({'relative_loss_per_hour': 1.4}, r'relative_loss_per_hour must be in \[0.0, 1.0\]'),
            ({'relative_loss_per_hour': [0.1, 1.4]}, 'relative_loss_per_hour must be in'),
        ],
    )
    def test_refused_at_construction(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            self._storage(**kwargs)

    def test_a_profile_ref_is_checked_when_it_is_resolved(self):
        """Its numbers live elsewhere, so the element cannot see them.

        This is the path that keeps the data-layer range check alive: an
        element accepts the reference, and the values only exist once
        profiles are bound.
        """
        system = FlowSystem(
            timesteps=ts(3),
            carriers=[Carrier(id='e')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='g', imports=[Flow(carrier='e', size=10, effects_per_flow_hour={'cost': 1.0})])],
            storages=[self._storage(capacity=10, eta_charge=ProfileRef(dataset='p', variable='eta'))],
        )
        with pytest.raises(ValueError, match='eta_charge must be in'):
            system.build_data({'p': {'eta': xr.DataArray([0.9, 0.9, 1.7], dims=['time'])}})
