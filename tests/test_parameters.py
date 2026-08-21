"""The parameter set is an artifact: readable, saveable, and the same on reload."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port
from fluxopt.math import Parameters

if TYPE_CHECKING:
    from pathlib import Path


def _system() -> FlowSystem:
    return FlowSystem(
        timesteps=ts(3),
        carriers=[Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([5, 8, 6]))]),
            Port(id='grid', imports=[Flow(carrier='heat', size=100, effects_per_flow_hour={'cost': 2})]),
        ],
    )


class TestReadable:
    def test_parameters_answers_the_other_half_of_math(self) -> None:
        """`math()` is the equations, `parameters()` the numbers bound to them."""
        system = _system()
        math, params = system.math(), system.parameters()

        assert set(params.parameters) >= set(math.parameters), (
            'every parameter the program declares should have a table bound for it'
        )
        assert set(params.dimensions) == set(math.dimensions)
        assert set(params.lookups) == set(math.lookups)

    def test_every_table_is_polars(self) -> None:
        """One format out, whatever the binder happened to build it with."""
        params = _system().parameters()
        for group in (params.dimensions, params.lookups, params.parameters):
            assert all(isinstance(f, pl.DataFrame) for f in group.values())

    def test_a_table_carries_the_numbers_the_elements_declared(self) -> None:
        """`effects_per_flow_hour` is the flow's rate times the step duration."""
        params = _system().parameters()
        charged = params['effects_per_flow_hour']
        assert charged['flow'].unique().to_list() == ['grid(heat)']
        assert charged['value'].to_list() == [2.0, 2.0, 2.0]

    def test_a_lookup_is_a_map_from_one_dimension_into_another(self) -> None:
        """`carrier_of: {over: flow, into: carrier}` is a `(flow, carrier)` table."""
        params = _system().parameters()
        carrier_of = params.lookups['carrier_of']
        assert carrier_of.columns == ['flow', 'carrier']
        assert carrier_of.sort('flow')['carrier'].to_list() == ['heat', 'heat']
        # The dimension keeps only its labels; the map is its own table.
        assert params.dimensions['flow'].columns == ['flow']

    def test_a_lookup_holds_only_the_labels_it_is_defined_at(self) -> None:
        """No storage here, so nothing charges one — an absent map, not null rows."""
        params = _system().parameters()
        assert params.lookups['charge_storage'].is_empty()
        assert params.lookups['carrier_of'].height == 2


class TestRoundtrip:
    def test_save_and_load_are_the_same_set(self, tmp_path: Path) -> None:
        params = _system().parameters()
        params.save(tmp_path / 'parameters')
        back = Parameters.load(tmp_path / 'parameters')

        for group in ('dimensions', 'lookups', 'parameters'):
            mine, theirs = getattr(params, group), getattr(back, group)
            assert set(theirs) == set(mine), group
            for name, frame in mine.items():
                assert theirs[name].equals(frame), f'{group}/{name}'

    def test_an_empty_table_keeps_its_dtypes(self, tmp_path: Path) -> None:
        """The reason it is parquet: a column with no rows still knows what it holds."""
        params = _system().parameters()
        empty = [n for n, f in params.parameters.items() if f.is_empty()]
        assert empty, 'this system declares no storage or status, so some tables are empty'

        params.save(tmp_path / 'parameters')
        back = Parameters.load(tmp_path / 'parameters')
        for name in empty:
            assert back.parameters[name].schema == params.parameters[name].schema, name

    def test_load_says_so_when_there_is_nothing_to_load(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match='No fluxopt parameter set'):
            Parameters.load(tmp_path)


class TestDerivedNotAuthored:
    def test_the_set_is_what_the_solver_adds_up_not_what_the_user_wrote(self) -> None:
        """A rate of 2/MWh over 1 h steps binds as 2; the fold is already applied.

        Recorded as a test because it is the reason the set is persistable but
        not editable — see docs/design/parameters-as-artifact.md.
        """
        system = FlowSystem(
            timesteps=ts(2),
            carriers=[Carrier(id='heat')],
            effects=[Effect(id='co2'), Effect(id='cost', contribution_from={'co2': 10.0})],
            objective='cost',
            ports=[
                Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([1, 1]))]),
                Port(id='grid', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'co2': 3})]),
            ],
        )
        charged = system.parameters()['effects_per_flow_hour']
        by_effect = dict(zip(charged['effect'], charged['value'], strict=True))
        # co2 as declared, and cost as folded through `contribution_from`
        assert by_effect['co2'] == 3.0
        assert by_effect['cost'] == 30.0


class TestSupplyingALookup:
    """A caller who adds a lookup in `math=` can supply it.

    A map is a source key of its own, so this is the same channel a parameter
    uses and needs nothing of ours: what refuses a bad one is lpspec, against
    the program, and those refusals are not restated here.
    """

    def _grouped(self) -> tuple[FlowSystem, object]:
        """A system, and math that groups its flows by a caller-named region."""
        system = FlowSystem(
            timesteps=ts(2),
            carriers=[Carrier(id='heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([4, 4]))]),
                Port(id='north', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'cost': 1})]),
                Port(id='south', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'cost': 5})]),
            ],
        )
        math = system.math()
        raw = math.to_dict()
        raw['dimensions']['region'] = {'dtype': 'str'}
        raw['lookups']['region_of'] = {'over': 'flow', 'into': 'region'}
        raw['parameters']['region_cap'] = {'dims': ['region', 'time', 'period']}
        raw['constraints']['region_limit'] = {
            'foreach': ['region', 'time', 'period'],
            'where': 'region_cap',
            'expression': 'sum(rate, by=region_of) <= region_cap',
        }
        import lpspec

        return system, lpspec.load_model(raw)

    def test_a_supplied_lookup_reaches_the_constraint_that_reads_it(self) -> None:
        """Capping the cheap region forces the expensive one, which changes the cost."""
        system, math = self._grouped()
        assert_allclose(system.optimize().effect_totals.sel(effect='cost').item(), 8.0, rtol=1e-6)

        result = system.optimize(
            math=math,
            dimensions={'region': pl.DataFrame({'region': ['cheap']})},
            lookups={'region_of': pl.DataFrame({'flow': ['north(heat)'], 'region': ['cheap']})},
            parameters={'region_cap': pl.DataFrame({'region': ['cheap'], 'time': [0], 'period': [0], 'value': [1.0]})},
        )
        # t0: 1 cheap + 3 expensive = 16; t1: 4 cheap = 4
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-6)

    def test_the_three_channels_are_the_three_declaration_blocks(self) -> None:
        """What `parameters()` returns under three names, `optimize` takes under three."""
        import inspect

        system, _ = self._grouped()
        taken = set(inspect.signature(system.optimize).parameters)
        returned = {f.name for f in dataclasses.fields(system.parameters())}
        assert returned <= taken, 'every kind the artifact reports should have a channel to supply it'
