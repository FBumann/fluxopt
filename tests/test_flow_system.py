"""FlowSystem: declarative construction, YAML IO, ProfileRef resolution."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from fluxopt import (
    Carrier,
    Effect,
    Flow,
    FlowSystem,
    Port,
    ProfileRef,
)


def _merit_order_spec(demand: object) -> FlowSystem:
    """Two priced sources meeting a fixed heat demand (see test_bus.py)."""
    return FlowSystem(
        timesteps=[0, 1],
        carriers=[Carrier(id='Heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)]),
            Port(id='Src1', imports=[Flow(carrier='Heat', size=20, effects_per_flow_hour={'cost': 1})]),
            Port(id='Src2', imports=[Flow(carrier='Heat', size=20, effects_per_flow_hour={'cost': 2})]),
        ],
    )


class TestPythonConstruction:
    def test_optimize_matches_free_function(self) -> None:
        result = _merit_order_spec(np.array([30, 30])).optimize()
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        assert_allclose(result.flow_rate('Src1(Heat)').values, [20, 20], rtol=1e-5)
        assert_allclose(result.flow_rate('Src2(Heat)').values, [10, 10], rtol=1e-5)


class TestRoundTrip:
    def test_dict_roundtrip_solves_identically(self) -> None:
        spec = _merit_order_spec([30, 30])
        rebuilt = FlowSystem.from_dict(spec.to_dict())
        assert_allclose(rebuilt.optimize().effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)

    def test_yaml_roundtrip(self, tmp_path) -> None:
        spec = _merit_order_spec([30, 30])
        path = tmp_path / 'system.yaml'
        spec.to_yaml(path)
        rebuilt = FlowSystem.from_yaml(path)
        assert_allclose(rebuilt.optimize().effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)


class TestProfileRefResolution:
    def _profiles(self, values: list[float]) -> dict[str, dict[str, xr.DataArray]]:
        return {'load': {'demand': xr.DataArray(values, dims=['time'])}}

    def test_ref_resolved_from_sources(self) -> None:
        spec = _merit_order_spec(ProfileRef(dataset='load', variable='demand'))
        result = spec.optimize(profiles=self._profiles([30, 30]))
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)

    def test_spec_reusable_across_sources(self) -> None:
        # Resolution runs on a copy, so the same spec solves with different data.
        spec = _merit_order_spec(ProfileRef(dataset='load', variable='demand'))
        c_low = spec.optimize(profiles=self._profiles([10, 10])).effect_totals.sel(effect='cost').item()
        c_high = spec.optimize(profiles=self._profiles([30, 30])).effect_totals.sel(effect='cost').item()
        assert c_low == pytest.approx(20.0)  # Src1 @1 covers 10 for 2h
        assert c_high == pytest.approx(80.0)  # Src1 @1 x20 + Src2 @2 x10, for 2h
        # The spec itself still carries the ProfileRef (not consumed).
        ref = spec.to_dict()['ports'][0]['exports'][0]['fixed_relative_profile']
        assert ref == {'dataset': 'load', 'variable': 'demand'}

    def test_missing_profiles_raises(self) -> None:
        spec = _merit_order_spec(ProfileRef(dataset='load', variable='demand'))
        with pytest.raises(KeyError, match='dataset'):
            spec.optimize()


class TestBuildModel:
    def test_build_data_returns_the_data_both_lanes_read(self) -> None:
        spec = _merit_order_spec([30, 30])
        data = spec.build_data()
        assert 'cost' in [str(e) for e in data.effects.total_min.coords['effect'].values]
        assert spec.optimize().effect_totals.sel(effect='cost').item() == pytest.approx(80.0)

    def test_math_is_readable_without_data(self) -> None:
        """The equations are an artefact, available before anything is bound."""
        math = _merit_order_spec([30, 30]).math()
        assert 'carrier_balance' in math.constraints
        assert 'rate' in math.variables
        assert 'constraints:' in math.to_yaml()

    def test_build_model_resolves_sources(self) -> None:
        spec = _merit_order_spec(ProfileRef(dataset='load', variable='demand'))
        profiles = {'load': {'demand': xr.DataArray([30.0, 30.0], dims=['time'])}}
        result = spec.optimize(profiles)
        assert result.effect_totals.sel(effect='cost').item() == pytest.approx(80.0)
        # spec still carries the ref — resolution ran on a copy
        assert isinstance(spec.ports[0].exports[0].fixed_relative_profile, ProfileRef)


class TestFreeOptimizeProfiles:
    def test_free_optimize_resolves_profiles(self) -> None:
        from fluxopt import optimize

        result = optimize(
            timesteps=[0, 1],
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(
                            carrier='Heat', size=1, fixed_relative_profile=ProfileRef(dataset='load', variable='demand')
                        )
                    ],
                ),
                Port(id='Src', imports=[Flow(carrier='Heat', size=40, effects_per_flow_hour={'cost': 1})]),
            ],
            profiles={'load': {'demand': xr.DataArray([30.0, 30.0], dims=['time'])}},
        )
        assert result.effect_totals.sel(effect='cost').item() == pytest.approx(60.0)


class TestProfileErgonomics:
    def _two_ref_spec(self) -> FlowSystem:
        return FlowSystem(
            timesteps=[0, 1],
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(
                            carrier='Heat', size=1, fixed_relative_profile=ProfileRef(dataset='load', variable='demand')
                        )
                    ],
                ),
                Port(
                    id='Src',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=40,
                            effects_per_flow_hour={'cost': ProfileRef(dataset='market', variable='price')},
                        )
                    ],
                ),
            ],
        )

    def test_required_profiles_enumerates_refs(self) -> None:
        assert self._two_ref_spec().required_profiles() == {'load': {'demand'}, 'market': {'price'}}

    def test_required_profiles_empty_for_inline_spec(self) -> None:
        assert _merit_order_spec([30, 30]).required_profiles() == {}

    def test_unresolvable_refs_reported_comprehensively(self) -> None:
        # one dataset missing entirely, one variable missing — a single error names both, with paths
        spec = self._two_ref_spec()
        with pytest.raises(KeyError) as exc:
            spec.optimize(profiles={'market': {'wrong_name': xr.DataArray([1.0, 1.0], dims=['time'])}})
        msg = str(exc.value)
        assert "dataset 'load' not supplied" in msg
        assert "variable 'price' not in dataset 'market'" in msg
        assert 'fixed_relative_profile' in msg  # element/field provenance
        assert 'effects_per_flow_hour' in msg
