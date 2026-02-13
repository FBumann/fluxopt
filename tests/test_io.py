from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest

from fluxopt import Bus, Converter, Effect, Flow, Port, Storage, solve
from fluxopt.results import SolvedModel

xr = pytest.importorskip('xarray')

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    return tmp_path / 'result.nc'


def _solve_simple(timesteps: list[datetime] | list[int]) -> SolvedModel:
    """Simple source -> demand system with cost tracking."""
    demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
    source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
    return solve(
        timesteps=timesteps,
        buses=[Bus('elec')],
        effects=[Effect('cost', is_objective=True)],
        ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    )


def _solve_with_converter(timesteps: list[datetime]) -> SolvedModel:
    """Gas source -> boiler -> heat demand with cost tracking."""
    demand = Flow(bus='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5])
    gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})
    fuel = Flow(bus='gas', size=300)
    heat = Flow(bus='heat', size=200)
    return solve(
        timesteps=timesteps,
        buses=[Bus('gas'), Bus('heat')],
        effects=[Effect('cost', is_objective=True)],
        ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand])],
        converters=[Converter.boiler('boiler', 0.9, fuel, heat)],
    )


def _solve_with_storage(timesteps: list[datetime]) -> SolvedModel:
    """Boiler + storage system."""
    demand = Flow(bus='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
    fuel = Flow(bus='gas', size=300)
    heat_out = Flow(bus='heat', size=200)
    charge = Flow(bus='heat', size=100)
    discharge = Flow(bus='heat', size=100)
    storage = Storage('heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return solve(
        timesteps=timesteps,
        buses=[Bus('gas'), Bus('heat')],
        effects=[Effect('cost', is_objective=True)],
        ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand])],
        converters=[Converter.boiler('boiler', 0.9, fuel, heat_out)],
        storages=[storage],
    )


class TestRoundtrip:
    def test_simple_datetime(self, tmp_nc: Path) -> None:
        """Roundtrip: simple model with datetime timesteps."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.objective_value == pytest.approx(result.objective_value, abs=1e-6)
        assert loaded.flow_rates['solution'].to_list() == pytest.approx(
            result.flow_rates['solution'].to_list(), abs=1e-6
        )
        assert loaded.effects['solution'].to_list() == pytest.approx(result.effects['solution'].to_list(), abs=1e-6)
        assert loaded.effects_per_timestep['solution'].to_list() == pytest.approx(
            result.effects_per_timestep['solution'].to_list(), abs=1e-6
        )

    def test_simple_int_timesteps(self, tmp_nc: Path) -> None:
        """Roundtrip with integer timesteps."""
        result = _solve_simple([0, 1, 2])

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.objective_value == pytest.approx(result.objective_value, abs=1e-6)
        assert loaded.flow_rates['time'].dtype == pl.Int64

    def test_with_converter(self, tmp_nc: Path) -> None:
        """Roundtrip: model with converter."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_converter(ts)

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.objective_value == pytest.approx(result.objective_value, abs=1e-6)
        assert len(loaded.flow_rates) == len(result.flow_rates)
        assert len(loaded.contributions) == len(result.contributions)

    def test_with_storage(self, tmp_nc: Path) -> None:
        """Roundtrip: model with storage."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.objective_value == pytest.approx(result.objective_value, abs=1e-6)
        assert len(loaded.charge_states) == len(result.charge_states)
        assert loaded.charge_states['solution'].to_list() == pytest.approx(
            result.charge_states['solution'].to_list(), abs=1e-6
        )

    def test_contributions_roundtrip(self, tmp_nc: Path) -> None:
        """Verify contributions survive roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_converter(ts)
        assert len(result.contributions) > 0

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        # Same shape
        assert len(loaded.contributions) == len(result.contributions)
        # Same total
        assert loaded.contributions['solution'].sum() == pytest.approx(result.contributions['solution'].sum(), abs=1e-6)


class TestXarrayDataset:
    def test_root_group_shows_solution(self, tmp_nc: Path) -> None:
        """xr.open_dataset(path) shows solution data by default."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)
        result.to_netcdf(tmp_nc)

        ds = xr.open_dataset(tmp_nc)
        assert 'flow_rates' in ds
        assert 'effects' in ds
        assert 'objective_value' in ds.attrs
        ds.close()

    def test_model_group_accessible(self, tmp_nc: Path) -> None:
        """xr.open_dataset(path, group='model') shows model data."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)
        result.to_netcdf(tmp_nc)

        ds = xr.open_dataset(tmp_nc, group='model')
        assert 'dt' in ds
        assert 'weight' in ds
        assert 'flow_lb' in ds
        ds.close()

    def test_to_xarray_returns_dataset(self) -> None:
        """to_xarray() returns an xr.Dataset with solution data."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)

        ds = result.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert 'flow_rates' in ds
        assert ds.attrs['objective_value'] == pytest.approx(result.objective_value)


class TestModelDataRoundtrip:
    def test_model_data_preserved(self, tmp_nc: Path) -> None:
        """Model data is preserved through roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert loaded.data.effects.objective_effect == result.data.effects.objective_effect
        assert len(loaded.data.flows.relative_bounds) == len(result.data.flows.relative_bounds)
        assert loaded.data.dt['dt'].to_list() == pytest.approx(result.data.dt['dt'].to_list())

    def test_storage_model_data(self, tmp_nc: Path) -> None:
        """Storage model data is preserved through roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert len(loaded.data.storages.params) == len(result.data.storages.params)
        assert loaded.data.storages.params['capacity'].to_list() == result.data.storages.params['capacity'].to_list()


class TestEdgeCases:
    def test_no_data_field(self, tmp_nc: Path) -> None:
        """SolvedModel without data field still serializes solution."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)
        result.data = None

        result.to_netcdf(tmp_nc)
        loaded = SolvedModel.from_netcdf(tmp_nc)

        assert loaded.objective_value == pytest.approx(result.objective_value, abs=1e-6)
        # Model data is None since it wasn't written
        assert loaded.data is None
