from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest
import xarray as xr

from fluxopt import Bus, Converter, Effect, Flow, Port, Storage, optimize
from fluxopt.results import Result

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    return tmp_path / 'result.nc'


def _solve_simple(timesteps: list[datetime] | list[int]) -> Result:
    """Simple source -> demand system with cost tracking."""
    demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
    source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
    return optimize(
        timesteps=timesteps,
        buses=[Bus('elec')],
        effects=[Effect('cost', is_objective=True)],
        ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    )


def _solve_with_storage(timesteps: list[datetime]) -> Result:
    """Boiler + storage system."""
    demand = Flow(bus='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
    fuel = Flow(bus='gas', size=300)
    heat_out = Flow(bus='heat', size=200)
    charge = Flow(bus='heat', size=100)
    discharge = Flow(bus='heat', size=100)
    storage = Storage('heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return optimize(
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
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.objective == pytest.approx(result.objective, abs=1e-6)

    def test_with_storage(self, tmp_nc: Path) -> None:
        """Roundtrip: model with storage."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.objective == pytest.approx(result.objective, abs=1e-6)

    def test_model_data_preserved(self, tmp_nc: Path) -> None:
        """ModelData survives a NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        # Flows dataset preserved
        assert list(loaded.data.flows.rel_lb.coords['flow'].values) == list(
            result.data.flows.rel_lb.coords['flow'].values
        )
        # Effects attrs preserved
        assert loaded.data.effects.objective_effect == result.data.effects.objective_effect
        # Storages dataset preserved
        assert loaded.data.storages is not None
        assert result.data.storages is not None
        assert list(loaded.data.storages.capacity.coords['storage'].values) == list(
            result.data.storages.capacity.coords['storage'].values
        )
        # dt preserved
        assert loaded.data.dt.values == pytest.approx(result.data.dt.values)
        # time_extra preserved
        assert len(loaded.data.time_extra) == len(result.data.time_extra)

    def test_model_data_resolve(self, tmp_nc: Path) -> None:
        """Loaded ModelData can build and solve a new model."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)
        assert loaded.data is not None

        # Re-solve from loaded data
        from fluxopt import FlowSystem

        model = FlowSystem(loaded.data)
        model.build()
        result2 = model.solve()
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestRoundtripContributionFrom:
    def test_roundtrip_with_contribution_from(self, tmp_nc: Path) -> None:
        """ModelData with contribution_from survives NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts,
            buses=[Bus('elec')],
            effects=[
                Effect('cost', is_objective=True, contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )
        assert result.data is not None
        assert result.data.effects.cf_invest is not None
        assert result.data.effects.cf_per_hour is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert loaded.data.effects.cf_invest is not None
        assert loaded.data.effects.cf_per_hour is not None
        xr.testing.assert_equal(loaded.data.effects.cf_invest, result.data.effects.cf_invest)
        xr.testing.assert_equal(loaded.data.effects.cf_per_hour, result.data.effects.cf_per_hour)

        # Re-solve gives same objective
        from fluxopt import FlowSystem

        model = FlowSystem(loaded.data)
        model.build()
        result2 = model.solve()
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestSolutionDataset:
    def test_solution_is_dataset(self) -> None:
        """solution is an xr.Dataset with solution data."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)

        ds = result.solution
        assert isinstance(ds, xr.Dataset)
        assert 'flow--rate' in ds
        assert ds.attrs['objective'] == pytest.approx(result.objective)


class TestEdgeCases:
    def test_no_data_field(self, tmp_nc: Path) -> None:
        """Result without data field still serializes solution."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)
        result.data = None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.objective == pytest.approx(result.objective, abs=1e-6)
