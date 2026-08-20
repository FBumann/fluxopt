from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
import xarray as xr

from fluxopt import Carrier, Converter, Effect, Flow, Port, Storage, optimize
from fluxopt.results import Result

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    return tmp_path / 'result.nc'


def _solve_simple(timesteps: list[datetime] | list[int]) -> Result:
    """Simple source -> demand system with cost tracking."""
    demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
    source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
    return optimize(
        timesteps=timesteps,
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
    )


def _solve_with_storage(timesteps: list[datetime]) -> Result:
    """Boiler + storage system."""
    demand = Flow(carrier='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
    fuel = Flow(carrier='gas', size=300)
    heat_out = Flow(carrier='heat', size=200)
    charge = Flow(carrier='heat', size=100)
    discharge = Flow(carrier='heat', size=100)
    storage = Storage(id='heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return optimize(
        timesteps=timesteps,
        carriers=[Carrier(id='gas'), Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[gas_source]), Port(id='demand', exports=[demand])],
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
        # Storages dataset preserved
        assert loaded.data.storages is not None
        assert result.data.storages is not None
        assert list(loaded.data.storages.capacity.coords['storage'].values) == list(
            result.data.storages.capacity.coords['storage'].values
        )
        # Dims roundtrip: dt, time, and weights preserved with coordinates
        xr.testing.assert_equal(loaded.data.dims.dt, result.data.dims.dt)
        xr.testing.assert_equal(loaded.data.dims.time, result.data.dims.time)
        xr.testing.assert_equal(loaded.data.dims.weights, result.data.dims.weights)

    def test_model_data_resolve(self, tmp_nc: Path) -> None:
        """Loaded ModelData can build and solve a new model."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)
        assert loaded.data is not None

        # Re-solve from loaded data
        from fluxopt.math import solve

        result2 = solve(loaded.data, 'cost')
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestUnicodePath:
    """Reading non-ASCII netCDF paths: clarify the misleading error on Windows.

    netcdf4/libnetcdf (through 4.9.3) fails to open files under non-ASCII
    *directories* on Windows with a misleading PermissionError. On a read
    failure fluxopt replaces it with an actionable message; the guard is purely
    reactive (it only fires if netcdf4 actually raises) and read-only. Other
    platforms are unaffected. See #189 and Unidata/netcdf4-python#1482.
    """

    @pytest.mark.parametrize(
        ('os_name', 'relpath', 'clarified'),
        [
            ('nt', 'ümlaut/r.nc', True),  # Windows + non-ASCII -> clarified ValueError
            ('nt', 'ascii/r.nc', False),  # Windows + ASCII -> original error passes through
            ('posix', 'ümlaut/r.nc', False),  # other platforms work -> original error passes through
        ],
    )
    def test_read_error_clarified_only_on_windows_nonascii(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, os_name: str, relpath: str, clarified: bool
    ) -> None:
        """Read failures get a clear message only for non-ASCII paths on Windows; else propagate."""
        from fluxopt.model_data import _raise_netcdf_read_error

        monkeypatch.setattr('fluxopt.model_data.os.name', os_name)
        original = PermissionError(13, 'Permission denied')
        with pytest.raises((ValueError, OSError)) as excinfo:
            _raise_netcdf_read_error(tmp_path / relpath, original)
        if clarified:
            assert isinstance(excinfo.value, ValueError)
            assert 'non-ASCII' in str(excinfo.value)
            assert excinfo.value.__cause__ is original  # original preserved in the chain
        else:
            assert excinfo.value is original  # untouched

    @pytest.mark.skipif(os.name != 'nt', reason='upstream bug is Windows-only')
    @pytest.mark.xfail(
        strict=True,
        reason='Upstream bug: netcdf4 cannot open files in non-ASCII dirs on Windows '
        '(Unidata/netcdf4-python#1482). When this XPASSes, upstream is fixed -- drop the '
        '_raise_netcdf_read_error guard.',
    )
    def test_upstream_netcdf4_nonascii_dir_canary(self, tmp_path: Path) -> None:
        """Probe raw netcdf4 directly; alerts us (strict xfail) the day upstream fixes this."""
        from netCDF4 import Dataset  # type: ignore[import-untyped]

        d = tmp_path / 'umlaut_äöü'
        d.mkdir()
        with Dataset(str(d / 'probe.nc'), 'w') as ds:
            ds.createDimension('x', 1)
            ds.createVariable('value', 'f4', ('x',))[:] = [42]


class TestCarrierMetadataRoundtrip:
    def test_carrier_metadata_preserved(self, tmp_nc: Path) -> None:
        """Carrier unit, color, and description survive a NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        result = optimize(
            timesteps=ts,
            carriers=[Carrier(id='elec', unit='kWh', color='#ff0000', description='Electrical energy')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        )
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert loaded.data.carriers.carriers.filter(pl.col('carrier') == 'elec')['unit'][0] == 'kWh'
        assert loaded.data.carriers.carriers.filter(pl.col('carrier') == 'elec')['color'][0] == '#ff0000'
        assert (
            loaded.data.carriers.carriers.filter(pl.col('carrier') == 'elec')['description'][0] == 'Electrical energy'
        )


class TestRoundtripContributionFrom:
    def test_roundtrip_with_contribution_from(self, tmp_nc: Path) -> None:
        """ModelData with contribution_from survives NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts,
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )
        assert result.data is not None
        assert result.data.effects.cf_pair_effect is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert loaded.data.effects.cf_pair_effect is not None
        # The pairs survive, and so does the matrix they build on demand.
        xr.testing.assert_equal(loaded.data.effects.cf_matrix(), result.data.effects.cf_matrix())

        # Re-solve gives same objective
        from fluxopt.math import solve

        result2 = solve(loaded.data, 'cost')
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


class TestExpressionsRoundtrip:
    def test_named_quantities_survive_a_roundtrip(self, tmp_nc: Path) -> None:
        """Everything the model names travels with the answer, not just its variables."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        assert result.expressions.data_vars

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert set(loaded.expressions.data_vars) == set(result.expressions.data_vars)
        for name in result.expressions.data_vars:
            xr.testing.assert_allclose(loaded.expressions[name], result.expressions[name])

    def test_the_breakdown_is_a_view_over_them(self, tmp_nc: Path) -> None:
        """Contributions are assembled from the stored expressions, so they survive too."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        for view in ('temporal', 'lump', 'total'):
            xr.testing.assert_allclose(loaded.stats.effect_contributions[view], result.stats.effect_contributions[view])

    def test_a_file_without_them_says_so_rather_than_re_deriving(self, tmp_nc: Path) -> None:
        """Re-deriving on load would answer with today's logic against yesterday's numbers."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        result.solution.to_netcdf(tmp_nc, mode='w', engine='netcdf4')
        result.data.to_netcdf(tmp_nc)

        with pytest.warns(UserWarning, match="no 'expressions' group"):
            loaded = Result.from_netcdf(tmp_nc)
        assert not loaded.expressions.data_vars
        with pytest.raises(ValueError, match='cannot re-derive'):
            _ = loaded.stats.effect_contributions

    def test_roundtrip_does_not_warn(self, tmp_nc: Path) -> None:
        """Loading a file that carries them emits no missing-group warning."""
        import warnings

        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        result.to_netcdf(tmp_nc)

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            loaded = Result.from_netcdf(tmp_nc)
        assert loaded.expressions.data_vars

    def test_netcdf_group_structure(self, tmp_nc: Path) -> None:
        """The saved file carries an 'expressions' group, readable without Result."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        result.to_netcdf(tmp_nc)

        solution = xr.load_dataset(tmp_nc)
        assert 'flow--rate' in solution

        expressions = xr.load_dataset(tmp_nc, group='expressions')
        assert 'contribution_flow_hour' in expressions
        assert set(expressions['contribution_flow_hour'].dims) >= {'flow', 'effect', 'time'}


class TestBuildValidation:
    def test_undeclared_effect_rejected_without_flow_system(self) -> None:
        """The raw ModelData.build path rejects undeclared effect references."""
        from fluxopt import ModelData

        with pytest.raises(ValueError, match=r"undeclared effect\(s\) \['co2'\]"):
            ModelData.build(
                [datetime(2024, 1, 1, h) for h in range(3)],
                carriers=[Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=10, effects_per_flow_hour={'co2': 1.0})])],
            )


class TestWaistGuards:
    """Model-semantic invariants enforced at the data level, not only on elements."""

    def _status_system_path(self, tmp_nc: Path) -> Path:
        from fluxopt import ModelData, Status

        boiler_fuel = Flow(carrier='elec', size=100, relative_rate_min=0.3, status=Status())
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[boiler_fuel]), Port(id='demand', exports=[demand])],
        )
        data.to_netcdf(tmp_nc, mode='w')
        return tmp_nc

    def test_zeroed_status_lower_bound_rejected_on_load(self, tmp_nc: Path) -> None:
        """rel_lb = 0 on a status flow would make on/off degenerate; load fails loudly."""
        import netCDF4

        from fluxopt import ModelData

        p = self._status_system_path(tmp_nc)
        with netCDF4.Dataset(p, 'a') as nc:
            nc['model/flows']['rel_lb'][:] = 0.0

        with pytest.raises(ValueError, match='on/off is indistinguishable'):
            ModelData.from_netcdf(p)

    def test_nan_size_on_ramp_flow_rejected_on_load(self, tmp_nc: Path) -> None:
        """Removing the size under a ramp-limited flow fails at load, not as NaN math."""
        import netCDF4
        import numpy as np

        from fluxopt import ModelData

        source = Flow(carrier='elec', size=100, ramp_up_per_hour=0.5)
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        )
        data.to_netcdf(tmp_nc, mode='w')
        with netCDF4.Dataset(tmp_nc, 'a') as nc:
            nc['model/flows']['size'][:] = np.nan

        with pytest.raises(ValueError, match='ramp_up requires a sized flow'):
            ModelData.from_netcdf(tmp_nc)

    def test_dangling_storage_flow_reference_rejected_on_load(self, tmp_nc: Path) -> None:
        """A charge_flow naming a nonexistent flow fails at load, not as a KeyError at build."""
        import netCDF4

        from fluxopt import ModelData, Storage

        charge = Flow(carrier='elec', size=50)
        discharge = Flow(carrier='elec', size=50)
        source = Flow(carrier='elec', size=100)
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
            storages=[Storage(id='bat', charging=charge, discharging=discharge, capacity=100.0)],
        )
        data.to_netcdf(tmp_nc, mode='w')
        with netCDF4.Dataset(tmp_nc, 'a') as nc:
            nc['model/stor']['charge_flow'][0] = 'bat(gone)'

        with pytest.raises(
            ValueError, match=r"storages\.charge_flow references unknown flow id\(s\) \['bat\(gone\)'\]"
        ):
            ModelData.from_netcdf(tmp_nc)
