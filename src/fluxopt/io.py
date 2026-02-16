"""NetCDF IO for fluxopt — serialize SolvedModel to/from NetCDF via xarray.

File layout (NetCDF groups):
    /              — solution: flow_rates, charge_states, effects, objective_value
    /model/flows   — ModelData.flows
    /model/buses   — ModelData.buses
    /model/conv    — ModelData.converters
    /model/effects — ModelData.effects
    /model/stor    — ModelData.storages
    /model/meta    — dt, weights
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from fluxopt.results import SolvedModel
    from fluxopt.tables import ModelData

_MODEL_GROUPS = {
    'flows': 'model/flows',
    'buses': 'model/buses',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
}


def solved_model_to_xarray(result: SolvedModel) -> xr.Dataset:
    """Convert solution data to an xarray Dataset."""
    data_vars: dict[str, xr.DataArray] = {}

    if result.flow_rates.size > 0:
        data_vars['flow_rates'] = result.flow_rates
    if result.charge_states.size > 0:
        data_vars['charge_states'] = result.charge_states
    if result.effects.size > 0:
        data_vars['effects'] = result.effects
    if result.effects_per_timestep.size > 0:
        data_vars['effects_per_timestep'] = result.effects_per_timestep

    ds = xr.Dataset(data_vars)
    ds.attrs['objective_value'] = result.objective_value
    return ds


def solved_model_to_netcdf(result: SolvedModel, path: str | Path) -> None:
    """Write a SolvedModel to a NetCDF file (solution + model data)."""
    p = Path(path)

    # Root group: solution
    solved_model_to_xarray(result).to_netcdf(p, mode='w', engine='netcdf4')

    # Model data groups
    if result.data is not None:
        _write_model_data(result.data, p)


def solved_model_from_netcdf(path: str | Path) -> SolvedModel:
    """Read a SolvedModel from a NetCDF file."""
    from fluxopt.results import SolvedModel

    p = Path(path)

    # Root group: solution
    ds = xr.open_dataset(p, engine='netcdf4')
    objective_value = float(ds.attrs['objective_value'])

    flow_rates = ds['flow_rates'] if 'flow_rates' in ds else xr.DataArray()
    charge_states = ds['charge_states'] if 'charge_states' in ds else xr.DataArray()
    effects = ds['effects'] if 'effects' in ds else xr.DataArray()
    effects_per_timestep = ds['effects_per_timestep'] if 'effects_per_timestep' in ds else xr.DataArray()
    ds.close()

    # Try to load model data
    data = _read_model_data(p)

    return SolvedModel(
        objective_value=objective_value,
        flow_rates=flow_rates,
        charge_states=charge_states,
        effects=effects,
        effects_per_timestep=effects_per_timestep,
        data=data,
    )


def _write_model_data(data: ModelData, path: Path) -> None:
    """Write ModelData datasets as NetCDF groups."""
    for field, group in _MODEL_GROUPS.items():
        ds: xr.Dataset = getattr(data, field)
        if ds.data_vars:
            ds.to_netcdf(path, mode='a', group=group, engine='netcdf4')

    # Meta group: dt, weights
    meta = xr.Dataset({'dt': data.dt, 'weights': data.weights})
    meta.to_netcdf(path, mode='a', group='model/meta', engine='netcdf4')


def _read_model_data(path: Path) -> ModelData | None:
    """Read ModelData from NetCDF groups. Returns None if not present."""
    import pandas as pd

    from fluxopt.tables import ModelData

    try:
        meta = xr.open_dataset(path, group='model/meta', engine='netcdf4')
    except OSError:
        return None

    dt = meta['dt']
    weights = meta['weights']
    time = pd.Index(dt.coords['time'].values)
    meta.close()

    datasets: dict[str, xr.Dataset] = {}
    for field, group in _MODEL_GROUPS.items():
        try:
            datasets[field] = xr.open_dataset(path, group=group, engine='netcdf4')
        except OSError:
            datasets[field] = xr.Dataset()

    # Reconstruct time_extra from storages dataset or compute from time + dt
    if 'time_extra' in datasets['storages'].coords:
        time_extra = pd.Index(datasets['storages'].coords['time_extra'].values)
        time_extra.name = 'time_extra'
    else:
        from fluxopt.tables import _compute_time_extra

        time_extra = _compute_time_extra(time, dt)

    return ModelData(
        flows=datasets['flows'],
        buses=datasets['buses'],
        converters=datasets['converters'],
        effects=datasets['effects'],
        storages=datasets['storages'],
        time=time,
        dt=dt,
        weights=weights,
        time_extra=time_extra,
    )
