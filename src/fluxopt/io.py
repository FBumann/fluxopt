"""NetCDF IO for fluxopt — serialize SolvedModel to/from NetCDF via xarray.

Requires the ``io`` extra for file IO: ``pip install fluxopt[io]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from fluxopt.results import SolvedModel


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
    """Write a SolvedModel to a NetCDF file."""
    ds = solved_model_to_xarray(result)
    ds.to_netcdf(Path(path), mode='w')


def solved_model_from_netcdf(path: str | Path) -> SolvedModel:
    """Read a SolvedModel from a NetCDF file."""
    from fluxopt.results import SolvedModel

    ds = xr.open_dataset(path)
    objective_value = float(ds.attrs['objective_value'])

    flow_rates = ds['flow_rates'] if 'flow_rates' in ds else xr.DataArray()
    charge_states = ds['charge_states'] if 'charge_states' in ds else xr.DataArray()
    effects = ds['effects'] if 'effects' in ds else xr.DataArray()
    effects_per_timestep = ds['effects_per_timestep'] if 'effects_per_timestep' in ds else xr.DataArray()

    ds.close()

    return SolvedModel(
        objective_value=objective_value,
        flow_rates=flow_rates,
        charge_states=charge_states,
        effects=effects,
        effects_per_timestep=effects_per_timestep,
    )
