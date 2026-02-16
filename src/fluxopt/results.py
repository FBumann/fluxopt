from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from fluxopt.model import FlowSystemModel
    from fluxopt.model_data import ModelData


@dataclass
class SolvedModel:
    solution: xr.Dataset
    data: ModelData | None = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Objective function value."""
        return float(self.solution.attrs['objective'])

    @property
    def flow_rates(self) -> xr.DataArray:
        """All flow rates as (flow, time) DataArray."""
        return self.solution['flow--rate']

    @property
    def storage_levels(self) -> xr.DataArray:
        """All storage levels as (storage, time_extra) DataArray."""
        return self.solution['storage--level'] if 'storage--level' in self.solution else xr.DataArray()

    @property
    def sizes(self) -> xr.DataArray:
        """Optimized flow sizes as (flow,) DataArray."""
        return self.solution['flow--size'] if 'flow--size' in self.solution else xr.DataArray()

    @property
    def storage_capacities(self) -> xr.DataArray:
        """Optimized storage capacities as (storage,) DataArray."""
        return self.solution['storage--capacity'] if 'storage--capacity' in self.solution else xr.DataArray()

    @property
    def effect_totals(self) -> xr.DataArray:
        """Total effect values as (effect,) DataArray."""
        return self.solution['effect--total']

    @property
    def effects_per_timestep(self) -> xr.DataArray:
        """Per-timestep effect values as (effect, time) DataArray."""
        return self.solution['effect--per_timestep']

    def flow_rate(self, id: str) -> xr.DataArray:
        """Get flow rate time series for a single flow.

        Args:
            id: Qualified flow id.
        """
        return self.flow_rates.sel(flow=id)

    def storage_level(self, id: str) -> xr.DataArray:
        """Get charge state time series for a single storage.

        Args:
            id: Storage id.
        """
        return self.storage_levels.sel(storage=id)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution and model data to NetCDF.

        Args:
            path: Output file path.
        """
        p = Path(path)
        self.solution.to_netcdf(p, mode='w', engine='netcdf4')
        if self.data is not None:
            self.data.to_netcdf(p)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SolvedModel:
        """Read a SolvedModel from a NetCDF file.

        Args:
            path: Input file path.
        """
        from fluxopt.model_data import ModelData

        p = Path(path)
        solution = xr.open_dataset(p, engine='netcdf4')
        data = ModelData.from_netcdf(p)
        return cls(solution=solution, data=data)

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> SolvedModel:
        """Extract solution from a solved linopy model.

        Args:
            model: Solved FlowSystemModel instance.
        """
        sol_vars: dict[str, xr.DataArray] = {
            'flow--rate': model.flow_rate.solution,
            'effect--total': model.effect_total.solution,
            'effect--per_timestep': model.effect_per_timestep.solution,
        }

        if hasattr(model, 'storage_level'):
            sol_vars['storage--level'] = model.storage_level.solution
        if hasattr(model, 'flow_size'):
            sol_vars['flow--size'] = model.flow_size.solution
        if hasattr(model, 'flow_size_indicator'):
            sol_vars['flow--size_indicator'] = model.flow_size_indicator.solution
        if hasattr(model, 'storage_capacity'):
            sol_vars['storage--capacity'] = model.storage_capacity.solution
        if hasattr(model, 'storage_size_indicator'):
            sol_vars['storage--size_indicator'] = model.storage_size_indicator.solution

        obj_effect = model.data.effects.objective_effect
        obj_val = float(sol_vars['effect--total'].sel(effect=obj_effect).values)

        solution = xr.Dataset(sol_vars, attrs={'objective': obj_val})
        return cls(solution=solution, data=model.data)
