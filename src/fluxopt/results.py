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
        return float(self.solution.attrs['objective'])

    @property
    def flow_rates(self) -> xr.DataArray:
        return self.solution['flow--rate']

    @property
    def storage_levels(self) -> xr.DataArray:
        return self.solution['storage--level'] if 'storage--level' in self.solution else xr.DataArray()

    @property
    def effect_totals(self) -> xr.DataArray:
        return self.solution['effect--total']

    @property
    def effects_per_timestep(self) -> xr.DataArray:
        return self.solution['effect--per_timestep']

    def flow_rate(self, id: str) -> xr.DataArray:
        """Get time series for a single flow."""
        return self.flow_rates.sel(flow=id)

    def storage_level(self, id: str) -> xr.DataArray:
        """Get time series for a single storage."""
        return self.storage_levels.sel(storage=id)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution (+ model data if available) to NetCDF."""
        p = Path(path)
        self.solution.to_netcdf(p, mode='w', engine='netcdf4')
        if self.data is not None:
            self.data.to_netcdf(p)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SolvedModel:
        """Read a SolvedModel from a NetCDF file."""
        from fluxopt.model_data import ModelData

        p = Path(path)
        solution = xr.open_dataset(p, engine='netcdf4')
        data = ModelData.from_netcdf(p)
        return cls(solution=solution, data=data)

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> SolvedModel:
        """Extract solution from a solved linopy model."""
        sol_vars: dict[str, xr.DataArray] = {
            'flow--rate': model.flow_rate.solution,
            'effect--total': model.effect_total.solution,
            'effect--per_timestep': model.effect_per_timestep.solution,
        }

        if hasattr(model, 'storage_level'):
            sol_vars['storage--level'] = model.storage_level.solution

        obj_effect = model.data.effects.attrs['objective_effect']
        obj_val = float(sol_vars['effect--total'].sel(effect=obj_effect).values)

        solution = xr.Dataset(sol_vars, attrs={'objective': obj_val})
        return cls(solution=solution, data=model.data)
