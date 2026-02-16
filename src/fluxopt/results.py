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
        return self.solution['flow_rates']

    @property
    def charge_states(self) -> xr.DataArray:
        return self.solution['charge_states'] if 'charge_states' in self.solution else xr.DataArray()

    @property
    def effects(self) -> xr.DataArray:
        return self.solution['effects']

    @property
    def effects_per_timestep(self) -> xr.DataArray:
        return self.solution['effects_per_timestep']

    def flow_rate(self, id: str) -> xr.DataArray:
        """Get time series for a single flow."""
        return self.flow_rates.sel(flow=id)

    def charge_state(self, id: str) -> xr.DataArray:
        """Get time series for a single storage."""
        return self.charge_states.sel(storage=id)

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
            'flow_rates': model.flow_rate.solution,
            'effects': model.effect_total.solution,
            'effects_per_timestep': model.effect_per_timestep.solution,
        }

        if hasattr(model, 'charge_state'):
            sol_vars['charge_states'] = model.charge_state.solution

        obj_effect = model.data.effects.attrs['objective_effect']
        obj_val = float(sol_vars['effects'].sel(effect=obj_effect).values)

        solution = xr.Dataset(sol_vars, attrs={'objective': obj_val})
        return cls(solution=solution, data=model.data)
