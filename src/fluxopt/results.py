from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from fluxopt.model import FlowSystemModel
    from fluxopt.tables import ModelData


@dataclass
class SolvedModel:
    objective_value: float
    flow_rates: xr.DataArray  # (flow, time)
    charge_states: xr.DataArray  # (storage, time_extra)
    effects: xr.DataArray  # (effect,)
    effects_per_timestep: xr.DataArray  # (effect, time)
    data: ModelData | None = field(default=None, repr=False)

    def flow_rate(self, id: str) -> xr.DataArray:
        """Get time series for a single flow."""
        return self.flow_rates.sel(flow=id)

    def charge_state(self, id: str) -> xr.DataArray:
        """Get time series for a single storage."""
        return self.charge_states.sel(storage=id)

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> SolvedModel:
        """Extract solution from a solved linopy model."""
        d = model.data

        # linopy .solution returns xr.DataArray directly
        flow_sol = model.flow_rate.solution
        cs_sol = model.charge_state.solution if hasattr(model, 'charge_state') else xr.DataArray()
        effect_total_sol = model.effect_total.solution if hasattr(model, 'effect_total') else xr.DataArray()
        effect_ts_sol = model.effect_per_timestep.solution if hasattr(model, 'effect_per_timestep') else xr.DataArray()

        obj_effect = d.effects.attrs['objective_effect']
        obj_val = float(effect_total_sol.sel(effect=obj_effect).values)

        return cls(
            objective_value=obj_val,
            flow_rates=flow_sol,
            charge_states=cs_sol,
            effects=effect_total_sol,
            effects_per_timestep=effect_ts_sol,
            data=model.data,
        )

    def to_xarray(self) -> xr.Dataset:
        """Convert solution data to an xarray Dataset."""
        from fluxopt.io import solved_model_to_xarray

        return solved_model_to_xarray(self)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution to a NetCDF file."""
        from fluxopt.io import solved_model_to_netcdf

        solved_model_to_netcdf(self, path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SolvedModel:
        """Read a SolvedModel from a NetCDF file."""
        from fluxopt.io import solved_model_from_netcdf

        return solved_model_from_netcdf(path)
