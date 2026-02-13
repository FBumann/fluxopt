from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from fluxopt.model import FlowSystemModel
    from fluxopt.tables import ModelData


@dataclass
class SolvedModel:
    objective_value: float
    flow_rates: pl.DataFrame  # (flow, time, solution)
    charge_states: pl.DataFrame  # (storage, time, solution)
    effects: pl.DataFrame  # (effect, solution)
    effects_per_timestep: pl.DataFrame  # (effect, time, solution)
    contributions: pl.DataFrame  # (source, contributor, effect, time, solution)
    data: ModelData | None = field(default=None, repr=False)

    def flow_rate(self, id: str) -> pl.DataFrame:
        """Get time series for a single flow."""
        return self.flow_rates.filter(pl.col('flow') == id).select('time', 'solution')

    def charge_state(self, id: str) -> pl.DataFrame:
        """Get time series for a single storage."""
        return self.charge_states.filter(pl.col('storage') == id).select('time', 'solution')

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> SolvedModel:
        m = model.m
        d = model.data
        time_dtype = d.timesteps.schema['time']

        # Extract flow rates
        flow_sol = m.flow_rate.solution

        # Extract charge states (if storages exist)
        if len(d.storages.index) > 0:
            cs_sol = m.charge_state.solution
        else:
            cs_sol = pl.DataFrame(schema={'storage': pl.String, 'time': time_dtype, 'solution': pl.Float64})

        # Extract effects
        if len(d.effects.index) > 0:
            effect_total_sol = m.effect_total.solution
            effect_ts_sol = m.effect_per_timestep.solution
        else:
            effect_total_sol = pl.DataFrame(schema={'effect': pl.String, 'solution': pl.Float64})
            effect_ts_sol = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'solution': pl.Float64})

        # Extract per-source contributions
        contrib_frames: list[pl.DataFrame] = []
        for src in model._temporal_sources:
            var_name = f'contributions({src.name})'
            var = getattr(m, var_name)
            sol = var.solution.with_columns(pl.lit(src.name).alias('source'))
            contrib_frames.append(sol.select('source', 'contributor', 'effect', 'time', 'solution'))

        if contrib_frames:
            contributions = pl.concat(contrib_frames)
        else:
            contributions = pl.DataFrame(
                schema={
                    'source': pl.String,
                    'contributor': pl.String,
                    'effect': pl.String,
                    'time': time_dtype,
                    'solution': pl.Float64,
                }
            )

        # Objective value
        obj_effect = d.effects.objective_effect
        obj_val = effect_total_sol.filter(pl.col('effect') == obj_effect)['solution'][0]

        return cls(
            objective_value=obj_val,
            flow_rates=flow_sol,
            charge_states=cs_sol,
            effects=effect_total_sol,
            effects_per_timestep=effect_ts_sol,
            contributions=contributions,
            data=model.data,
        )

    def to_xarray(self) -> xr.Dataset:
        """Convert solution data to an xarray Dataset."""
        from fluxopt.io import solved_model_to_xarray

        return solved_model_to_xarray(self)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution + model data to a NetCDF file."""
        from fluxopt.io import solved_model_to_netcdf

        solved_model_to_netcdf(self, path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> SolvedModel:
        """Read a SolvedModel from a NetCDF file."""
        from fluxopt.io import solved_model_from_netcdf

        return solved_model_from_netcdf(path)
