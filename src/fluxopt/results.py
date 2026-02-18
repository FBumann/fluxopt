from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from fluxopt.model import FlowSystem
    from fluxopt.model_data import ModelData


@dataclass
class Result:
    solution: xr.Dataset
    data: ModelData | None = field(default=None, repr=False)
    _contributions_cache: xr.Dataset | None = field(default=None, repr=False, init=False)

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
        """All storage levels as (storage, time) DataArray."""
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
    def effects_temporal(self) -> xr.DataArray:
        """Per-timestep effect values as (effect, time) DataArray."""
        return self.solution['effect--temporal']

    @property
    def effects_periodic(self) -> xr.DataArray:
        """Per-period (investment) effect values as (effect,) DataArray."""
        return self.solution['effect--periodic']

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

    def effect_contributions(self) -> xr.Dataset:
        """Per-flow breakdown of effect contributions.

        Returns:
            Dataset with ``operational`` (flow, effect, time),
            ``investment`` (flow, effect), ``total`` (flow, effect),
            and optionally ``storage_investment`` (storage, effect).

        Raises:
            ValueError: If ``data`` is not available on this Result.
        """
        if self._contributions_cache is not None:
            return self._contributions_cache

        if self.data is None:
            raise ValueError('ModelData is required for effect_contributions (not available on this Result)')

        from fluxopt.contributions import compute_effect_contributions

        self._contributions_cache = compute_effect_contributions(self.solution, self.data)
        return self._contributions_cache

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
    def from_netcdf(cls, path: str | Path) -> Result:
        """Read a Result from a NetCDF file.

        Args:
            path: Input file path.
        """
        from fluxopt.model_data import ModelData

        p = Path(path)
        solution = xr.open_dataset(p, engine='netcdf4')
        data = ModelData.from_netcdf(p)
        return cls(solution=solution, data=data)

    @classmethod
    def from_model(cls, model: FlowSystem) -> Result:
        """Extract solution from a solved linopy model.

        Args:
            model: Solved FlowSystem instance.
        """
        sol_vars: dict[str, xr.DataArray] = {
            'flow--rate': model.flow_rate.solution,
            'effect--total': model.effect_total.solution,
            'effect--temporal': model.effect_temporal.solution,
            'effect--periodic': model.effect_periodic.solution,
        }

        if model.storage_level is not None:
            sol_vars['storage--level'] = model.storage_level.solution
        if model.flow_size is not None:
            sol_vars['flow--size'] = model.flow_size.solution
        if model.flow_size_indicator is not None:
            sol_vars['flow--size_indicator'] = model.flow_size_indicator.solution
        if model.storage_capacity is not None:
            sol_vars['storage--capacity'] = model.storage_capacity.solution
        if model.storage_capacity_indicator is not None:
            sol_vars['storage--size_indicator'] = model.storage_capacity_indicator.solution
        if model.flow_on is not None:
            sol_vars['flow--on'] = model.flow_on.solution
        if model.flow_startup is not None:
            sol_vars['flow--startup'] = model.flow_startup.solution
        if model.flow_shutdown is not None:
            sol_vars['flow--shutdown'] = model.flow_shutdown.solution

        # Include custom variables added after build()
        for var_name in model.m.variables:
            if var_name not in model._builtin_var_names and var_name not in sol_vars:
                sol_vars[var_name] = model.m.variables[var_name].solution

        obj_effect = model.data.effects.objective_effect
        obj_val = float(sol_vars['effect--total'].sel(effect=obj_effect).values)

        solution = xr.Dataset(sol_vars, attrs={'objective': obj_val})
        return cls(solution=solution, data=model.data)
