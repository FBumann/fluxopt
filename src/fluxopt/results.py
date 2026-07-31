from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr

from fluxopt.contract import Var

try:
    from fluxopt_plot.accessor import PlotAccessor  # pyrefly: ignore[missing-import]
except ImportError:
    PlotAccessor = None

if TYPE_CHECKING:
    from fluxopt.model import FlowSystemModel
    from fluxopt.model_data import ModelData
    from fluxopt.stats import StatsAccessor


@dataclass
class Result:
    """Optimization result with solution variables and model data.

    Provides access to flow rates, storage levels, effect totals, and
    investment decisions. Key properties::

        result.objective  # scalar objective value
        result.flow_rates  # (flow, time) DataArray
        result.flow_rate('id')  # single flow time series
        result.storage_levels  # (storage, time) DataArray
        result.effect_totals  # (effect,) DataArray
        result.effects_temporal  # (effect, time) DataArray
        result.effects_lump  # (effect,) DataArray
        result.sizes  # (flow,) DataArray — invested sizes
        result.storage_capacities  # (storage,) DataArray

    Per-contributor effect breakdown is available via ``result.stats``.

    Args:
        solution: Solved variable values as xr.Dataset.
        data: ModelData used to build the optimization.
        duals: Dual values (shadow prices) from the solver.
        contributions: Cached *direct* per-contributor effect breakdown (no
            cross-effect propagation). Surfaced via
            ``result.stats.effect_contributions_direct``;
            ``result.stats.effect_contributions`` applies Leontief on top.
    """

    solution: xr.Dataset
    data: ModelData = field(repr=False)
    duals: xr.Dataset = field(default_factory=xr.Dataset, repr=False)
    contributions: xr.Dataset | None = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Objective function value."""
        return float(self.solution.attrs['objective'])

    @property
    def objective_weights(self) -> dict[str, float]:
        """Effect weights the objective was minimized with (provenance).

        Includes the built-in penalty effect (auto-added at 1.0 unless
        named in ``objective``). Empty for results saved before
        this field existed.
        """
        return json.loads(self.solution.attrs.get('objective_weights', '{}'))

    @property
    def flow_rates(self) -> xr.DataArray:
        """All flow rates as (flow, time) DataArray."""
        return self.solution[Var.FLOW_RATE]

    @property
    def storage_levels(self) -> xr.DataArray:
        """All storage levels as (storage, time) DataArray."""
        return self.solution[Var.STORAGE_LEVEL] if Var.STORAGE_LEVEL in self.solution else xr.DataArray()

    @property
    def sizes(self) -> xr.DataArray:
        """Optimized flow sizes as (flow,) DataArray."""
        return self.solution[Var.FLOW_SIZE] if Var.FLOW_SIZE in self.solution else xr.DataArray()

    @property
    def storage_capacities(self) -> xr.DataArray:
        """Optimized storage capacities as (storage,) DataArray."""
        return self.solution[Var.STORAGE_CAPACITY] if Var.STORAGE_CAPACITY in self.solution else xr.DataArray()

    @property
    def effect_totals(self) -> xr.DataArray:
        """Total effect values as (effect,) DataArray."""
        return self.solution[Var.EFFECT_TOTAL]

    @property
    def effects_temporal(self) -> xr.DataArray:
        """Per-timestep effect values as (effect, time) DataArray.

        Reconstructed from flow rates and coefficients — the model carries
        no per-timestep effect variables (temporal closure).
        """
        return self.stats.effect_contributions['temporal'].sum('contributor')

    @property
    def effects_lump(self) -> xr.DataArray:
        """Non-temporal effect values as (effect,) DataArray."""
        return self.solution[Var.EFFECT_LUMP]

    def flow_rate(self, flow_id: str) -> xr.DataArray:
        """Get flow rate time series for a single flow.

        Args:
            flow_id: Qualified flow id.
        """
        return self.flow_rates.sel(flow=flow_id)

    def storage_level(self, storage_id: str) -> xr.DataArray:
        """Get charge state time series for a single storage.

        Args:
            storage_id: Storage id.
        """
        return self.storage_levels.sel(storage=storage_id)

    @cached_property
    def topology(self) -> dict[Literal['carriers', 'converters'], dict[str, dict[str, list[str]]]]:
        """Carrier and converter connectivity derived from model data.

        Returns a dict with ``carriers`` and ``converters`` keys, each mapping
        element ids to their ``inputs`` (flows that produce into the element)
        and ``outputs`` (flows that consume from it).
        """
        fc = self.data.carriers.flow_coeff  # (carrier, flow), +1/-1/NaN

        # Per-flow sign lookup: nanmax collapses carrier dim (each flow has exactly one)
        flow_ids = [str(f) for f in fc.coords['flow'].values]
        signs = fc.max('carrier').values  # (flow,) — +1 or -1

        carriers: dict[str, dict[str, list[str]]] = {}
        for cid in fc.coords['carrier'].values:
            row = fc.sel(carrier=cid).dropna('flow')
            carriers[str(cid)] = {
                'inputs': list(row.coords['flow'].values[row.values > 0]),
                'outputs': list(row.coords['flow'].values[row.values < 0]),
            }

        flow_sign = dict(zip(flow_ids, signs, strict=True))

        converters: dict[str, dict[str, list[str]]] = {}
        if self.data.converters is not None:
            cd = self.data.converters
            # Deduplicate pairs (pair dim may repeat per equation index)
            pairs = dict.fromkeys(zip(cd.pair_converter.values, cd.pair_flow.values, strict=True))
            for conv_id, fid in pairs:
                conv_id, fid = str(conv_id), str(fid)
                if conv_id not in converters:
                    converters[conv_id] = {'inputs': [], 'outputs': []}
                target = 'inputs' if flow_sign[fid] < 0 else 'outputs'
                converters[conv_id][target].append(fid)

        return {'carriers': carriers, 'converters': converters}

    @cached_property
    def stats(self) -> StatsAccessor:
        """Post-processing statistics accessor."""
        from fluxopt.stats import StatsAccessor

        return StatsAccessor(self)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution and model data to NetCDF.

        Args:
            path: Output file path.
        """
        p = Path(path)
        self.solution.to_netcdf(p, mode='w', engine='netcdf4')
        self.data.to_netcdf(p)
        if self.contributions is not None:
            self.contributions.to_netcdf(p, mode='a', group='contributions', engine='netcdf4')

    @classmethod
    def from_netcdf(cls, path: str | Path) -> Result:
        """Read a Result from a NetCDF file.

        Args:
            path: Input file path.

        Raises:
            ValueError: On Windows when reading a non-ASCII path (netcdf4 limitation).
        """
        from fluxopt.model_data import ModelData, _raise_netcdf_read_error

        p = Path(path)
        try:
            solution = xr.load_dataset(p, engine='netcdf4')
        except OSError as e:
            _raise_netcdf_read_error(p, e)
        data = ModelData.from_netcdf(p)

        try:
            contributions = xr.load_dataset(p, group='contributions', engine='netcdf4')
        except OSError:
            contributions = None
            import warnings

            warnings.warn(
                f"NetCDF file {p} has no 'contributions' group; per-contributor effect "
                'breakdown will be re-derived from solution + ModelData on first access. '
                'Results may differ from the original solve if the contribution-decomposition '
                'logic has changed since the file was written. Re-save the Result to refresh '
                'the cached breakdown.',
                stacklevel=2,
            )

        return cls(solution=solution, data=data, contributions=contributions)

    @cached_property
    def plot(self) -> PlotAccessor:
        """Plotting accessor (requires ``fluxopt-plot``)."""
        if PlotAccessor is None:
            raise ImportError('Plotting requires fluxopt-plot. Install it with: pip install fluxopt-plot')
        return PlotAccessor(self)

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> Result:
        """Extract solution from a solved linopy model.

        Args:
            model: Solved FlowSystemModel instance.
        """
        sol_vars: dict[str, xr.DataArray] = {
            Var.FLOW_RATE: model.flow_rate.solution,
            Var.EFFECT_TOTAL: model.effect_total.solution,
            Var.EFFECT_LUMP: model.effect_lump.solution,
        }

        if model.storage_level is not None:
            sol_vars[Var.STORAGE_LEVEL] = model.storage_level.solution
        if model.flow_size is not None:
            sol_vars[Var.FLOW_SIZE] = model.flow_size.solution
        if model.flow_size_indicator is not None:
            sol_vars[Var.FLOW_SIZE_INDICATOR] = model.flow_size_indicator.solution
        if model.storage_capacity is not None:
            sol_vars[Var.STORAGE_CAPACITY] = model.storage_capacity.solution
        if model.storage_capacity_indicator is not None:
            sol_vars[Var.STORAGE_SIZE_INDICATOR] = model.storage_capacity_indicator.solution
        if model.invest_size is not None:
            sol_vars[Var.INVEST_SIZE] = model.invest_size.solution
        if model.invest_build is not None:
            sol_vars[Var.INVEST_BUILD] = model.invest_build.solution
        if model.invest_active is not None:
            sol_vars[Var.INVEST_ACTIVE] = model.invest_active.solution
        if model.invest_size_at_build is not None:
            sol_vars[Var.INVEST_SIZE_AT_BUILD] = model.invest_size_at_build.solution
        if model.flow_on is not None:
            sol_vars[Var.FLOW_ON] = model.flow_on.solution
        if model.flow_startup is not None:
            sol_vars[Var.FLOW_STARTUP] = model.flow_startup.solution
        if model.flow_shutdown is not None:
            sol_vars[Var.FLOW_SHUTDOWN] = model.flow_shutdown.solution
        if model.component_on is not None:
            sol_vars[Var.COMPONENT_ON] = model.component_on.solution
        if model.component_startup is not None:
            sol_vars[Var.COMPONENT_STARTUP] = model.component_startup.solution
        if model.component_shutdown is not None:
            sol_vars[Var.COMPONENT_SHUTDOWN] = model.component_shutdown.solution

        # Piecewise auxiliary variables (from linopy.add_piecewise_formulation).
        # Stored under their linopy-generated names so they survive IO roundtrip.
        for formulation in model._piecewise.values():
            for var_name in formulation.variable_names:
                if var_name not in sol_vars:
                    sol_vars[var_name] = model.m.variables[var_name].solution

        # Include custom variables added after build()
        for var_name in model.m.variables:
            if var_name not in model._builtin_var_names and var_name not in sol_vars:
                sol_vars[var_name] = model.m.variables[var_name].solution

        raw = model.m.objective.value
        obj_val = float(raw) if raw is not None else 0.0

        solution = xr.Dataset(
            sol_vars,
            attrs={'objective': obj_val, 'objective_weights': json.dumps(model._objective_weights)},
        )
        if model.data.dims.time_period is not None:
            solution = solution.assign_coords(time_period=model.data.dims.time_period)
        duals = model.m.dual

        from fluxopt.contributions import _with_cross_effects, compute_effect_contributions

        try:
            # Cache the direct (no cross-effect) view — it's the primitive both
            # accessors build on. effect_contributions applies Leontief on top
            # via _with_cross_effects, which is cheap relative to _compute_direct.
            contributions = compute_effect_contributions(solution, model.data, cross_effects=False)
            # Sanity-check at solve time: applying cross-effects must reproduce
            # the solver's effect--total. Result discarded; caches stay direct.
            _with_cross_effects(contributions, model.data, solution)
        except Exception as exc:
            import warnings

            warnings.warn(
                f'Failed to compute effect contributions during solve ({exc!r}); '
                'result.contributions will be None (re-derive via result.stats.effect_contributions)',
                stacklevel=2,
            )
            contributions = None

        return cls(solution=solution, data=model.data, duals=duals, contributions=contributions)
