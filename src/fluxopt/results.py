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
        expressions: Every quantity the model *names*, evaluated at this
            solution. A named expression is evaluated against a solve, so it
            can only be had here — carrying them is what makes a ``Result``
            answer for the model rather than only for its variables. Read one
            with :meth:`expression`; the effect breakdown is a view over
            them (``result.stats.effect_contributions``).
    """

    solution: xr.Dataset
    data: ModelData = field(repr=False)
    duals: xr.Dataset = field(default_factory=xr.Dataset, repr=False)
    expressions: xr.Dataset = field(default_factory=xr.Dataset, repr=False)

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
        cd = self.data.carriers
        flow_ids = cd.membership['flow'].to_list()
        of = cd.membership['carrier'].to_list()
        signs = cd.membership['sign'].to_numpy()

        carriers: dict[str, dict[str, list[str]]] = {cid: {'inputs': [], 'outputs': []} for cid in cd.ids}
        for fid, carrier, sign in zip(flow_ids, of, signs, strict=True):
            carriers[carrier]['inputs' if sign > 0 else 'outputs'].append(fid)

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

    def expression(self, name: str) -> xr.DataArray:
        """One quantity the model names, at this solution.

        Whatever the program declares under ``expressions:`` — including
        anything a caller added through ``optimize(math=...)``.

        Args:
            name: The expression's name in the program.

        Raises:
            KeyError: Naming what this Result does carry, since a model that
                declared it and a Result that kept it are different things.
        """
        if name not in self.expressions:
            available = ', '.join(sorted(str(n) for n in self.expressions.data_vars)) or 'nothing'
            msg = f'{name!r} is not among the expressions this Result carries ({available})'
            raise KeyError(msg)
        return self.expressions[name]

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
        if self.expressions.data_vars:
            self.expressions.to_netcdf(p, mode='a', group='expressions', engine='netcdf4')

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
            expressions = xr.load_dataset(p, group='expressions', engine='netcdf4')
        except OSError:
            expressions = xr.Dataset()
            import warnings

            warnings.warn(
                f"NetCDF file {p} has no 'expressions' group, so this Result carries none of "
                'the quantities the model names — including the per-contributor effect '
                'breakdown. They are evaluated against a solve and cannot be recovered from '
                'the solution alone; re-solve, or re-save a Result that has them.',
                stacklevel=2,
            )

        return cls(solution=solution, data=data, expressions=expressions)

    @cached_property
    def plot(self) -> PlotAccessor:
        """Plotting accessor (requires ``fluxopt-plot``)."""
        if PlotAccessor is None:
            raise ImportError('Plotting requires fluxopt-plot. Install it with: pip install fluxopt-plot')
        return PlotAccessor(self)
