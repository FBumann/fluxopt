"""Derived statistics from optimization results.

Computes post-processing quantities that require ModelData (dt, weights)
— energy totals, effect contributions, solver metadata.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fluxopt.results import Result


class StatsAccessor:
    """Post-processing statistics for a solved optimization result.

    Accessed via ``result.stats``.

    Args:
        result: Solved Result.
    """

    def __init__(self, result: Result) -> None:
        self._result = result

    @cached_property
    def flow_hours(self) -> xr.DataArray:
        """Energy per flow per timestep: P_{f,t} * dt_t.

        Returns:
            DataArray (flow, time) in energy units (e.g. MWh).
        """
        return self._result.flow_rates * self._result.data.dims.dt

    @cached_property
    def total_flow_hours(self) -> xr.DataArray:
        """Total energy per flow, weighted; per period in multi-period models.

        Returns:
            DataArray (flow[, period]) — weighted sum of flow_hours over time.
        """
        dims = self._result.data.dims
        return dims.sum_time(self.flow_hours * dims.weights)

    @cached_property
    def carrier_balance(self) -> xr.DataArray:
        """Signed balance per carrier: coeff * P. (carrier, flow, time)."""
        coeff = self._result.data.carriers.flow_coeff
        return coeff * self._result.flow_rates

    @cached_property
    def effect_contributions_direct(self) -> xr.Dataset:
        """Per-contributor effect breakdown *without* cross-effect propagation.

        Each contributor only carries effects it directly emits —
        ``contribution_from`` chains are ignored. Useful for attributing
        physical quantities (e.g. raw CO₂ emissions) without conflating them
        with priced-in monetary effects.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        if self._result.contributions is not None:
            return self._result.contributions

        from fluxopt.contributions import compute_effect_contributions

        return compute_effect_contributions(self._result.solution, self._result.data, cross_effects=False)

    @cached_property
    def effect_contributions(self) -> xr.Dataset:
        """Per-contributor effect breakdown with cross-effects propagated.

        Decomposes effect totals into per-contributor parts on a unified
        ``contributor`` dimension (flow IDs + storage IDs)::

            contrib = result.stats.effect_contributions
            contrib['temporal']  # (contributor, effect, time)
            contrib['lump']  # (contributor, effect)
            contrib['total']  # (contributor, effect) — temporal sum + lump

        Cross-effects (e.g. CO₂ → cost via ``Effect.contribution_from``) are
        propagated through the Leontief inverse, so each contributor is
        charged the full priced-in cost. The contributions are validated
        against solver totals; a ``ValueError`` is raised if they don't match.

        Built on top of :attr:`effect_contributions_direct` — when both views
        are accessed, the heavy direct computation runs only once.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        from fluxopt.contributions import _with_cross_effects

        return _with_cross_effects(
            self.effect_contributions_direct,
            self._result.data,
            self._result.solution,
        )

    @cached_property
    def resolved_sizes(self) -> xr.DataArray:
        """Per-flow size resolved from declared or optimized values.

        Unlike :attr:`Result.sizes` (optimized/invested flows only), this
        carries every flow — fixed sizes as declared, invested sizes from the
        solution, and NaN for unsized flows.

        Named ``resolved_*`` rather than ``installed_*`` on purpose: advanced
        multi-period investment (#88 — cumulative builds, early retirement)
        will give "installed capacity" a precise per-period meaning
        (``cap[t] = cap[t-1] + cap_new[t] - cap_retired[t]``). This stays the
        plainer "the size value resolved for each flow" so it doesn't collide
        with that future modeling term. The computation already keys off
        ``flow--size`` (the solver's per-period in-place capacity), so it
        remains correct when #88 lands; only the value, not the API, changes.

        Returns:
            DataArray (flow,) in power units (e.g. MW).
        """
        size = self._result.data.flows.size
        invested = self._result.sizes
        # `invested` is an empty 0-d DataArray when no flow is invested; only
        # merge when it actually carries a `flow` dim.
        if 'flow' in invested.dims:
            size = size.fillna(invested)
        return size

    @cached_property
    def total_duration(self) -> xr.DataArray:
        """Weighted length of the modeled horizon: Σ_t dt_t · w_t.

        The reference window for utilization metrics. Full load hours, if
        wanted, are ``capacity_factor * total_duration``.

        Returns:
            Scalar DataArray in hours; (period,) in multi-period models.
        """
        dims = self._result.data.dims
        return dims.sum_time(dims.dt * dims.weights)

    @cached_property
    def capacity_factor(self) -> xr.DataArray:
        """Mean utilization per flow: Σ(P·dt·w) / (size · Σ(dt·w)).

        Dimensionless in [0, 1] — energy delivered relative to running at full
        size over the whole weighted horizon. Unlike full load hours, the
        weighted duration cancels, so it is independent of horizon length and
        weight convention, and a per-period value needs no period weighting.
        NaN where the size is unknown (unsized flows) or zero.

        Returns:
            DataArray (flow[, period]) — fraction of rated capacity used.
        """
        with xr.set_options(keep_attrs=True):
            cf = self.total_flow_hours / (self.resolved_sizes * self.total_duration)
            return cf.where(lambda x: np.isfinite(x))

    @cached_property
    def resolved_capacities(self) -> xr.DataArray:
        """Per-storage energy capacity resolved from declared or optimized values.

        Storage analogue of :attr:`resolved_sizes` (see its note on the
        ``resolved_*`` naming vs #88's future "installed capacity"). Empty when
        the model has no storages.

        Returns:
            DataArray (storage,) in energy units (e.g. MWh).
        """
        if self._result.data.storages is None:
            return xr.DataArray()
        cap = self._result.data.storages.capacity
        invested = self._result.storage_capacities
        if 'storage' in invested.dims:
            cap = cap.fillna(invested)
        return cap

    @cached_property
    def relative_mean_level(self) -> xr.DataArray:
        """Mean fractional charge level per storage: ⟨E⟩ / capacity.

        Time-mean level (weighted by dt·w) over the reservoir capacity — the
        storage analogue of :attr:`capacity_factor`, and the running-average
        sibling of the ``relative_level_min`` / ``relative_level_max``
        bounds. Dimensionless in [0, 1] and horizon-independent. NaN where
        capacity is unknown or zero. Empty when the model has no storages.

        Returns:
            DataArray (storage[, period]) — mean fill fraction.
        """
        if self._result.data.storages is None:
            return xr.DataArray()
        dims = self._result.data.dims
        with xr.set_options(keep_attrs=True):
            mean_level = dims.sum_time(self._result.storage_levels * dims.dt * dims.weights) / self.total_duration
            rel = mean_level / self.resolved_capacities
            return rel.where(lambda x: np.isfinite(x))

    @cached_property
    def summary(self) -> xr.Dataset:
        """Headline KPIs as a named namespace.

        Composes existing accessors — :attr:`Result.objective`,
        :attr:`Result.effect_totals`, :attr:`total_duration`,
        :attr:`resolved_sizes`, :attr:`total_flow_hours`,
        :attr:`capacity_factor`, and (when the model has storages)
        :attr:`resolved_capacities` and :attr:`relative_mean_level` — into
        one Dataset. The variables live on different dimensions (``effect`` vs
        ``flow`` vs ``storage``), so access them by name: this is a KPI
        namespace, not a flat table, and ``to_dataframe()`` would broadcast
        the unrelated axes together.

        Returns:
            Dataset with ``objective``, ``effect_totals``, ``total_duration``,
            ``size``, ``total_flow_hours``, ``capacity_factor``, and — with
            storages — ``capacity`` and ``relative_mean_level``.
        """
        kpis = {
            'objective': self._result.objective,
            'effect_totals': self._result.effect_totals,
            'total_duration': self.total_duration,
            'size': self.resolved_sizes,
            'total_flow_hours': self.total_flow_hours,
            'capacity_factor': self.capacity_factor,
        }
        if self._result.data.storages is not None:
            kpis['capacity'] = self.resolved_capacities
            kpis['relative_mean_level'] = self.relative_mean_level
        return xr.Dataset(kpis)
