"""The single declaration of every effect contribution in the model.

An :class:`EffectTerm` says: *this solver variable, weighted by this
coefficient array, contributes to the effects* — in the temporal domain
(per timestep, optionally scaled by Δt) or the lump domain (sizing and
investment costs). :func:`effect_terms` enumerates the terms for a given
``ModelData``.

Two consumers derive their math from this one list:

- ``model._create_effects`` builds the linopy expressions that accumulate
  into ``effect--total`` / ``effect--lump``;
- ``contributions._compute_direct`` multiplies the same coefficients with
  the *solved* variable values to decompose totals per contributor.

Keeping both sides on one declaration is what guarantees the post-solve
decomposition reconstructs the solver totals (checked at runtime by
``contributions._validate_against_solver``).

Terms with an all-zero coefficient are omitted, so consumers can treat
"no terms" as "no contribution" without re-checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from fluxopt.contract import Contribution, Dim, Var

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData, SizingData


@dataclass(frozen=True)
class EffectTerm:
    """One contribution of a solver variable (or constant) to the effects."""

    key: str
    """Stable identifier of the term (e.g. ``Contribution.FLOW_HOUR``, ``Contribution.INVEST_FIXED_AT_BUILD``)."""
    domain: Literal['temporal', 'lump']
    """Temporal terms accumulate per timestep; lump terms are one-shot."""
    entity_dim: str
    """Contributor dimension of ``coeff``: ``'flow'``, ``'component'``, or ``'storage'``."""
    coeff: xr.DataArray
    """Coefficient array ``(entity, effect[, time, period])``, entity dim already renamed."""
    var: str | None
    """Solution variable name (:class:`~fluxopt.contract.Var`); None for a constant term."""
    select: tuple[str, ...] | None = None
    """Entity ids to select from the variable before multiplying (the ids ``coeff`` covers)."""
    scale_dt: bool = False
    """Temporal only: multiply by the timestep duration Δt."""
    sparse: bool = False
    """Model-side hint: use the sparse weighted-sum kernel (dense flow coefficients)."""


def effect_terms(data: ModelData) -> list[EffectTerm]:
    """Enumerate every effect contribution declared by *data*.

    Args:
        data: The model data to read coefficient arrays from.
    """
    terms: list[EffectTerm] = []

    def add(
        key: str,
        domain: Literal['temporal', 'lump'],
        entity_dim: str,
        coeff: xr.DataArray,
        var: str | None,
        select: tuple[str, ...] | None = None,
        scale_dt: bool = False,
        sparse: bool = False,
    ) -> None:
        if bool((coeff != 0).any()):
            terms.append(EffectTerm(key, domain, entity_dim, coeff, var, select, scale_dt, sparse))

    fds = data.flows

    # --- Temporal domain -------------------------------------------------
    add(Contribution.FLOW_HOUR, 'temporal', 'flow', fds.effect_coeff, Var.FLOW_RATE, scale_dt=True, sparse=True)

    if fds.status is not None:
        ids = tuple(map(str, fds.status.uptime_min.coords[Dim.STATUS_FLOW].values))
        rename = {Dim.STATUS_FLOW: 'flow'}
        add(
            Contribution.STATUS_RUNNING,
            'temporal',
            'flow',
            fds.status.effects_running.rename(rename),
            Var.FLOW_ON,
            ids,
            scale_dt=True,
        )
        add(
            Contribution.STATUS_STARTUP,
            'temporal',
            'flow',
            fds.status.effects_startup.rename(rename),
            Var.FLOW_STARTUP,
            ids,
        )

    if fds.cstatus is not None:
        cids = tuple(map(str, fds.cstatus.uptime_min.coords[Dim.CSTATUS_COMPONENT].values))
        rename = {Dim.CSTATUS_COMPONENT: 'component'}
        add(
            Contribution.COMPONENT_RUNNING,
            'temporal',
            'component',
            fds.cstatus.effects_running.rename(rename),
            Var.COMPONENT_ON,
            cids,
            scale_dt=True,
        )
        add(
            Contribution.COMPONENT_STARTUP,
            'temporal',
            'component',
            fds.cstatus.effects_startup.rename(rename),
            Var.COMPONENT_STARTUP,
            cids,
        )

    # --- Lump domain: sizing (flows and storages) ------------------------
    def sizing_terms(
        sizing: SizingData | None,
        sdim: str,
        entity: str,
        size_var: str,
        indicator_var: str,
        keys: tuple[str, str, str],
    ) -> None:
        if sizing is None:
            return
        per_size_key, fixed_optional_key, fixed_mandatory_key = keys
        ids = np.asarray(sizing.min.coords[sdim].values, dtype=str)
        rename = {sdim: entity}
        add(per_size_key, 'lump', entity, sizing.effects_per_size.rename(rename), size_var, tuple(ids))
        mandatory = sizing.mandatory.values
        if optional_ids := tuple(ids[~mandatory]):
            coeff = sizing.effects_fixed.sel({sdim: list(optional_ids)}).rename(rename)
            add(fixed_optional_key, 'lump', entity, coeff, indicator_var, optional_ids)
        if mandatory_ids := tuple(ids[mandatory]):
            coeff = sizing.effects_fixed.sel({sdim: list(mandatory_ids)}).rename(rename)
            add(fixed_mandatory_key, 'lump', entity, coeff, None)

    sizing_terms(
        fds.sizing,
        Dim.SIZING_FLOW,
        'flow',
        Var.FLOW_SIZE,
        Var.FLOW_SIZE_INDICATOR,
        (
            Contribution.FLOW_SIZING_PER_SIZE,
            Contribution.FLOW_SIZING_FIXED_OPTIONAL,
            Contribution.FLOW_SIZING_FIXED_MANDATORY,
        ),
    )
    if data.storages is not None:
        sizing_terms(
            data.storages.sizing,
            Dim.SIZING_STORAGE,
            'storage',
            Var.STORAGE_CAPACITY,
            Var.STORAGE_SIZE_INDICATOR,
            (
                Contribution.STORAGE_SIZING_PER_SIZE,
                Contribution.STORAGE_SIZING_FIXED_OPTIONAL,
                Contribution.STORAGE_SIZING_FIXED_MANDATORY,
            ),
        )

    # --- Lump domain: investment (flows) ---------------------------------
    if fds.invest is not None:
        inv = fds.invest
        inv_ids = tuple(map(str, inv.min.coords[Dim.INVEST_FLOW].values))
        rename = {Dim.INVEST_FLOW: 'flow'}
        add(
            Contribution.INVEST_PER_SIZE_AT_BUILD,
            'lump',
            'flow',
            inv.effects_per_size_at_build.rename(rename),
            Var.INVEST_SIZE_AT_BUILD,
            inv_ids,
        )
        add(
            Contribution.INVEST_FIXED_AT_BUILD,
            'lump',
            'flow',
            inv.effects_fixed_at_build.rename(rename),
            Var.INVEST_BUILD,
            inv_ids,
        )
        add(
            Contribution.INVEST_PER_SIZE_RECURRING,
            'lump',
            'flow',
            inv.effects_per_size_recurring.rename(rename),
            Var.FLOW_SIZE,
            inv_ids,
        )
        add(
            Contribution.INVEST_FIXED_RECURRING,
            'lump',
            'flow',
            inv.effects_fixed_recurring.rename(rename),
            Var.INVEST_ACTIVE,
            inv_ids,
        )

    return terms


def _lump_bearing_effects(terms: list[EffectTerm], cf_lump: xr.DataArray) -> xr.DataArray:
    """Boolean mask over ``effect``: which effects receive lump contributions.

    An effect is lump-bearing when a lump-domain term contributes to it
    directly, or when it receives from a lump-bearing effect through the
    (acyclic) cross-effect matrix ``(effect, source_effect[, ...])``.
    """
    effect_ids = cf_lump.indexes['effect']
    bearing = xr.DataArray(np.zeros(len(effect_ids), dtype=bool), coords={'effect': effect_ids}, dims='effect')
    for term in (t for t in terms if t.domain == 'lump'):
        nonzero = (term.coeff.notnull() & (term.coeff != 0)).any([d for d in term.coeff.dims if d != 'effect'])
        bearing = bearing | nonzero.reindex_like(bearing, fill_value=False)

    adjacency = (cf_lump.notnull() & (cf_lump != 0)).any(
        [d for d in cf_lump.dims if d not in ('effect', 'source_effect')]
    )
    for _ in range(bearing.sizes['effect']):
        grown = bearing | (adjacency & bearing.rename({'effect': 'source_effect'})).any('source_effect')
        if grown.equals(bearing):
            break
        bearing = grown
    return bearing
