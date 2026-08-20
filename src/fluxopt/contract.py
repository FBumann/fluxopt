"""The cross-layer data contract.

Names shared between the layers — ``model_data`` (writes them),
``model`` (builds variables/constraints from them), and ``results`` /
``contributions`` / ``stats`` (read the solution) — live here so a rename
is a one-line change instead of a silent cross-file drift.

Core dimensions
    The entity dimensions ``flow``, ``carrier``, ``converter``, ``storage``,
    ``effect``, ``source_effect``, ``time``, ``period`` and the converter
    equation axis ``eq_idx`` are stable vocabulary used as plain literals
    throughout — ubiquitous enough that constants would hurt readability.
    The *regime* dimensions below index optional feature tables over a
    subset of an entity dimension and are renamed back to the entity
    dimension at constraint time.

Sentinel conventions
    - ``NaN`` in a parameter array means "not set": an unsized flow's
      ``size``, unbounded aggregate limits (``flow_hours_*``,
      ``load_factor_*``, effect ``total_*`` / ``periodic_*``), a free
      storage ``prior_level`` / final level, an ``Investment`` lifetime
      (forever), unknown prior status durations, and a carrier
      ``flow_coeff`` entry for a flow not on that carrier.
    - ``''`` (empty string) means "not set" for color fields and for
      ``governed_by``, where it means no component's Status governs the flow.
    - ``None`` at container level means the feature is absent entirely
      (the converters/storages/piecewise tables, the sizing / invest /
      status sub-containers, and all-NaN optional per-flow bounds).
    - ``bound_type`` is the one explicit sentinel: :class:`BoundType`.

Solution variables
    linopy variable names follow ``<family>--<field>`` with family one of
    ``flow`` / ``component`` / ``storage`` / ``invest`` / ``effect``.
    The same names key ``Result.solution``.
"""

from __future__ import annotations

from enum import StrEnum


class BoundType(StrEnum):
    """How a flow's rate envelope is bounded — the model layer's dispatch key."""

    UNSIZED = 'unsized'
    """No size: rate is only bounded below by 0 (and above by nothing)."""
    BOUNDED = 'bounded'
    """Rate in ``[rel_lb, rel_ub] * size``."""
    PROFILE = 'profile'
    """Rate fixed to ``fixed_profile * size``."""


class Var:
    """Solution variable names, ``<family>--<field>`` (also keys of ``Result.solution``)."""

    FLOW_RATE = 'flow--rate'
    FLOW_SIZE = 'flow--size'
    FLOW_SIZE_INDICATOR = 'flow--size_indicator'
    FLOW_ON = 'flow--on'
    FLOW_STARTUP = 'flow--startup'
    FLOW_SHUTDOWN = 'flow--shutdown'
    COMPONENT_ON = 'component--on'
    COMPONENT_STARTUP = 'component--startup'
    COMPONENT_SHUTDOWN = 'component--shutdown'
    STORAGE_LEVEL = 'storage--level'
    STORAGE_CHARGING = 'storage--charging'
    STORAGE_CAPACITY = 'storage--capacity'
    STORAGE_SIZE_INDICATOR = 'storage--size_indicator'
    INVEST_SIZE = 'invest--size'
    INVEST_SIZE_AT_BUILD = 'invest--size_at_build'
    INVEST_BUILD = 'invest--build'
    INVEST_ACTIVE = 'invest--active'
    EFFECT_TOTAL = 'effect--total'
    EFFECT_LUMP = 'effect--lump'
