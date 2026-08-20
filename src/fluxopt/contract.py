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

Absence
    Not-set is a missing row, throughout. An unsized flow has no row in
    ``flows.sizes``, an unbounded aggregate none in ``flows.aggregates``, an
    ``Investment`` that never expires none in ``invest.lifetime``. There are
    no sentinel values to test for and nothing to spell "not set" as.

    ``None`` survives at container level, where it means the feature is
    absent from the whole system: the converters / storages / piecewise /
    status tables, and the sizing and invest sub-containers.

Solution variables
    linopy variable names follow ``<family>--<field>`` with family one of
    ``flow`` / ``component`` / ``storage`` / ``invest`` / ``effect``.
    The same names key ``Result.solution``.
"""

from __future__ import annotations


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
