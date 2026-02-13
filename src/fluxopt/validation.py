from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Storage


# -- System-level checks (cross-cutting) ------------------------------------


def validate_system(
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter] | None,
    storages: list[Storage] | None,
    flows: list[Flow],
) -> None:
    _check_id_uniqueness(buses, effects, ports, converters or [], storages or [])
    _check_bus_references(buses, flows)
    _check_objective(effects)
    _check_flow_uniqueness(flows)
    _check_sizing_effect_references(effects, flows, storages or [])


def _check_id_uniqueness(
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage],
) -> None:
    all_ids = [bus.id for bus in buses]
    all_ids.extend(effect.id for effect in effects)
    all_ids.extend(port.id for port in ports)
    all_ids.extend(conv.id for conv in converters)
    all_ids.extend(stor.id for stor in storages)

    seen: set[str] = set()
    for id_ in all_ids:
        if id_ in seen:
            raise ValueError(f'Duplicate id: {id_!r}')
        seen.add(id_)


def _check_bus_references(buses: list[Bus], flows: list[Flow]) -> None:
    bus_ids = {b.id for b in buses}
    for flow in flows:
        if flow.bus not in bus_ids:
            raise ValueError(f'Flow {flow.id!r} references unknown bus {flow.bus!r}')


def _check_objective(effects: list[Effect]) -> None:
    objective_effects = [e for e in effects if e.is_objective]
    if len(objective_effects) == 0:
        raise ValueError('At least one Effect must have is_objective=True')
    if len(objective_effects) > 1:
        ids = [e.id for e in objective_effects]
        raise ValueError(f'Multiple objective effects: {ids}. Only one is allowed.')


def _check_flow_uniqueness(flows: list[Flow]) -> None:
    seen: set[str] = set()
    for flow in flows:
        if flow.id in seen:
            raise ValueError(f'Duplicate flow id: {flow.id!r}')
        seen.add(flow.id)


# -- Table-level validation (called by *Table.from_elements) ----------------


def validate_flow_bounds(relative_bounds: pl.DataFrame) -> None:
    """Validate (flow, time, rel_lb, rel_ub) DataFrame."""
    bad_lb = relative_bounds.filter(pl.col('rel_lb') < 0)
    if len(bad_lb) > 0:
        flows = bad_lb['flow'].unique().sort().to_list()
        raise ValueError(f'Negative lower bounds on flows: {flows}')

    bad_order = relative_bounds.filter(pl.col('rel_lb') > pl.col('rel_ub'))
    if len(bad_order) > 0:
        flows = bad_order['flow'].unique().sort().to_list()
        raise ValueError(f'Lower bound > upper bound on flows: {flows}')


def validate_storage_params(params: pl.DataFrame) -> None:
    """Validate (storage, capacity, initial_charge, cyclic) DataFrame."""
    bad_cap = params.filter(pl.col('capacity').is_not_null() & (pl.col('capacity') < 0))
    if len(bad_cap) > 0:
        ids = bad_cap['storage'].to_list()
        raise ValueError(f'Negative capacity on storages: {ids}')


def validate_storage_time_params(time_params: pl.DataFrame) -> None:
    """Validate (storage, time, eta_c, eta_d, loss) DataFrame."""
    bad_eta_c = time_params.filter((pl.col('eta_c') <= 0) | (pl.col('eta_c') > 1))
    if len(bad_eta_c) > 0:
        ids = bad_eta_c['storage'].unique().sort().to_list()
        raise ValueError(f'eta_charge must be in (0, 1] on storages: {ids}')

    bad_eta_d = time_params.filter((pl.col('eta_d') <= 0) | (pl.col('eta_d') > 1))
    if len(bad_eta_d) > 0:
        ids = bad_eta_d['storage'].unique().sort().to_list()
        raise ValueError(f'eta_discharge must be in (0, 1] on storages: {ids}')

    bad_loss = time_params.filter(pl.col('loss') < 0)
    if len(bad_loss) > 0:
        ids = bad_loss['storage'].unique().sort().to_list()
        raise ValueError(f'Negative relative_loss_per_hour on storages: {ids}')


def validate_sizing_params(df: pl.DataFrame, *, entity_col: str) -> None:
    """Validate sizing_params DataFrame: (entity, min_size, max_size, mandatory)."""
    bad_min = df.filter(pl.col('min_size') < 0)
    if len(bad_min) > 0:
        ids = bad_min[entity_col].to_list()
        raise ValueError(f'Negative min_size on {entity_col}s: {ids}')

    bad_max = df.filter(pl.col('max_size') <= 0)
    if len(bad_max) > 0:
        ids = bad_max[entity_col].to_list()
        raise ValueError(f'max_size must be > 0 on {entity_col}s: {ids}')

    bad_order = df.filter(pl.col('min_size') > pl.col('max_size'))
    if len(bad_order) > 0:
        ids = bad_order[entity_col].to_list()
        raise ValueError(f'min_size > max_size on {entity_col}s: {ids}')


def _check_sizing_effect_references(effects: list[Effect], flows: list[Flow], storages: list[Storage]) -> None:
    """Check that effect ids referenced in Sizing dicts exist in the effects list."""
    from fluxopt.elements import Sizing

    effect_ids = {e.id for e in effects}
    for flow in flows:
        if isinstance(flow.size, Sizing):
            for eid in (*flow.size.effects_per_size, *flow.size.effects_of_size):
                if eid not in effect_ids:
                    raise ValueError(f'Flow {flow.id!r} sizing references unknown effect {eid!r}')
    for stor in storages:
        if isinstance(stor.capacity, Sizing):
            for eid in (*stor.capacity.effects_per_size, *stor.capacity.effects_of_size):
                if eid not in effect_ids:
                    raise ValueError(f'Storage {stor.id!r} sizing references unknown effect {eid!r}')
