from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Storage


def validate_system(
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
    flows: list[Flow],
) -> None:
    _check_id_uniqueness(buses, effects, ports, converters, storages or [])
    _check_bus_references(buses, flows)
    _check_objective(effects)
    _check_flow_uniqueness(flows)


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
