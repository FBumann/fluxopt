from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.components import LinearConverter, Port
    from fluxopt.elements import Bus, Effect, Flow, Storage


def validate_system(
    buses: list[Bus],
    effects: list[Effect],
    components: list[Port | LinearConverter],
    storages: list[Storage] | None,
    flows: list[Flow],
) -> None:
    _check_label_uniqueness(buses, effects, components, storages or [])
    _check_bus_references(buses, flows)
    _check_objective(effects)


def _check_label_uniqueness(
    buses: list[Bus],
    effects: list[Effect],
    components: list[Port | LinearConverter],
    storages: list[Storage],
) -> None:
    all_labels = [bus.label for bus in buses]
    all_labels.extend(effect.label for effect in effects)
    all_labels.extend(comp.label for comp in components)
    all_labels.extend(stor.label for stor in storages)

    seen: set[str] = set()
    for lbl in all_labels:
        if lbl in seen:
            raise ValueError(f'Duplicate label: {lbl!r}')
        seen.add(lbl)


def _check_bus_references(buses: list[Bus], flows: list[Flow]) -> None:
    bus_labels = {b.label for b in buses}
    for flow in flows:
        if flow.bus not in bus_labels:
            raise ValueError(f'Flow {flow.label!r} references unknown bus {flow.bus!r}')


def _check_objective(effects: list[Effect]) -> None:
    objective_effects = [e for e in effects if e.is_objective]
    if len(objective_effects) == 0:
        raise ValueError('At least one Effect must have is_objective=True')
    if len(objective_effects) > 1:
        labels = [e.label for e in objective_effects]
        raise ValueError(f'Multiple objective effects: {labels}. Only one is allowed.')
