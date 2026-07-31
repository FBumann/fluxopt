"""``FlowSystem`` — the declarative top of the element layer.

A ``FlowSystem`` is an inert, validated description of a flow system: the
same lists you would pass to :func:`fluxopt.optimize`, gathered into one object
that round-trips to dict/YAML. It carries *structure* (components, effects,
config) and ``ProfileRef`` references to time-series; the actual series are
supplied at solve time via ``profiles`` (``system.optimize(profiles=...)``) and
resolved into arrays just before the model is built.

The FlowSystem has no modeling behavior of its own — ``.optimize()`` runs the existing
pipeline (:meth:`ModelData.build` → :class:`FlowSystemModel`). Declaration (the system)
and use (building/solving) stay separate.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fluxopt.components import Converter, Port
from fluxopt.elements import Carrier, Effect, Storage
from fluxopt.model import FlowSystemModel
from fluxopt.model_data import ModelData
from fluxopt.schema import from_dict, to_dict
from fluxopt.types import IdList, ProfileRef, Timesteps
from fluxopt.validation import validate_system

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from fluxopt.results import Result

_PYDANTIC_CFG = ConfigDict(arbitrary_types_allowed=True)


def _resolve_refs(obj: Any, profiles: Mapping[str, Any]) -> Any:
    """Recursively replace every ``ProfileRef`` in *obj* with a resolved array.

    Walks element dataclasses, dicts, lists, and ``IdList`` containers,
    mutating in place. Non-container leaves (scalars, arrays) pass through.

    Args:
        obj: The value or element to walk.
        profiles: Mapping passed to :meth:`ProfileRef.resolve`.
    """
    if isinstance(obj, ProfileRef):
        return obj.resolve(profiles)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _resolve_refs(value, profiles)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _resolve_refs(value, profiles)
        return obj
    if isinstance(obj, IdList):
        for item in obj:
            _resolve_refs(item, profiles)
        return obj
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            setattr(obj, name, _resolve_refs(getattr(obj, name), profiles))
        return obj
    return obj


def _collect_profile_refs(obj: Any, path: str, out: list[tuple[str, ProfileRef]]) -> None:
    """Recursively collect every ``ProfileRef`` in *obj* with a readable path.

    Path segments name elements by class and id (``Flow('Demand(Heat)')``) and
    descend through fields, dict keys, and list positions.

    Args:
        obj: The value or element to walk.
        path: Path accumulated so far.
        out: Collected ``(path, ref)`` pairs, appended in walk order.
    """
    if isinstance(obj, ProfileRef):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect_profile_refs(value, f'{path}[{key!r}]', out)
    elif isinstance(obj, (list, IdList)):
        for i, value in enumerate(obj):
            _collect_profile_refs(value, f'{path}[{i}]', out)
    elif isinstance(obj, BaseModel):
        element_id = getattr(obj, 'id', '') or getattr(obj, 'short_id', '')
        base = f'{type(obj).__name__}({element_id!r})' if element_id else path
        for name in type(obj).model_fields:
            _collect_profile_refs(getattr(obj, name), f'{base}.{name}', out)


def _check_profiles_cover(refs: list[tuple[str, ProfileRef]], profiles: Mapping[str, Any]) -> None:
    """Raise one comprehensive error if any ref cannot be resolved.

    Args:
        refs: ``(path, ref)`` pairs from :func:`_collect_profile_refs`.
        profiles: The solve-time profile supply.

    Raises:
        KeyError: Listing every unresolvable ref with its element/field path.
    """
    missing = []
    for path, ref in refs:
        if ref.dataset not in profiles:
            missing.append(f'{path}: dataset {ref.dataset!r} not supplied (have {sorted(profiles)})')
        else:
            try:
                profiles[ref.dataset][ref.variable]
            except KeyError:
                missing.append(f'{path}: variable {ref.variable!r} not in dataset {ref.dataset!r}')
    if missing:
        raise KeyError('unresolvable ProfileRef(s):\n  ' + '\n  '.join(missing))


class FlowSystem(BaseModel):
    """A declarative flow-system description (see module docstring)."""

    model_config = _PYDANTIC_CFG

    timesteps: Timesteps | dict[int, Timesteps]
    """Time index for the optimization horizon."""
    carriers: list[Carrier]
    """Carrier declarations."""
    effects: list[Effect]
    """Effects to track (costs, emissions, …)."""
    ports: list[Port]
    """System boundary ports with imports/exports."""
    objective: str | dict[str, float]
    """Effect(s) to minimize — a name or ``{effect: weight}``. Must name at
    least one non-penalty effect."""
    converters: list[Converter] = Field(default_factory=list)
    """Linear/piecewise converters between carriers."""
    storages: list[Storage] = Field(default_factory=list)
    """Energy storages."""
    dt: float | list[float] | None = None
    """Timestep duration in hours. Auto-derived if None."""
    periods: list[int] | None = None
    """Integer period labels for multi-period optimization."""
    period_weights: list[float] | None = None
    """Explicit weights per period. Inferred from gaps if None."""

    @model_validator(mode='after')
    def _validate_references(self) -> FlowSystem:
        """Fail fast on undeclared references and duplicate ids (at construction/load)."""
        validate_system(
            carriers=self.carriers,
            effects=self.effects,
            ports=self.ports,
            converters=self.converters,
            storages=self.storages,
            objective=self.objective,
        )
        return self

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FlowSystem:
        """Build a system from a mapping (e.g. parsed YAML/JSON)."""
        return from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlowSystem:
        """Load a system from a YAML file.

        Args:
            path: Path to a YAML document describing the system.
        """
        import yaml

        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system to a JSON-safe dict (profiles as ``ProfileRef``)."""
        return to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write the system to a YAML file.

        Args:
            path: Destination path.
        """
        import yaml

        with open(path, 'w') as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def required_profiles(self) -> dict[str, set[str]]:
        """Enumerate the external data this system needs, as ``{dataset: variables}``.

        The contract a ``profiles`` supply must cover before
        :meth:`build_model` / :meth:`optimize` can run. Empty when every value
        is inline.
        """
        refs: list[tuple[str, ProfileRef]] = []
        for group in (self.carriers, self.effects, self.ports, self.converters, self.storages):
            _collect_profile_refs(group, '', refs)
        out: dict[str, set[str]] = {}
        for _, ref in refs:
            out.setdefault(ref.dataset, set()).add(ref.variable)
        return out

    def build_model(self, profiles: Mapping[str, Any] | None = None) -> FlowSystemModel:
        """Materialize an unbuilt solver model from this declaration.

        Resolves ``ProfileRef`` references (on a copy — the system stays
        reusable across different ``profiles``), builds the ``ModelData``, and
        returns a :class:`FlowSystemModel` carrying this system's
        :attr:`objective`. Call ``build()`` on the result to inspect the linopy
        model before solving, or ``optimize()`` to build and solve in one step.

        Args:
            profiles: Mapping from ``ProfileRef.dataset`` to a dataset (or mapping)
                holding the referenced variables. Required if the system uses
                any ``ProfileRef`` — see :meth:`required_profiles`.

        Raises:
            KeyError: If any ``ProfileRef`` cannot be resolved from *profiles*;
                lists every unresolvable ref with its element/field path.
        """
        refs: list[tuple[str, ProfileRef]] = []
        for group in (self.carriers, self.effects, self.ports, self.converters, self.storages):
            _collect_profile_refs(group, '', refs)
        _check_profiles_cover(refs, profiles or {})

        carriers, effects, ports, converters, storages = (
            # No refs → nothing to substitute, so no copy or walk needed.
            (self.carriers, self.effects, self.ports, self.converters, self.storages)
            if not refs
            else copy.deepcopy((self.carriers, self.effects, self.ports, self.converters, self.storages))
        )
        if refs:
            for group in (carriers, effects, ports, converters, storages):
                _resolve_refs(group, profiles or {})

        data = ModelData.build(
            self.timesteps,
            carriers,
            effects,
            ports,
            converters,
            storages,
            self.dt,
            periods=self.periods,
            period_weights=self.period_weights,
        )
        return FlowSystemModel(data, objective=self.objective)

    def optimize(
        self,
        profiles: Mapping[str, Any] | None = None,
        *,
        solver: str = 'highs',
        customize: Callable[[FlowSystemModel], None] | None = None,
        **kwargs: Any,
    ) -> Result:
        """Resolve profile references, build the model, and solve.

        Shorthand for ``build_model(profiles).optimize(...)``.

        Args:
            profiles: Mapping from ``ProfileRef.dataset`` to a dataset (or mapping)
                holding the referenced variables. Required if the system uses
                any ``ProfileRef``.
            solver: Solver backend name.
            customize: Callback to modify the linopy model between build and
                solve; receives the built ``FlowSystemModel`` (use ``model.m``).
            **kwargs: Passed through to ``linopy.Model.solve()``.
        """
        return self.build_model(profiles).optimize(customize=customize, solver=solver, **kwargs)
