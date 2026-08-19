"""JSON Schema and dict/JSON round-trip for the element layer.

The user-facing elements are ``pydantic.dataclasses``, so pydantic can emit a
JSON Schema for each type and round-trip instances to plain dicts. This is the
machine-readable contract behind a future config front-end, GUI, or
LLM-assisted authoring surface.

Structural round-trip (:func:`to_dict` / :func:`from_dict`) preserves ids,
scalars, nested elements, and :class:`~fluxopt.types.ProfileRef` references.
Inline array-valued ``Variate`` fields (raw time-series) do *not* serialize —
use a ``ProfileRef`` instead, so profiles live in data files rather than in the
config (see ``docs/design/config-and-pydantic-direction.md``). Such fields also
appear as permissive ``{}`` in the JSON Schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from fluxopt.components import Converter, Port
from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage

if TYPE_CHECKING:
    from collections.abc import Mapping

ELEMENT_TYPES: tuple[type, ...] = (
    Carrier,
    Effect,
    Flow,
    Sizing,
    Investment,
    Status,
    Storage,
    Port,
    Converter,
)
"""Every user-facing element type, in declaration order."""


def element_schema(element_type: type) -> dict[str, Any]:
    """Return the JSON Schema for one element type.

    Args:
        element_type: An element dataclass (e.g. ``Flow``, ``Effect``).
    """
    return TypeAdapter(element_type).json_schema()


def all_element_schemas() -> Mapping[str, dict[str, Any]]:
    """Return JSON Schemas for every element type, keyed by class name."""
    return {t.__name__: element_schema(t) for t in ELEMENT_TYPES}


def _inline_array_paths(obj: Any, path: str, out: list[str]) -> None:
    """Collect paths of array-valued leaves that cannot serialize to JSON."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from pydantic import BaseModel

    if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame, xr.DataArray)):
        out.append(path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _inline_array_paths(value, f'{path}[{key!r}]', out)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _inline_array_paths(value, f'{path}[{i}]', out)
    elif isinstance(obj, BaseModel):
        element_id = getattr(obj, 'id', '') or getattr(obj, 'short_id', '')
        base = f'{type(obj).__name__}({element_id!r})' if element_id else path
        for name in type(obj).model_fields:
            _inline_array_paths(getattr(obj, name), f'{base}.{name}', out)


def to_dict(element: object) -> dict[str, Any]:
    """Serialize an element to a JSON-safe dict.

    Nested elements and ``ProfileRef`` references are
    included; inline array-valued ``Variate`` fields are not serializable —
    reference them with a ``ProfileRef`` instead.

    Args:
        element: Any element instance (e.g. ``Flow``, ``Converter``).

    Raises:
        ValueError: If the element holds inline array values, listing each
            offending field.
    """
    try:
        return TypeAdapter(type(element)).dump_python(element, mode='json')
    except Exception as exc:
        arrays: list[str] = []
        _inline_array_paths(element, type(element).__name__, arrays)
        if arrays:
            msg = (
                'cannot serialize inline array values at: '
                + '; '.join(arrays)
                + ' — reference them with ProfileRef(dataset=..., variable=...) and supply the data via profiles={...} at solve time'
            )
            raise ValueError(msg) from exc
        raise


def from_dict[T](element_type: type[T], data: Mapping[str, Any]) -> T:
    """Reconstruct an element from a dict produced by :func:`to_dict`.

    Args:
        element_type: The element class to build (e.g. ``Flow``).
        data: A mapping of field values.
    """
    return TypeAdapter(element_type).validate_python(data)
