from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

type TimeSeries = float | int | list[float] | np.ndarray | pd.Series | xr.DataArray
type Timesteps = list[datetime] | list[int] | pd.DatetimeIndex | np.ndarray


@runtime_checkable
class Identified(Protocol):
    @property
    def id(self) -> str: ...


class IdList[T: Identified]:
    """Frozen, ordered container with access by id (str) or position (int).

    Supports concatenation via ``+``.

    Args:
        items: Elements to store. Must have unique ids.

    Raises:
        ValueError: On duplicate ids.
    """

    __slots__ = ('_by_id', '_items')

    def __init__(self, items: Iterable[T]) -> None:
        self._items: tuple[T, ...] = tuple(items)
        self._by_id: dict[str, T] = {}
        for item in self._items:
            if item.id in self._by_id:
                raise ValueError(f"Duplicate id: '{item.id}'")
            self._by_id[item.id] = item

    @overload
    def __getitem__(self, key: str) -> T: ...
    @overload
    def __getitem__(self, key: int) -> T: ...
    def __getitem__(self, key: str | int) -> T:
        if isinstance(key, str):
            return self._by_id[key]
        return self._items[key]

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self._by_id
        return key in self._items

    def __add__(self, other: IdList[T]) -> IdList[T]:
        return IdList([*self._items, *other._items])

    def __repr__(self) -> str:
        return f'IdList({list(self._items)!r})'


def fast_concat(arrays: list[xr.DataArray], dim: pd.Index) -> xr.DataArray:
    """Stack DataArrays along a new leading dimension.

    Drop-in replacement for ``xr.concat`` when all slices already share the
    same dims, shape, and coords. Skips alignment, deepcopy, and reindex —
    just stacks the underlying numpy arrays.

    Args:
        arrays: DataArrays with identical dims, shape, and coords.
        dim: Index for the new leading dimension.

    Raises:
        ValueError: If any slice has a different shape or dims than the first.
    """
    first = arrays[0]
    expected_shape = first.shape
    expected_dims = first.dims
    for i, a in enumerate(arrays[1:], 1):
        if a.shape != expected_shape:
            raise ValueError(f'fast_concat: slice {i} shape {a.shape} != expected {expected_shape}')
        if a.dims != expected_dims:
            raise ValueError(f'fast_concat: slice {i} dims {a.dims} != expected {expected_dims}')
    data = np.array([a.values for a in arrays])
    name = str(dim.name)
    dims = [name, *expected_dims]
    coords: dict[str, object] = {name: dim}
    for d in expected_dims:
        key = str(d)
        if key in first.coords:
            coords[key] = first.coords[key]
    return xr.DataArray(data, dims=dims, coords=coords)


def as_dataarray(
    value: TimeSeries,
    coords: Mapping[str, Any],
    *,
    name: str = 'value',
    broadcast: bool = True,
) -> xr.DataArray:
    """Convert a TimeSeries to a DataArray aligned to given coordinates.

    Args:
        value: Scalar, list, ndarray, Series, or DataArray.
        coords: Target coordinates, e.g. ``{"time": idx}``.
        name: Name for the resulting DataArray.
        broadcast: Expand result to span all dimensions in coords.
    """
    coord_idx = {k: v if isinstance(v, pd.Index) else pd.Index(v) for k, v in coords.items()}

    # --- scalar: 0-dim unless broadcast ---
    if isinstance(value, (int, float)):
        if not broadcast:
            return xr.DataArray(float(value), name=name)
        shape = tuple(len(v) for v in coord_idx.values())
        return xr.DataArray(
            np.full(shape, float(value)),
            dims=list(coord_idx),
            coords=coord_idx,
            name=name,
        )

    # --- existing DataArray: align + optionally broadcast ---
    if isinstance(value, xr.DataArray):
        foreign = [str(d) for d in value.dims if d not in coord_idx]
        if foreign:
            raise ValueError(
                f'DataArray has dims {foreign} not in target coords {list(coord_idx)}. '
                f'Rename before calling as_dataarray().'
            )
        da = value.rename(name)
        if broadcast:
            for dim, idx in coord_idx.items():
                if dim not in da.dims:
                    da = da.expand_dims({dim: idx})
            da = da.transpose(*coord_idx)
        return da

    # --- 1D: list / ndarray / Series ---
    if isinstance(value, pd.Series):
        arr = value.values.astype(float)
    elif isinstance(value, np.ndarray):
        arr = value.astype(float)
    elif isinstance(value, list):
        arr = np.array(value, dtype=float)
    else:
        raise TypeError(f'Unsupported TimeSeries type: {type(value)}')

    n = len(arr)
    matches = [k for k, v in coord_idx.items() if len(v) == n]
    if len(matches) == 0:
        lengths = ', '.join(f'{k}({len(v)})' for k, v in coord_idx.items())
        raise ValueError(f'Length {n} does not match any coordinate: {lengths}')
    if len(matches) > 1:
        raise ValueError(
            f'Length {n} matches multiple coordinates: {matches}. '
            f'Pass an xr.DataArray with explicit dims to disambiguate.'
        )
    dim = matches[0]
    da = xr.DataArray(arr, dims=[dim], coords={dim: coord_idx[dim]}, name=name)
    if broadcast:
        for d, idx in coord_idx.items():
            if d not in da.dims:
                da = da.expand_dims({d: idx})
        da = da.transpose(*coord_idx)
    return da


def normalize_timesteps(timesteps: Timesteps) -> pd.Index:
    """Convert any Timesteps input to a pd.Index.

    Args:
        timesteps: Datetime or integer timesteps.
    """
    if isinstance(timesteps, pd.DatetimeIndex):
        return timesteps

    if isinstance(timesteps, np.ndarray):
        if np.issubdtype(timesteps.dtype, np.datetime64):
            return pd.DatetimeIndex(timesteps)
        return pd.Index(timesteps)

    # list[datetime] or list[int]
    if not isinstance(timesteps, list):
        raise TypeError(f'Unsupported Timesteps type: {type(timesteps)}')

    if len(timesteps) == 0:
        return pd.DatetimeIndex([])

    if isinstance(timesteps[0], datetime):
        return pd.DatetimeIndex(timesteps)
    if isinstance(timesteps[0], int):
        return pd.Index(timesteps, dtype=np.int64)
    raise TypeError(f'Unsupported timestep element type: {type(timesteps[0])}. Use datetime or int.')


def compute_dt(timesteps: pd.Index, dt: float | list[float] | None) -> xr.DataArray:
    """Compute dt (hours) for each timestep as a DataArray.

    When dt is None, auto-derives from timesteps:
    - Datetime: consecutive differences in hours; first = second (forward-looking).
    - Integer: 1.0 for all.
    - Single timestep: 1.0.

    Args:
        timesteps: Time index.
        dt: Override timestep duration. Validated against timesteps length.
    """
    n = len(timesteps)

    if dt is not None:
        if isinstance(dt, (int, float)):
            values = np.full(n, float(dt))
        elif isinstance(dt, list):
            if len(dt) != n:
                raise ValueError(f'dt length {len(dt)} does not match timesteps length {n}')
            values = np.array([float(v) for v in dt])
        else:
            raise TypeError(f'Unsupported dt type: {type(dt)}')
        return xr.DataArray(values, dims=['time'], coords={'time': timesteps}, name='dt')

    # Auto-derive
    if n <= 1:
        return xr.DataArray(np.ones(n), dims=['time'], coords={'time': timesteps}, name='dt')

    if not isinstance(timesteps, pd.DatetimeIndex):
        # Integer timesteps: default to 1.0
        return xr.DataArray(np.ones(n), dims=['time'], coords={'time': timesteps}, name='dt')

    # Datetime: derive from diff in hours
    diffs = np.diff(timesteps.values) / np.timedelta64(1, 'h')
    dt_values = np.empty(n)
    dt_values[0] = diffs[0]
    dt_values[1:] = diffs
    return xr.DataArray(dt_values, dims=['time'], coords={'time': timesteps}, name='dt')
