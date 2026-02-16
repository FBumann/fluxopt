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
    """Frozen, ordered container with access by ``id`` (str) or position (int).

    Raises :class:`ValueError` on duplicate ids at construction time.
    Supports concatenation via ``+`` (returns a new ``IdList``).
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


def to_data_array(value: TimeSeries, time: pd.Index, name: str = 'value') -> xr.DataArray:
    """Convert any TimeSeries input to an xr.DataArray aligned with time.

    Scalar -> broadcast, list -> length-check, Series -> align.
    """
    n = len(time)
    if isinstance(value, xr.DataArray):
        if len(value) != n:
            raise ValueError(f'DataArray length {len(value)} does not match timesteps length {n}')
        return value.rename(name)
    if isinstance(value, (int, float)):
        return xr.DataArray(np.full(n, float(value)), dims=['time'], coords={'time': time}, name=name)
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f'List length {len(value)} does not match timesteps length {n}')
        return xr.DataArray([float(v) for v in value], dims=['time'], coords={'time': time}, name=name)
    if isinstance(value, np.ndarray):
        if len(value) != n:
            raise ValueError(f'Array length {len(value)} does not match timesteps length {n}')
        return xr.DataArray(value.astype(float), dims=['time'], coords={'time': time}, name=name)
    if isinstance(value, pd.Series):
        if len(value) != n:
            raise ValueError(f'Series length {len(value)} does not match timesteps length {n}')
        return xr.DataArray(value.values.astype(float), dims=['time'], coords={'time': time}, name=name)
    raise TypeError(f'Unsupported TimeSeries type: {type(value)}')


def as_dataarray(
    value: TimeSeries,
    coords: Mapping[str, Any],
    *,
    name: str = 'value',
    broadcast: bool = False,
) -> xr.DataArray:
    """Convert a TimeSeries to a DataArray aligned to the given coordinates.

    Parameters
    ----------
    value
        Scalar, list, ndarray, Series, or DataArray.
    coords
        Target coordinates, e.g. ``{"time": idx}`` or ``{"flow": ids, "time": idx}``.
    name
        Name for the resulting DataArray.
    broadcast
        If True, expand the result to span *all* dimensions in *coords*.
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

    Supports datetime and integer timesteps. Strings are not supported.
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
    """Compute dt (hours) for each timestep as an xr.DataArray.

    - dt=None + datetime: derive from consecutive differences in hours; last = second-to-last.
    - dt=None + integer: broadcast 1.0.
    - dt provided: validate length, return as DataArray.
    - Single timestep: default to 1.0.
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
