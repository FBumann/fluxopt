from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, overload, override, runtime_checkable

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, GetCoreSchemaHandler
from pydantic_core import core_schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


class ProfileRef(BaseModel):
    """Reference to a time-series stored outside the model definition.

    A serializable stand-in for an inline ``Variate`` array: the profile lives
    in a data file / dataset and is named here, so structural definitions
    round-trip to YAML/JSON without inlining 8760-point series. Resolve it to a
    :class:`xr.DataArray` with :meth:`resolve` before building the model.
    """

    model_config = ConfigDict(frozen=True)

    dataset: str
    """Id of the dataset holding the profile (a key into ``profiles``)."""
    variable: str
    """Variable / column name within *dataset*."""

    def resolve(self, profiles: Mapping[str, xr.Dataset | Mapping[str, xr.DataArray]]) -> xr.DataArray:
        """Look up the referenced series in *profiles*.

        Args:
            profiles: Mapping from dataset id to a dataset (or mapping) that
                contains ``variable``.

        Raises:
            KeyError: If *dataset* or *variable* is absent from *profiles*.
        """
        if self.dataset not in profiles:
            raise KeyError(f'ProfileRef dataset {self.dataset!r} not in profiles {sorted(profiles)}')
        ds = profiles[self.dataset]
        try:
            return xr.DataArray(ds[self.variable])
        except KeyError as exc:
            raise KeyError(f'ProfileRef variable {self.variable!r} not in dataset {self.dataset!r}') from exc


# -- User input types --------------------------------------------------
type PeriodMap = dict[int, float | list[float] | np.ndarray | pd.Series | xr.DataArray]
"""Per-period operational input: one value or within-period series per period
label. Aligned onto the flat time axis by ``_TimeMapper`` (ragged-safe)."""

type Variate = float | int | list[float] | np.ndarray | pd.Series | pd.DataFrame | xr.DataArray | ProfileRef | PeriodMap
"""Any input that varies over a subset of the model's variate dims (``time``,
optionally ``period``, eventually ``scenario``).

- Scalar: broadcast to all variate dims.
- 1-D (``list``/``ndarray``): matched to a coord by length (must be unambiguous).
- 1-D (``pd.Series``): index name selects the dim if set; else matched by length.
- 2-D (``pd.DataFrame``): ``index.name`` and ``columns.name`` must match target dims.
- n-D (``xr.DataArray``): dims must be a subset of the target; coords must match exactly.
- Mapping (``{period: value-or-series}``): per-period values on the flat time axis.

Per-field reach (which dims a particular field can vary over) is documented on
the field itself; ``as_dataarray`` enforces that user input only uses dims the
caller declared in *coords*.
"""

type Timesteps = list[datetime] | list[int] | pd.DatetimeIndex | pd.Index

# -- Internal types (after normalization) ------------------------------
type TimeIndex = pd.DatetimeIndex | pd.Index

# -- Piecewise formulation method (mirrors linopy.add_piecewise_formulation) --
type PiecewiseMethod = Literal['auto', 'sos2', 'incremental', 'lp']


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

    @override
    def __repr__(self) -> str:
        return f'IdList({list(self._items)!r})'

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Validate from a list (or existing IdList) and serialize to a list."""
        args = get_args(source)
        item_schema = handler.generate_schema(args[0]) if args else core_schema.any_schema()
        list_schema = core_schema.list_schema(item_schema)

        def _coerce(value: object) -> IdList[Any]:
            return value if isinstance(value, IdList) else IdList(value)  # type: ignore[arg-type]

        return core_schema.no_info_after_validator_function(
            _coerce,
            core_schema.union_schema([core_schema.is_instance_schema(IdList), list_schema]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                list, return_schema=list_schema, when_used='always'
            ),
        )


def fast_concat(arrays: list[xr.DataArray], dim: pd.Index) -> xr.DataArray:
    """Stack DataArrays along a new leading dimension.

    Drop-in replacement for ``xr.concat`` when all slices already share the
    same dims, shape, and coords. Skips alignment, deepcopy, and reindex —
    just stacks the underlying numpy arrays.

    Args:
        arrays: DataArrays with identical dims, shape, and coords.
        dim: Index for the new leading dimension.

    Raises:
        ValueError: If *arrays* is empty or any slice has a different shape or dims than the first.
    """
    if not arrays:
        raise ValueError("fast_concat: 'arrays' must not be empty")
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
    value: Variate,
    coords: Mapping[str, Any],
    *,
    name: str = 'value',
    broadcast: bool = True,
) -> xr.DataArray:
    """Convert a Variate to a DataArray aligned to given coordinates.

    Pipeline: ``convert → validate dims → validate coord values → broadcast``.

    See :data:`Variate` for accepted inputs. Pandas inputs (``Series``,
    ``DataFrame``) follow the same convention as ``linopy.as_dataarray``: the
    axis ``name`` attribute selects the corresponding target dim. For
    ``ndarray``/``list``, the dim is selected by length (must be unambiguous).
    For ``DataArray``, dims must be a subset of *coords* and coord values must
    match exactly — alignment errors are surfaced loudly, not silently masked.

    Args:
        value: Scalar, list, ndarray, Series, DataFrame, or DataArray.
        coords: Target coordinates, e.g. ``{"time": idx, "period": pidx}``.
            Used both as the reach declaration and as alignment targets.
        name: Name for the resulting DataArray.
        broadcast: Expand result to span all dimensions in *coords*.
    """
    if isinstance(value, ProfileRef):
        raise ValueError(
            f'Unresolved ProfileRef {value!r}: resolve it to an array via ProfileRef.resolve(profiles) '
            f'before building the model.'
        )

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

    # --- 1) Convert to DataArray ---
    da: xr.DataArray
    if isinstance(value, xr.DataArray):
        da = value
    elif isinstance(value, (pd.Series, pd.DataFrame)):
        # Mirror linopy: pandas axes already carry coords; use axis.name as dim.
        # Fall back to length-matching only when no axis is named.
        named = [a.name for a in value.axes if a.name is not None]
        if len(named) == value.ndim:
            da = xr.DataArray(value)
        elif value.ndim == 1 and not named:
            return _from_unnamed_1d(np.asarray(value.values, dtype=float), coord_idx, name, broadcast)
        else:
            raise ValueError(
                f'{type(value).__name__} requires axis.name set on every axis '
                f'(got {[a.name for a in value.axes]!r}). '
                f"Set e.g. df.index.name='time', df.columns.name='period'."
            )
    elif isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(
                f'np.ndarray must be 1-D (got ndim={value.ndim}); pass an xr.DataArray '
                f'or pd.DataFrame with named axes for higher-dim inputs.'
            )
        return _from_unnamed_1d(value, coord_idx, name, broadcast)
    elif isinstance(value, list):
        return _from_unnamed_1d(np.asarray(value, dtype=float), coord_idx, name, broadcast)
    else:
        raise TypeError(f'Unsupported Variate type: {type(value)}')

    # --- 2) Validate dims are a subset of the target ---
    foreign = [str(d) for d in da.dims if d not in coord_idx]
    if foreign:
        raise ValueError(
            f'{type(value).__name__} has dims {foreign} not in target coords {list(coord_idx)}. '
            f'Rename before calling as_dataarray().'
        )

    # --- 3) Validate coord values match exactly (close the alignment gap) ---
    for d in da.dims:
        dim_name = str(d)
        if d in da.coords and not pd.Index(da.coords[d].values).equals(coord_idx[dim_name]):
            raise ValueError(
                f'Coord mismatch on dim {dim_name!r}: input coord does not equal target. '
                f"Use the same index as the model's {dim_name}."
            )

    da = da.rename(name)
    if broadcast:
        for dim, idx in coord_idx.items():
            if dim not in da.dims:
                da = da.expand_dims({dim: idx})
        da = da.transpose(*coord_idx)
    return da


def _from_unnamed_1d(arr: np.ndarray, coord_idx: dict[str, pd.Index], name: str, broadcast: bool) -> xr.DataArray:
    """Length-match an unnamed 1-D array to a single target coord.

    Tie-breaking: ``time`` wins over other dims when multiple match. Pass a
    named ``pd.Series`` or ``xr.DataArray`` to override.
    """
    arr = arr.astype(float)
    n = len(arr)
    matches = [k for k, v in coord_idx.items() if len(v) == n]
    if len(matches) == 0:
        lengths = ', '.join(f'{k}({len(v)})' for k, v in coord_idx.items())
        raise ValueError(f'Length {n} does not match any coordinate: {lengths}')
    if len(matches) > 1 and 'time' in matches:
        dim = 'time'
    elif len(matches) > 1:
        raise ValueError(
            f'Length {n} matches multiple coordinates: {matches}. '
            f'Pass an xr.DataArray, named pd.Series/DataFrame to disambiguate.'
        )
    else:
        dim = matches[0]
    da = xr.DataArray(arr, dims=[dim], coords={dim: coord_idx[dim]}, name=name)
    if broadcast:
        for d, idx in coord_idx.items():
            if d not in da.dims:
                da = da.expand_dims({d: idx})
        da = da.transpose(*coord_idx)
    return da


def normalize_timesteps(timesteps: Timesteps) -> TimeIndex:
    """Normalize user-provided timesteps to an internal time index.

    Args:
        timesteps: Datetime objects, integers, or a DatetimeIndex.

    Returns:
        A datetime index for datetime inputs, or an integer index for integer inputs.

    Raises:
        ValueError: If timesteps are not strictly monotonically increasing.
    """
    if len(timesteps) == 0:
        raise ValueError('Timesteps must not be empty')

    if isinstance(timesteps, pd.DatetimeIndex):
        idx: TimeIndex = timesteps
    elif isinstance(timesteps, pd.Index):
        if isinstance(timesteps, pd.RangeIndex) or pd.api.types.is_integer_dtype(timesteps.dtype):
            idx = timesteps
        elif pd.api.types.is_datetime64_any_dtype(timesteps.dtype):
            idx = pd.DatetimeIndex(timesteps)
        else:
            raise TypeError(f'Unsupported pd.Index dtype: {timesteps.dtype}. Use datetime or integer index.')
    elif not isinstance(timesteps, list):
        raise TypeError(f'Unsupported Timesteps type: {type(timesteps)}')
    elif isinstance(timesteps[0], datetime):
        idx = pd.DatetimeIndex(timesteps)
    elif type(timesteps[0]) is int:
        idx = pd.Index(timesteps)
        if not pd.api.types.is_integer_dtype(idx.dtype):
            raise TypeError('Integer timesteps contain non-integer values')
    else:
        raise TypeError(f'Unsupported timestep element type: {type(timesteps[0])}. Use datetime or int.')

    if len(idx) > 1 and not idx.is_monotonic_increasing:
        raise ValueError('Timesteps must be strictly monotonically increasing')
    if not idx.is_unique:
        raise ValueError('Timesteps contain duplicates')
    return idx


def compute_dt(timesteps: TimeIndex, dt: float | list[float] | None) -> xr.DataArray:
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
            values = np.array(dt, dtype=float)
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
