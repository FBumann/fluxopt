from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

type TimeSeries = float | int | list[float] | pl.Series | pd.Series
type Timesteps = list[datetime] | list[str] | pl.Series | pd.DatetimeIndex


def to_polars_series(value: TimeSeries, timesteps: pl.Series, name: str = 'value') -> pl.Series:
    """Convert any TimeSeries input to a pl.Series aligned with timesteps.

    Scalar -> broadcast, list -> length-check, Series -> rename/align.
    """
    n = len(timesteps)
    if isinstance(value, (int, float)):
        return pl.Series(name, [float(value)] * n)
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f'List length {len(value)} does not match timesteps length {n}')
        return pl.Series(name, [float(v) for v in value])
    if not isinstance(value, pl.Series):
        try:
            value = pl.from_pandas(value)
        except TypeError:
            raise TypeError(f'Unsupported TimeSeries type: {type(value)}') from None
    if len(value) != n:
        raise ValueError(f'Series length {len(value)} does not match timesteps length {n}')
    return value.alias(name).cast(pl.Float64)


def normalize_timesteps(timesteps: Timesteps) -> pl.Series:
    """Convert any Timesteps input to a pl.Series named 'time'.

    Preserves dtype: pl.Datetime for datetime inputs, pl.String for string inputs.
    """
    if isinstance(timesteps, pl.Series):
        return timesteps.alias('time')

    # pandas DatetimeIndex
    if not isinstance(timesteps, list):
        try:
            return pl.Series('time', timesteps.to_pydatetime().tolist(), dtype=pl.Datetime)
        except AttributeError:
            raise TypeError(f'Unsupported Timesteps type: {type(timesteps)}') from None

    # list[datetime] or list[str]
    if len(timesteps) == 0:
        return pl.Series('time', [], dtype=pl.String)
    if isinstance(timesteps[0], datetime):
        return pl.Series('time', timesteps, dtype=pl.Datetime)
    return pl.Series('time', timesteps, dtype=pl.String)


def compute_dt(timesteps: pl.Series, dt: float | list[float] | pl.Series | None) -> pl.Series:
    """Compute dt (hours) for each timestep.

    - dt=None + datetime: derive from consecutive differences in hours; last = second-to-last.
    - dt=None + string: broadcast 1.0.
    - dt provided: validate length, return as pl.Series.
    - Single timestep: default to 1.0.
    """
    n = len(timesteps)

    if dt is not None:
        if isinstance(dt, (int, float)):
            return pl.Series('dt', [float(dt)] * n)
        if isinstance(dt, pl.Series):
            if len(dt) != n:
                raise ValueError(f'dt length {len(dt)} does not match timesteps length {n}')
            return dt.alias('dt').cast(pl.Float64)
        # list
        if len(dt) != n:
            raise ValueError(f'dt length {len(dt)} does not match timesteps length {n}')
        return pl.Series('dt', [float(v) for v in dt])

    # Auto-derive
    if n <= 1:
        return pl.Series('dt', [1.0] * n)

    if timesteps.dtype == pl.String:
        return pl.Series('dt', [1.0] * n)

    # Datetime: derive from diff in hours
    diffs = timesteps.diff().dt.total_seconds() / 3600.0
    dt_values = diffs.to_list()
    # First element is None from diff(); use the second element
    dt_values[0] = dt_values[1]
    # Extend last: already correct from diff, but last element uses second-to-last
    # Actually diff() gives [None, d1, d2, ..., d_{n-1}] so we have n values
    # dt_values[0] = dt_values[1] already handles first
    # The last value is the diff between last and second-to-last, which is correct
    return pl.Series('dt', dt_values)


def compute_end_time(timesteps: pl.Series, dt_series: pl.Series) -> object:
    """Return the time value one step past the last timestep.

    For datetime: last_time + timedelta(hours=last_dt).
    For strings: '_end'.
    """
    if timesteps.dtype == pl.String:
        return '_end'
    last_time: datetime = timesteps[-1]
    last_dt: float = dt_series[-1]
    return last_time + timedelta(hours=last_dt)
