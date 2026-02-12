from __future__ import annotations

from typing import Union

import pandas as pd
import polars as pl

TimeSeries = Union[float, int, list[float], pl.Series, pd.Series]


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
    if isinstance(value, pd.Series):
        value = pl.from_pandas(value)
    if isinstance(value, pl.Series):
        if len(value) != n:
            raise ValueError(f'Series length {len(value)} does not match timesteps length {n}')
        return value.alias(name).cast(pl.Float64)
    raise TypeError(f'Unsupported TimeSeries type: {type(value)}')
