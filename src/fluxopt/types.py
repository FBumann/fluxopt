from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    import pandas as pd

type TimeSeries = float | int | list[float] | pl.Series | pd.Series


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
