from __future__ import annotations

from datetime import datetime

import pandas as pd
import polars as pl
import pytest

from fluxopt.types import compute_dt, compute_end_time, normalize_timesteps, to_polars_series


@pytest.fixture
def ts():
    return pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime)


class TestToPolarsSeriesScalar:
    def test_int(self, ts):
        result = to_polars_series(5, ts)
        assert result.to_list() == [5.0, 5.0, 5.0]

    def test_float(self, ts):
        result = to_polars_series(3.14, ts)
        assert result.to_list() == [3.14, 3.14, 3.14]


class TestToPolarsSeriesList:
    def test_matching_length(self, ts):
        result = to_polars_series([1.0, 2.0, 3.0], ts)
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_wrong_length(self, ts):
        with pytest.raises(ValueError, match='does not match'):
            to_polars_series([1.0, 2.0], ts)


class TestToPolarsSeriesSeries:
    def test_polars_series(self, ts):
        s = pl.Series('data', [10.0, 20.0, 30.0])
        result = to_polars_series(s, ts, name='val')
        assert result.name == 'val'
        assert result.to_list() == [10.0, 20.0, 30.0]

    def test_pandas_series(self, ts):
        s = pd.Series([10.0, 20.0, 30.0])
        result = to_polars_series(s, ts)
        assert result.to_list() == [10.0, 20.0, 30.0]

    def test_wrong_length_series(self, ts):
        s = pl.Series('data', [1.0, 2.0])
        with pytest.raises(ValueError, match='does not match'):
            to_polars_series(s, ts)


class TestToPolarsSeriesUnsupported:
    def test_dict(self, ts):
        with pytest.raises(TypeError, match='Unsupported'):
            to_polars_series({}, ts)


class TestNormalizeTimesteps:
    def test_datetime_list(self):
        dts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = normalize_timesteps(dts)
        assert result.name == 'time'
        assert result.dtype == pl.Datetime
        assert result.to_list() == dts

    def test_int_list(self):
        result = normalize_timesteps([0, 1, 2])
        assert result.name == 'time'
        assert result.dtype == pl.Int64
        assert result.to_list() == [0, 1, 2]

    def test_string_list_rejected(self):
        with pytest.raises(TypeError, match='Use datetime or int'):
            normalize_timesteps(['t0', 't1', 't2'])

    def test_polars_series_string_rejected(self):
        s = pl.Series('ts', ['a', 'b', 'c'])
        with pytest.raises(TypeError, match='String timesteps are not supported'):
            normalize_timesteps(s)

    def test_polars_series_datetime(self):
        dts = [datetime(2024, 1, 1, h) for h in range(3)]
        s = pl.Series('ts', dts, dtype=pl.Datetime)
        result = normalize_timesteps(s)
        assert result.name == 'time'
        assert result.dtype == pl.Datetime

    def test_polars_series_int(self):
        s = pl.Series('ts', [0, 1, 2], dtype=pl.Int64)
        result = normalize_timesteps(s)
        assert result.name == 'time'
        assert result.dtype == pl.Int64
        assert result.to_list() == [0, 1, 2]

    def test_pandas_datetimeindex(self):
        idx = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(3)])
        result = normalize_timesteps(idx)
        assert result.name == 'time'
        assert result.dtype == pl.Datetime
        assert len(result) == 3

    def test_empty_list(self):
        result = normalize_timesteps([])
        assert result.name == 'time'
        assert result.dtype == pl.Datetime
        assert len(result) == 0


class TestComputeDt:
    def test_explicit_scalar(self):
        ts = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime)
        result = compute_dt(ts, 0.5)
        assert result.name == 'dt'
        assert result.to_list() == [0.5, 0.5, 0.5]

    def test_explicit_list(self):
        ts = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime)
        result = compute_dt(ts, [1.0, 2.0, 3.0])
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_explicit_list_wrong_length(self):
        ts = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(2)], dtype=pl.Datetime)
        with pytest.raises(ValueError, match='dt length'):
            compute_dt(ts, [1.0, 2.0, 3.0])

    def test_explicit_series(self):
        ts = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime)
        dt = pl.Series('vals', [1.0, 2.0, 3.0])
        result = compute_dt(ts, dt)
        assert result.name == 'dt'
        assert result.to_list() == [1.0, 2.0, 3.0]

    def test_auto_int_defaults_to_1(self):
        ts = pl.Series('time', [0, 1, 2], dtype=pl.Int64)
        result = compute_dt(ts, None)
        assert result.to_list() == [1.0, 1.0, 1.0]

    def test_auto_datetime_hourly(self):
        dts = [datetime(2024, 1, 1, h) for h in range(4)]
        ts = pl.Series('time', dts, dtype=pl.Datetime)
        result = compute_dt(ts, None)
        assert result.to_list() == [1.0, 1.0, 1.0, 1.0]

    def test_auto_datetime_irregular(self):
        dts = [
            datetime(2024, 1, 1, 0),
            datetime(2024, 1, 1, 1),
            datetime(2024, 1, 1, 4),
        ]
        ts = pl.Series('time', dts, dtype=pl.Datetime)
        result = compute_dt(ts, None)
        assert result.to_list() == [1.0, 1.0, 3.0]

    def test_single_timestep(self):
        ts = pl.Series('time', [0], dtype=pl.Int64)
        result = compute_dt(ts, None)
        assert result.to_list() == [1.0]

    def test_single_datetime_timestep(self):
        ts = pl.Series('time', [datetime(2024, 1, 1)], dtype=pl.Datetime)
        result = compute_dt(ts, None)
        assert result.to_list() == [1.0]


class TestComputeEndTime:
    def test_int(self):
        ts = pl.Series('time', [0, 1, 2], dtype=pl.Int64)
        dt = pl.Series('dt', [1.0, 1.0, 1.0])
        assert compute_end_time(ts, dt) == 3

    def test_datetime(self):
        ts = pl.Series('time', [datetime(2024, 1, 1, h) for h in range(3)], dtype=pl.Datetime)
        dt = pl.Series('dt', [1.0, 1.0, 1.0])
        result = compute_end_time(ts, dt)
        assert result == datetime(2024, 1, 1, 3)

    def test_datetime_irregular(self):
        dts = [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 4)]
        ts = pl.Series('time', dts, dtype=pl.Datetime)
        dt = pl.Series('dt', [1.0, 1.0, 3.0])
        result = compute_end_time(ts, dt)
        assert result == datetime(2024, 1, 1, 7)
