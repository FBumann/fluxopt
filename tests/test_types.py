from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fluxopt.types import compute_dt, normalize_timesteps, to_data_array


@pytest.fixture
def ts():
    return pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(3)])


class TestToDataArrayScalar:
    def test_int(self, ts):
        result = to_data_array(5, ts)
        assert list(result.values) == [5.0, 5.0, 5.0]

    def test_float(self, ts):
        result = to_data_array(3.14, ts)
        assert list(result.values) == [3.14, 3.14, 3.14]


class TestToDataArrayList:
    def test_matching_length(self, ts):
        result = to_data_array([1.0, 2.0, 3.0], ts)
        assert list(result.values) == [1.0, 2.0, 3.0]

    def test_wrong_length(self, ts):
        with pytest.raises(ValueError, match='does not match'):
            to_data_array([1.0, 2.0], ts)


class TestToDataArraySeries:
    def test_numpy_array(self, ts):
        arr = np.array([10.0, 20.0, 30.0])
        result = to_data_array(arr, ts, name='val')
        assert result.name == 'val'
        assert list(result.values) == [10.0, 20.0, 30.0]

    def test_pandas_series(self, ts):
        s = pd.Series([10.0, 20.0, 30.0])
        result = to_data_array(s, ts)
        assert list(result.values) == [10.0, 20.0, 30.0]

    def test_wrong_length_array(self, ts):
        arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match='does not match'):
            to_data_array(arr, ts)


class TestToDataArrayUnsupported:
    def test_dict(self, ts):
        with pytest.raises(TypeError, match='Unsupported'):
            to_data_array({}, ts)


class TestNormalizeTimesteps:
    def test_datetime_list(self):
        dts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = normalize_timesteps(dts)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3

    def test_int_list(self):
        result = normalize_timesteps([0, 1, 2])
        assert list(result) == [0, 1, 2]
        assert result.dtype == np.int64

    def test_string_list_rejected(self):
        with pytest.raises(TypeError, match='Use datetime or int'):
            normalize_timesteps(['t0', 't1', 't2'])

    def test_pandas_datetimeindex(self):
        idx = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(3)])
        result = normalize_timesteps(idx)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3

    def test_empty_list(self):
        result = normalize_timesteps([])
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 0


class TestComputeDt:
    def test_explicit_scalar(self):
        ts = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(3)])
        result = compute_dt(ts, 0.5)
        assert list(result.values) == [0.5, 0.5, 0.5]

    def test_explicit_list(self):
        ts = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(3)])
        result = compute_dt(ts, [1.0, 2.0, 3.0])
        assert list(result.values) == [1.0, 2.0, 3.0]

    def test_explicit_list_wrong_length(self):
        ts = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(2)])
        with pytest.raises(ValueError, match='dt length'):
            compute_dt(ts, [1.0, 2.0, 3.0])

    def test_auto_int_defaults_to_1(self):
        ts = pd.Index([0, 1, 2], dtype=np.int64)
        result = compute_dt(ts, None)
        assert list(result.values) == [1.0, 1.0, 1.0]

    def test_auto_datetime_hourly(self):
        ts = pd.DatetimeIndex([datetime(2024, 1, 1, h) for h in range(4)])
        result = compute_dt(ts, None)
        assert list(result.values) == [1.0, 1.0, 1.0, 1.0]

    def test_auto_datetime_irregular(self):
        dts = [
            datetime(2024, 1, 1, 0),
            datetime(2024, 1, 1, 1),
            datetime(2024, 1, 1, 4),
        ]
        ts = pd.DatetimeIndex(dts)
        result = compute_dt(ts, None)
        assert list(result.values) == [1.0, 1.0, 3.0]

    def test_single_timestep(self):
        ts = pd.Index([0], dtype=np.int64)
        result = compute_dt(ts, None)
        assert list(result.values) == [1.0]

    def test_single_datetime_timestep(self):
        ts = pd.DatetimeIndex([datetime(2024, 1, 1)])
        result = compute_dt(ts, None)
        assert list(result.values) == [1.0]
