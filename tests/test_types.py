from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxopt.types import as_dataarray, compute_dt, normalize_timesteps, to_data_array


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


class TestAsDataArrayScalar:
    def test_no_broadcast_returns_0dim(self):
        result = as_dataarray(5.0, {'time': pd.RangeIndex(3)})
        assert result.shape == ()
        assert float(result) == 5.0
        assert result.name == 'value'

    def test_int_no_broadcast(self):
        result = as_dataarray(3, {'time': pd.RangeIndex(3)})
        assert result.shape == ()
        assert float(result) == 3.0

    def test_broadcast_single_coord(self):
        idx = pd.RangeIndex(3)
        result = as_dataarray(5.0, {'time': idx}, broadcast=True)
        assert result.shape == (3,)
        assert list(result.values) == [5.0, 5.0, 5.0]
        assert result.dims == ('time',)

    def test_broadcast_multi_coord(self):
        flows = pd.Index(['gas', 'elec'])
        time = pd.RangeIndex(4)
        result = as_dataarray(2.0, {'flow': flows, 'time': time}, broadcast=True)
        assert result.shape == (2, 4)
        assert result.dims == ('flow', 'time')
        np.testing.assert_array_equal(result.values, np.full((2, 4), 2.0))

    def test_custom_name(self):
        result = as_dataarray(1.0, {'t': [0, 1]}, name='cost', broadcast=True)
        assert result.name == 'cost'


class TestAsDataArrayList:
    def test_single_coord(self):
        result = as_dataarray([1.0, 2.0, 3.0], {'time': pd.RangeIndex(3)})
        assert result.dims == ('time',)
        assert list(result.values) == [1.0, 2.0, 3.0]

    def test_multi_coord_matches_correct_dim(self):
        flows = pd.Index(['a', 'b'])
        time = pd.RangeIndex(3)
        result = as_dataarray([10.0, 20.0, 30.0], {'flow': flows, 'time': time})
        assert result.dims == ('time',)
        assert list(result.values) == [10.0, 20.0, 30.0]

    def test_multi_coord_broadcast(self):
        flows = pd.Index(['a', 'b'])
        time = pd.RangeIndex(3)
        result = as_dataarray([10.0, 20.0, 30.0], {'flow': flows, 'time': time}, broadcast=True)
        assert result.dims == ('flow', 'time')
        assert result.shape == (2, 3)

    def test_broadcast_preserves_coord_order(self):
        """Data matches 'time' but coords list it second — dims must follow coords order."""
        time = pd.RangeIndex(3)
        flows = pd.Index(['a', 'b'])
        result = as_dataarray([1.0, 2.0, 3.0], {'time': time, 'flow': flows}, broadcast=True)
        assert result.dims == ('time', 'flow')
        assert result.shape == (3, 2)

    def test_ambiguous_length_raises(self):
        c1 = pd.RangeIndex(3)
        c2 = pd.Index(['a', 'b', 'c'])
        with pytest.raises(ValueError, match='matches multiple coordinates'):
            as_dataarray([1.0, 2.0, 3.0], {'x': c1, 'y': c2})

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match='does not match any coordinate'):
            as_dataarray([1.0, 2.0], {'time': pd.RangeIndex(5)})


class TestAsDataArrayNdarray:
    def test_array(self):
        arr = np.array([10.0, 20.0])
        result = as_dataarray(arr, {'flow': pd.Index(['a', 'b'])})
        assert result.dims == ('flow',)
        assert list(result.values) == [10.0, 20.0]


class TestAsDataArraySeries:
    def test_series(self):
        s = pd.Series([4.0, 5.0, 6.0])
        result = as_dataarray(s, {'time': pd.RangeIndex(3)})
        assert result.dims == ('time',)
        assert list(result.values) == [4.0, 5.0, 6.0]


class TestAsDataArrayDataArray:
    def test_passthrough(self):
        da = xr.DataArray([1.0, 2.0], dims=['time'], coords={'time': [0, 1]})
        result = as_dataarray(da, {'time': pd.RangeIndex(2)})
        assert result.name == 'value'
        assert result.dims == ('time',)
        assert list(result.values) == [1.0, 2.0]

    def test_broadcast_expands_dims(self):
        da = xr.DataArray([1.0, 2.0], dims=['time'], coords={'time': [0, 1]})
        flows = pd.Index(['a', 'b', 'c'])
        result = as_dataarray(da, {'flow': flows, 'time': pd.RangeIndex(2)}, broadcast=True)
        assert result.dims == ('flow', 'time')
        assert result.shape == (3, 2)

    def test_broadcast_preserves_coord_order(self):
        """Data has dim 'time' but coords list it first — dims must follow coords order."""
        da = xr.DataArray([1.0, 2.0], dims=['time'], coords={'time': [0, 1]})
        flows = pd.Index(['a', 'b', 'c'])
        result = as_dataarray(da, {'time': pd.RangeIndex(2), 'flow': flows}, broadcast=True)
        assert result.dims == ('time', 'flow')

    def test_foreign_dims_raises(self):
        """DataArray with dims not in coords raises ValueError."""
        da = xr.DataArray([10.0, 20.0], dims=['sizing_flow'])
        with pytest.raises(ValueError, match='not in target coords'):
            as_dataarray(da, {'flow': pd.Index(['a', 'b'])})


class TestAsDataArrayUnsupported:
    def test_dict_raises(self):
        with pytest.raises(TypeError, match='Unsupported'):
            as_dataarray({}, {'time': pd.RangeIndex(3)})
