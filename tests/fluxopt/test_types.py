from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from fluxopt.types import to_polars_series


@pytest.fixture
def ts():
    return pl.Series('time', ['t0', 't1', 't2'])


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
