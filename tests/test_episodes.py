"""Episodes: derived episode partitioning of a temporal dim."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from fluxopt.constraints import Episodes


def _coord(n: int, dim: str = 'time') -> xr.DataArray:
    labels = pd.date_range('2030-01-01', periods=n, freq='h')
    return xr.DataArray(labels, dims=[dim], coords={dim: labels})


class TestConstruction:
    def test_single_is_one_episode(self):
        eps = Episodes.single(_coord(4))
        assert list(eps.starts.values) == [True, False, False, False]
        assert eps.n_episodes == 1
        assert eps.dim == 'time'

    def test_from_changes_cuts_where_labeling_changes(self):
        coord = _coord(5)
        labeling = xr.DataArray([2030, 2030, 2040, 2040, 2040], dims=['time'], coords={'time': coord})
        eps = Episodes.from_changes(labeling)
        assert list(eps.starts.values) == [True, False, True, False, False]
        assert eps.n_episodes == 2

    def test_from_changes_unions_multiple_labelings(self):
        coord = _coord(6)
        period = xr.DataArray([1, 1, 1, 2, 2, 2], dims=['time'], coords={'time': coord})
        cluster = xr.DataArray([0, 0, 1, 0, 0, 1], dims=['time'], coords={'time': coord})
        eps = Episodes.from_changes(period, cluster)
        assert list(eps.starts.values) == [True, False, True, True, False, True]
        assert eps.n_episodes == 4

    def test_or_unions_boundary_sets(self):
        coord = _coord(4)
        a = Episodes.from_changes(xr.DataArray([1, 1, 2, 2], dims=['time'], coords={'time': coord}))
        b = Episodes.from_changes(xr.DataArray([1, 2, 2, 2], dims=['time'], coords={'time': coord}))
        assert list((a | b).starts.values) == [True, True, True, False]

    def test_rejects_false_first_position(self):
        flags = xr.DataArray([False, True], dims=['time'])
        with pytest.raises(ValueError, match='first timestep'):
            Episodes(flags)

    def test_rejects_non_boolean(self):
        with pytest.raises(ValueError, match='boolean'):
            Episodes(xr.DataArray([1, 0], dims=['time']))


class TestDerived:
    def _eps(self) -> Episodes:
        coord = _coord(5)
        labeling = xr.DataArray([1, 1, 1, 2, 2], dims=['time'], coords={'time': coord})
        return Episodes.from_changes(labeling)

    def test_positions(self):
        eps = self._eps()
        assert list(eps.start_positions) == [0, 3]
        assert list(eps.last_positions) == [2, 4]
        assert list(eps.episode_ids) == [0, 0, 0, 1, 1]

    def test_chain_mask_blocks_boundary_links(self):
        eps = self._eps()
        assert list(eps.chain_mask.values) == [True, True, False, True]

    def test_max_duration_is_longest_episode(self):
        eps = self._eps()
        dt = xr.DataArray(np.array([1.0, 1.0, 1.0, 4.0, 4.0]), dims=['time'])
        assert eps.max_duration(dt) == 8.0

    def test_check_rejects_wrong_axis(self):
        eps = self._eps()
        with pytest.raises(ValueError, match='do not match'):
            eps.check('time', 7)
        with pytest.raises(ValueError, match='do not match'):
            eps.check('snapshot', 5)
