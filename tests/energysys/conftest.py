from __future__ import annotations

import pytest


@pytest.fixture
def timesteps_3():
    return ['t0', 't1', 't2']


@pytest.fixture
def timesteps_4():
    return ['t0', 't1', 't2', 't3']
