from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def timesteps_3():
    return ['t0', 't1', 't2']


@pytest.fixture
def timesteps_4():
    return ['t0', 't1', 't2', 't3']


@pytest.fixture
def timesteps_3_dt():
    return [datetime(2024, 1, 1, h) for h in range(3)]


@pytest.fixture
def timesteps_4_dt():
    return [datetime(2024, 1, 1, h) for h in range(4)]
