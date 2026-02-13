from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def timesteps_3():
    return [datetime(2024, 1, 1, h) for h in range(3)]


@pytest.fixture
def timesteps_4():
    return [datetime(2024, 1, 1, h) for h in range(4)]
