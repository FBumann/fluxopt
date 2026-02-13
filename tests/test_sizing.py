from __future__ import annotations

import pytest

from fluxopt.elements import Sizing


class TestSizingConstruction:
    def test_basic(self):
        s = Sizing(min_size=10, max_size=100)
        assert s.min_size == 10
        assert s.max_size == 100
        assert s.mandatory is False
        assert s.effects_per_size == {}
        assert s.effects_of_size == {}

    def test_binary_min_eq_max(self):
        s = Sizing(min_size=50, max_size=50)
        assert s.min_size == s.max_size == 50

    def test_with_effects(self):
        s = Sizing(
            min_size=0,
            max_size=200,
            effects_per_size={'cost': 100.0},
            effects_of_size={'cost': 5000.0, 'co2': 50.0},
        )
        assert s.effects_per_size == {'cost': 100.0}
        assert s.effects_of_size == {'cost': 5000.0, 'co2': 50.0}

    def test_mandatory(self):
        s = Sizing(min_size=10, max_size=100, mandatory=True)
        assert s.mandatory is True


class TestSizingValidation:
    def test_negative_min_size(self):
        with pytest.raises(ValueError, match='min_size must be >= 0'):
            Sizing(min_size=-1, max_size=100)

    def test_zero_max_size(self):
        with pytest.raises(ValueError, match='max_size must be > 0'):
            Sizing(min_size=0, max_size=0)

    def test_negative_max_size(self):
        with pytest.raises(ValueError, match='max_size must be > 0'):
            Sizing(min_size=0, max_size=-5)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match=r'min_size.*must be <= max_size'):
            Sizing(min_size=100, max_size=50)

    def test_zero_min_size_is_valid(self):
        s = Sizing(min_size=0, max_size=100)
        assert s.min_size == 0
