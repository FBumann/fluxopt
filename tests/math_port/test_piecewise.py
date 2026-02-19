"""Mathematical correctness tests for piecewise linearization.

All tests skipped — piecewise conversion is not supported in fluxopt.
"""

import pytest


@pytest.mark.skip(reason='piecewise conversion not supported in fluxopt')
class TestPiecewise:
    def test_piecewise_selects_cheap_segment(self, optimize):
        """Proves: PiecewiseConversion correctly interpolates within the active segment."""

    def test_piecewise_conversion_at_breakpoint(self, optimize):
        """Proves: PiecewiseConversion is consistent at segment boundaries."""

    def test_piecewise_with_gap_forces_minimum_load(self, optimize):
        """Proves: Gaps between pieces create forbidden operating regions."""

    def test_piecewise_gap_allows_off_state(self, optimize):
        """Proves: Piecewise with off-state piece allows unit to be completely off."""

    def test_piecewise_varying_efficiency_across_segments(self, optimize):
        """Proves: Different segments can have different efficiency ratios."""
