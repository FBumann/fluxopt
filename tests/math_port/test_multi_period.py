"""Mathematical correctness tests for multi-period optimization.

All tests skipped — multi-period optimization is not supported in fluxopt.
"""

import pytest


@pytest.mark.skip(reason='multi-period optimization not supported in fluxopt')
class TestMultiPeriod:
    def test_period_weights_affect_objective(self, optimize):
        """Proves: period weights scale per-period costs in the objective."""

    def test_flow_hours_max_over_periods(self, optimize):
        """Proves: flow_hours_max_over_periods caps the weighted total flow-hours."""

    def test_flow_hours_min_over_periods(self, optimize):
        """Proves: flow_hours_min_over_periods forces a minimum weighted total."""

    def test_effect_maximum_over_periods(self, optimize):
        """Proves: Effect.maximum_over_periods caps weighted total of an effect."""

    def test_effect_minimum_over_periods(self, optimize):
        """Proves: Effect.minimum_over_periods forces minimum weighted total."""

    def test_invest_linked_periods(self, optimize):
        """Proves: InvestParameters.linked_periods forces equal sizes across periods."""

    def test_effect_period_weights(self, optimize):
        """Proves: Effect.period_weights overrides default period weights."""

    def test_storage_relative_minimum_final_level_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_level works in multi-period."""

    def test_storage_relative_maximum_final_level_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_level works in multi-period."""
