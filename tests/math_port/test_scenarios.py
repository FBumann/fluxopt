"""Mathematical correctness tests for scenario optimization.

All tests skipped — scenario optimization is not supported in fluxopt.
"""

import pytest


@pytest.mark.skip(reason='scenario optimization not supported in fluxopt')
class TestScenarios:
    def test_scenario_weights_affect_objective(self, optimize):
        """Proves: scenario weights correctly weight per-scenario costs."""

    def test_scenario_independent_sizes(self, optimize):
        """Proves: scenario_independent_sizes=True forces the same invested size."""

    def test_scenario_independent_flow_rates(self, optimize):
        """Proves: scenario_independent_flow_rates forces identical flow rates."""

    def test_storage_relative_minimum_final_level_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_level works with scenarios."""

    def test_storage_relative_maximum_final_level_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_level works with scenarios."""
