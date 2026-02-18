"""Mathematical correctness tests for time-series clustering.

All tests skipped — clustering is not supported in fluxopt.
"""

import pytest


@pytest.mark.skip(reason='clustering not supported in fluxopt')
class TestClustering:
    def test_clustering_preserves_total_cost(self, optimize):
        """Proves: clustering with period weights approximates full-resolution cost."""

    def test_clustering_aggregation_consistency(self, optimize):
        """Proves: clustered solution variables are consistent with aggregation."""
