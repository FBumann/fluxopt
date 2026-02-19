"""Tests for legacy solution access patterns.

All tests skipped — flixopt-specific legacy access not applicable to fluxopt.
"""

import pytest


@pytest.mark.skip(reason='flixopt-specific legacy solution access not applicable to fluxopt')
class TestLegacySolutionAccess:
    def test_effect_access(self, optimize):
        """Test legacy effect access."""

    def test_flow_rate_access(self, optimize):
        """Test legacy flow rate access."""

    def test_flow_size_access(self, optimize):
        """Test legacy flow size access."""

    def test_storage_level_access(self, optimize):
        """Test legacy storage level access."""

    def test_legacy_access_disabled_by_default(self):
        """Test that legacy access is disabled when CONFIG.Legacy.solution_access is False."""

    def test_legacy_access_emits_deprecation_warning(self, optimize):
        """Test that legacy access emits DeprecationWarning."""
