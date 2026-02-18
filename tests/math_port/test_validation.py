"""Validation tests for input parameter checking.

All tests skipped — flixopt-specific validation (PlausibilityError) not applicable.
"""

import pytest


@pytest.mark.skip(reason='flixopt-specific validation not applicable to fluxopt')
class TestValidation:
    def test_source_and_sink_requires_size_with_prevent_simultaneous(self):
        """Proves: SourceAndSink with prevent_simultaneous raises PlausibilityError
        when flows don't have a size."""
