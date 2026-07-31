"""Guards on the public surface.

A narrow API is a feature we advertise, so widening it must be a deliberate,
reviewed diff rather than a side effect of adding an import to ``__init__``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import fluxopt

PUBLIC_API = {
    'PENALTY_EFFECT_ID',
    'Carrier',
    'Converter',
    'Dims',
    'Effect',
    'Flow',
    'FlowSystem',
    'FlowSystemModel',
    'IdList',
    'Investment',
    'ModelData',
    'PiecewiseConversion',
    'Port',
    'ProfileRef',
    'Result',
    'Sizing',
    'Status',
    'Storage',
    'TimeIndex',
    'Timesteps',
    'Variate',
    'all_element_schemas',
    'as_dataarray',
    'element_schema',
    'from_dict',
    'optimize',
    'to_dict',
}

README = Path(__file__).parent.parent / 'README.md'


def readme_public_api() -> set[str]:
    """The names listed in the README's public-API table."""
    block = re.search(
        r'<!--public-api-start-->(.*?)<!--public-api-end-->',
        README.read_text(encoding='utf-8'),
        re.DOTALL,
    )
    assert block is not None, 'README lost its <!--public-api-start--> markers'
    rows = [line for line in block.group(1).splitlines() if line.startswith('| **')]
    return {name for row in rows for name in re.findall(r'`([^`]+)`', row)}


def test_all_matches_the_pinned_surface() -> None:
    assert set(fluxopt.__all__) == PUBLIC_API


def test_all_has_no_duplicates() -> None:
    """Ordering is ruff's RUF022 job; duplicates it would not catch."""
    assert len(fluxopt.__all__) == len(set(fluxopt.__all__))


@pytest.mark.parametrize('name', sorted(PUBLIC_API))
def test_public_name_is_importable(name: str) -> None:
    assert hasattr(fluxopt, name), f'{name} is exported but missing from the package'


def test_readme_table_lists_exactly_the_public_surface() -> None:
    assert readme_public_api() == PUBLIC_API
