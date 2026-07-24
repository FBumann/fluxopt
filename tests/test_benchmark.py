"""Smoke tests for the user-facing benchmark (``python -m fluxopt.benchmark``)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from fluxopt.benchmark import SYSTEMS, main, measure


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_reference_system_builds(name):
    """Each reference system builds through Elements -> ModelData -> linopy at a small horizon."""
    row = measure(name, timesteps=48)
    assert row['model'] == name
    assert row['variables'] > 0
    assert row['constraints'] > 0
    assert row['binaries'] >= 0
    assert row['components'] > 0
    assert row['flows'] > 0
    assert row['effects'] > 0
    assert row['series'] > 0
    assert row['time'] > 0
    assert row['periods'] >= 1
    assert all(row[key] >= 0 for key in ('elements_s', 'data_s', 'build_s'))
    assert 'solve_s' not in row


def test_measure_with_solve():
    """--solve runs HiGHS and reports its timing."""
    row = measure('district_heating', timesteps=24, solve=True)
    assert row['solve_s'] >= 0


def test_worker_prints_json(capsys):
    """Worker mode prints a single JSON object for the parent process to collect."""
    exit_code = main(['--worker', 'district_heating', '--timesteps', '24'])
    assert exit_code == 0
    row = json.loads(capsys.readouterr().out)
    assert row['model'] == 'district_heating'
    assert row['timesteps'] == 24


def test_cli_end_to_end():
    """`python -m fluxopt.benchmark <model> --json` produces one JSON row per model."""
    proc = subprocess.run(
        [sys.executable, '-m', 'fluxopt.benchmark', 'district_heating', '--timesteps', '24', '--json'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert [row['model'] for row in rows] == ['district_heating']
    assert rows[0]['variables'] > 0
