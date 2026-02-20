from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from fluxopt import Flow, Port


def ts(n: int) -> list[datetime]:
    """Create *n* hourly timesteps starting 2024-01-01.

    Args:
        n: Number of timesteps to generate.
    """
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def waste(bus: str) -> Port:
    """Free-disposal port that absorbs excess on *bus* at zero cost.

    Args:
        bus: Name of the bus to attach the waste port to.
    """
    return Port(f'_waste_{bus}', exports=[Flow(bus=bus)])


def _block_lengths(on: np.ndarray, *, active: bool) -> list[tuple[int, int]]:
    """Return (start_index, length) for each contiguous block.

    Args:
        on: Binary array (values > 0.5 are "on").
        active: True to find on-blocks, False to find off-blocks.
    """
    binary = np.asarray(on) > 0.5
    if not active:
        binary = ~binary
    if len(binary) == 0:
        return []
    changes = np.diff(binary.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if binary[0]:
        starts = np.concatenate([[0], starts])
    if binary[-1]:
        ends = np.concatenate([ends, [len(binary)]])
    return list(zip(starts.tolist(), (ends - starts).tolist(), strict=True))


def _check_blocks(
    blocks: list[tuple[int, int]],
    on: np.ndarray,
    label: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> None:
    for start, length in blocks:
        if min_length is not None:
            assert length >= min_length, f'{label}-block of {length} < min {min_length} at t={start}: {on}'
        if max_length is not None:
            assert length <= max_length, f'{label}-block of {length} > max {max_length} at t={start}: {on}'


def assert_on_blocks(
    on: np.ndarray,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> None:
    """Assert every contiguous on-block has duration in [min_length, max_length].

    Args:
        on: Binary on/off array (values > 0.5 are "on").
        min_length: Minimum allowed block length (inclusive).
        max_length: Maximum allowed block length (inclusive).
    """
    _check_blocks(_block_lengths(on, active=True), on, 'on', min_length=min_length, max_length=max_length)


def assert_off_blocks(
    on: np.ndarray,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    skip_leading: bool = True,
) -> None:
    """Assert every contiguous off-block has duration in [min_length, max_length].

    Args:
        on: Binary on/off array (values <= 0.5 are "off").
        min_length: Minimum allowed block length (inclusive).
        max_length: Maximum allowed block length (inclusive).
        skip_leading: If True, ignore the first off-block (may be carry-over from prior).
    """
    blocks = _block_lengths(on, active=False)
    if skip_leading and blocks and blocks[0][0] == 0:
        blocks = blocks[1:]
    _check_blocks(blocks, on, 'off', min_length=min_length, max_length=max_length)
