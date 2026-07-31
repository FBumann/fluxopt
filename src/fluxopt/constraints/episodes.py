"""Episode partitioning of a temporal dimension.

An *episode* is a maximal run of timesteps that are consecutive in modeled
time: temporal-coupling constraints (state accumulation, duration windows,
switch transitions, ramps) may chain from one timestep to the next only
within an episode. Episode boundaries mark discontinuities in modeled time —
investment-period starts today, representative-period starts once time-series
aggregation lands — where every chain is cut and restarted.

Episodes are derived structure, not data: they are computed from ride-along
coordinates on the temporal dim (e.g. ``time_period``) and never serialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class Episodes:
    """Partition of a temporal dim into independent episodes.

    ``starts`` is the canonical representation — a boolean ``(dim,)`` array,
    True at each episode's first timestep; everything else derives from it.
    Constraint helpers require an ``Episodes`` explicitly: fluxopt models pass
    ``Dims.episodes``; custom single-episode axes pass :meth:`single`.
    """

    starts: xr.DataArray

    def __post_init__(self) -> None:
        """Validate a 1-D boolean array that opens with an episode start."""
        if self.starts.ndim != 1:
            raise ValueError(f'Episodes.starts must be 1-D, got dims {self.starts.dims}')
        if self.starts.dtype != bool:
            raise ValueError(f'Episodes.starts must be boolean, got dtype {self.starts.dtype}')
        if len(self.starts) and not bool(self.starts.values[0]):
            raise ValueError('Episodes.starts[0] must be True — the first timestep always begins an episode')

    @property
    def dim(self) -> str:
        """Name of the temporal dim this partition applies to."""
        return str(self.starts.dims[0])

    @classmethod
    def single(cls, coord: xr.DataArray) -> Episodes:
        """One episode spanning the whole axis (no internal boundaries).

        Args:
            coord: Coordinate labels of the temporal dim.
        """
        flags = np.zeros(len(coord), dtype=bool)
        if len(coord):
            flags[0] = True
        dim = str(coord.dims[0])
        return cls(xr.DataArray(flags, dims=[dim], coords={dim: coord}))

    @classmethod
    def from_changes(cls, *labelings: xr.DataArray) -> Episodes:
        """Episode boundaries wherever any per-timestep labeling changes value.

        Args:
            labelings: ``(dim,)`` arrays mapping each timestep to a group
                label (e.g. ``time_period``, later ``time_cluster``). A new
                episode starts at every position where any labeling differs
                from its predecessor.
        """
        first = labelings[0]
        flags = np.zeros(len(first), dtype=bool)
        if len(first):
            flags[0] = True
        for labeling in labelings:
            values = labeling.values
            flags[1:] |= values[1:] != values[:-1]
        dim = str(first.dims[0])
        return cls(xr.DataArray(flags, dims=[dim], coords={dim: first.coords[dim]}))

    def __or__(self, other: Episodes) -> Episodes:
        """Union of boundary sets: a chain is cut where either partition cuts it."""
        if self.dim != other.dim or len(self.starts) != len(other.starts):
            raise ValueError(
                f'Episodes on {self.dim!r} (n={len(self.starts)}) and {other.dim!r} '
                f'(n={len(other.starts)}) do not share a temporal axis'
            )
        return Episodes(self.starts | other.starts)

    def check(self, dim: str, n: int) -> Episodes:
        """Validate that this partition applies to an axis (dim name and length).

        Args:
            dim: Temporal dim name of the constrained variable.
            n: Length of that dim.

        Raises:
            ValueError: If dim name or length disagree.
        """
        if self.dim != dim or len(self.starts) != n:
            raise ValueError(
                f'Episodes on {self.dim!r} (n={len(self.starts)}) do not match the constraint axis {dim!r} (n={n})'
            )
        return self

    @cached_property
    def flags(self) -> np.ndarray:
        """Boolean start flags as a plain ``(n,)`` array."""
        return self.starts.values.astype(bool)

    @cached_property
    def chain_mask(self) -> xr.DataArray:
        """Boolean (dim[1:],): True where linking t to t-1 stays within an episode."""
        return ~self.starts.isel({self.dim: slice(1, None)})

    @cached_property
    def start_positions(self) -> np.ndarray:
        """Integer positions of episode starts, in order."""
        return np.flatnonzero(self.flags)

    @cached_property
    def last_positions(self) -> np.ndarray:
        """Integer positions of episode ends, in order."""
        return np.append(self.start_positions[1:] - 1, len(self.starts) - 1)

    @cached_property
    def episode_ids(self) -> np.ndarray:
        """0-based episode index per timestep."""
        return np.cumsum(self.flags) - 1

    @property
    def n_episodes(self) -> int:
        """Number of episodes."""
        return int(self.flags.sum())

    def max_duration(self, dt: xr.DataArray) -> float:
        """Longest episode duration in hours — the Big-M for duration tracking.

        Duration chains reset at episode starts, so the longest episode (not
        the whole axis) is the tight M; an axis-length M would loosen the MIP
        relaxation for nothing.

        Args:
            dt: Timestep durations ``(dim,)`` [h].
        """
        return float(np.bincount(self.episode_ids, weights=np.asarray(dt.values, dtype=float)).max())
