"""Status (on/off) constraint helpers.

Module-level functions that add binary status tracking constraints
to a linopy Model. Used by FlowSystemModel to build status features.

All temporal chains (duration tracking, switch transitions) respect
episode boundaries: in multi-period models each period is an independent
episode and no constraint links its first timestep to the previous
period's last.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    import numpy as np
    from linopy import Model, Variable
    from linopy.expressions import LinearExpression

    from fluxopt.constraints.episodes import Episodes

__all__ = [
    'add_duration_tracking',
    'add_switch_transitions',
    'compute_previous_duration',
]


def compute_previous_duration(
    previous_status: xr.DataArray,
    target_state: int,
    dt: xr.DataArray | float,
) -> float:
    """Compute consecutive duration of target_state at end of previous_status.

    Walks backward through previous_status counting timesteps that match
    the target state, then multiplies by timestep duration.

    Args:
        previous_status: Previous status values (time dimension).
        target_state: 1 for active (uptime), 0 for inactive (downtime).
        dt: Duration per timestep (scalar or DataArray).

    Returns:
        Total duration in target state at end of previous period.
    """
    values = previous_status.values
    count = 0
    for v in reversed(values):
        if (target_state == 1 and v > 0) or (target_state == 0 and v == 0):
            count += 1
        else:
            break

    if isinstance(dt, xr.DataArray):
        return float(dt.values[-count:].sum()) if count > 0 else 0.0
    return dt * count


def add_duration_tracking(
    m: Model,
    state: Variable | LinearExpression,
    dt: xr.DataArray,
    *,
    name: str,
    element_dim: str = 'flow',
    dim: str = 'time',
    minimum: xr.DataArray | None = None,
    maximum: xr.DataArray | None = None,
    previous: xr.DataArray | None = None,
    episodes: Episodes,
) -> Variable:
    """Add consecutive duration tracking for a binary state variable.

    Uses Big-M formulation to track how long each element has been
    continuously in the given state. Tracking resets at episode starts;
    ``previous`` applies at every episode start (each period sees the same
    pre-horizon history).

    Args:
        m: Linopy model to add constraints to.
        state: Binary state variable with (element_dim, time) dims.
        dt: Timestep durations (time,).
        name: Base name for created variables and constraints.
        element_dim: Element dimension name in state.
        dim: Temporal dimension name.
        minimum: Minimum duration per element. NaN = no constraint.
        maximum: Maximum duration per element. NaN = no constraint.
        previous: Previous duration per element. NaN = no previous.
        episodes: Episode partition of the ``dim`` axis — tracking resets at
            its boundaries. Pass ``Episodes.single(...)`` for one
            uninterrupted chain.

    Returns:
        Duration variable with same dims as state.
    """
    element_ids: xr.DataArray = state.coords[element_dim]
    labels = state.coords[dim].values
    starts = episodes.check(dim, len(labels)).flags
    chain_mask = xr.DataArray(~starts[1:], dims=[dim], coords={dim: labels[1:]})

    mega = episodes.max_duration(dt)
    if previous is not None:
        mega = mega + previous.fillna(0)

    # Variable upper bound: use maximum where provided, else mega
    upper: xr.DataArray | float = maximum.where(maximum.notnull(), mega) if maximum is not None else mega

    coords = [state.indexes[element_dim], state.indexes[dim]]
    duration = m.add_variables(lower=0, upper=upper, coords=coords, name=name)

    # duration[e,t] <= state[e,t] * M[e]
    m.add_constraints(duration <= state * mega, name=f'{name}|ub')

    curr = duration.isel({dim: slice(1, None)})
    prev = duration.isel({dim: slice(None, -1)}).assign_coords({dim: labels[1:]})
    dt_step = dt.isel({dim: slice(1, None)})

    # Forward: duration[e,t+1] <= duration[e,t] + dt[t+1] — within episodes only
    m.add_constraints(curr <= prev + dt_step, name=f'{name}|fwd', mask=chain_mask)

    # Backward: duration[e,t+1] >= duration[e,t] + dt[t+1] + (state[e,t+1] - 1) * M[e]
    m.add_constraints(
        curr >= prev + dt_step + (state.isel({dim: slice(1, None)}) - 1) * mega,
        name=f'{name}|bwd',
        mask=chain_mask,
    )

    # Initial constraints for elements with previous duration
    if previous is not None:
        has_prev = previous.notnull()
        if has_prev.any():
            prev_ids = list(element_ids.values[has_prev.values])
            _add_initial_constraints(
                m,
                state,
                duration,
                dt,
                previous=previous.sel({element_dim: prev_ids}),
                minimum=minimum.sel({element_dim: prev_ids}) if minimum is not None else None,
                name=name,
                dim=dim,
                element_dim=element_dim,
                start_positions=episodes.start_positions.tolist(),
            )

    # Minimum duration: duration[t] >= min * (state[t] - state[t+1])
    if minimum is not None:
        _add_minimum_constraints(m, state, duration, minimum, name, dim, element_dim, starts)

    return duration


def _add_initial_constraints(
    m: Model,
    state: Variable | LinearExpression,
    duration: Variable,
    dt: xr.DataArray,
    previous: xr.DataArray,
    minimum: xr.DataArray | None,
    name: str,
    dim: str,
    element_dim: str,
    start_positions: list[int],
) -> None:
    """Add initial duration constraints at each episode start.

    Args:
        m: Linopy model.
        state: Binary state variable.
        duration: Duration variable.
        dt: Timestep durations.
        previous: Previous duration per element (pre-filtered, no NaN).
        minimum: Minimum duration per element (pre-filtered to match previous).
        name: Base constraint name.
        dim: Temporal dimension name.
        element_dim: Element dimension name.
        start_positions: Positions of episode starts.
    """
    ids = list(previous.coords[element_dim].values)
    state_0 = state.sel({element_dim: ids}).isel({dim: start_positions})
    dur_0 = duration.sel({element_dim: ids}).isel({dim: start_positions})
    dt_0 = dt.isel({dim: start_positions})

    # duration[s] = state[s] * (previous + dt[s]) at each episode start
    m.add_constraints(dur_0 == state_0 * (previous + dt_0), name=f'{name}|init')

    # Force continuation if previous duration < minimum
    if minimum is not None:
        needs_cont = (previous > 0) & minimum.notnull() & (previous < minimum)
        if needs_cont.any():
            cont_ids = list(previous.coords[element_dim].values[needs_cont.values])
            m.add_constraints(
                state.sel({element_dim: cont_ids}).isel({dim: start_positions}) >= 1,
                name=f'{name}|init_cont',
            )


def _add_minimum_constraints(
    m: Model,
    state: Variable | LinearExpression,
    duration: Variable,
    minimum: xr.DataArray,
    name: str,
    dim: str,
    element_dim: str,
    starts: np.ndarray,
) -> None:
    """Add minimum duration constraints on state transitions.

    Args:
        m: Linopy model.
        state: Binary state variable.
        duration: Duration variable.
        minimum: Minimum duration per element.
        name: Base constraint name.
        dim: Temporal dimension name.
        element_dim: Element dimension name.
        starts: Boolean per-position episode start flags.
    """
    has_min = minimum.notnull()
    if not has_min.any():
        return

    min_ids = minimum.coords[element_dim].values[has_min.values]
    min_sub = minimum.sel({element_dim: min_ids})
    state_sub = state.sel({element_dim: min_ids})
    dur_sub = duration.sel({element_dim: min_ids})

    # duration[t] >= min * (state[t] - state[t+1]) — only for within-episode pairs
    labels = state.coords[dim].values
    pair_mask = xr.DataArray(~starts[1:], dims=[dim], coords={dim: labels[:-1]})
    state_diff = state_sub.isel({dim: slice(None, -1)}) - state_sub.isel({dim: slice(1, None)})
    m.add_constraints(
        dur_sub.isel({dim: slice(None, -1)}) >= min_sub * state_diff,
        name=f'{name}|min',
        mask=pair_mask,
    )


def add_switch_transitions(
    m: Model,
    status: Variable,
    startup: Variable,
    shutdown: Variable,
    *,
    name: str,
    element_dim: str = 'flow',
    dim: str = 'time',
    previous_state: xr.DataArray | None = None,
    episodes: Episodes,
) -> None:
    """Add startup/shutdown transition constraints.

    Links status changes to startup and shutdown indicator variables:
    ``startup[t] - shutdown[t] == status[t] - status[t-1]``, within episodes.
    ``previous_state`` pins the transition at each episode start.

    Args:
        m: Linopy model.
        status: Binary on/off variable.
        startup: Binary startup indicator variable.
        shutdown: Binary shutdown indicator variable.
        name: Base name for constraints.
        element_dim: Element dimension name in status.
        dim: Temporal dimension name.
        previous_state: Previous on/off per element (pre-filtered, no NaN).
        episodes: Episode partition of the ``dim`` axis — transitions never
            link across its boundaries. Pass ``Episodes.single(...)`` for one
            uninterrupted chain.
    """
    labels = status.coords[dim].values
    starts = episodes.check(dim, len(labels)).flags
    chain_mask = xr.DataArray(~starts[1:], dims=[dim], coords={dim: labels[1:]})

    # Transition within episodes (t not an episode start)
    m.add_constraints(
        startup.isel({dim: slice(1, None)}) - shutdown.isel({dim: slice(1, None)})
        == status.isel({dim: slice(1, None)}) - status.isel({dim: slice(None, -1)}),
        name=f'{name}|transition',
        mask=chain_mask,
    )

    # At most one of startup/shutdown per step. Together with the transition
    # equality this pins both binaries to actual state changes — without it,
    # a spurious (1, 1) pair satisfies the equality on non-transition steps
    # and could be exploited by constraints relaxed via startup (e.g. ramps).
    m.add_constraints(startup + shutdown <= 1, name=f'{name}|exclusive')

    # Initial transition from previous state, at each episode start
    if previous_state is not None:
        start_positions = episodes.start_positions.tolist()
        ids = list(previous_state.coords[element_dim].values)
        m.add_constraints(
            startup.sel({element_dim: ids}).isel({dim: start_positions})
            - shutdown.sel({element_dim: ids}).isel({dim: start_positions})
            == status.sel({element_dim: ids}).isel({dim: start_positions}) - previous_state,
            name=f'{name}|init',
        )
