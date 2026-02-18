"""Status (on/off) constraint helpers.

Module-level functions that add binary status tracking constraints
to a linopy Model. Used by FlowSystem to build status features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from linopy import Model, Variable
    from linopy.expressions import LinearExpression

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
        return float(dt.mean()) * count
    return float(dt) * count


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
) -> Variable:
    """Add consecutive duration tracking for a binary state variable.

    Uses Big-M formulation to track how long each element has been
    continuously in the given state.

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

    Returns:
        Duration variable with same dims as state.
    """
    element_ids: xr.DataArray = state.coords[element_dim]

    # Big-M per element: total horizon + any previous carryover
    mega = dt.sum(dim)
    if previous is not None:
        mega = mega + previous.fillna(0)

    # Variable upper bound: use maximum where provided, else mega
    upper: xr.DataArray = xr.where(maximum.notnull(), maximum, mega) if maximum is not None else mega

    coords = [state.coords[element_dim], state.coords[dim]]
    duration = m.add_variables(lower=0, upper=upper, coords=coords, name=name)

    # duration[e,t] <= state[e,t] * M[e]
    m.add_constraints(duration <= state * mega, name=f'{name}|ub')

    # Forward: duration[e,t+1] <= duration[e,t] + dt[t]
    m.add_constraints(
        duration.isel({dim: slice(1, None)}) <= duration.isel({dim: slice(None, -1)}) + dt.isel({dim: slice(None, -1)}),
        name=f'{name}|fwd',
    )

    # Backward: duration[e,t+1] >= duration[e,t] + dt[t] + (state[e,t+1] - 1) * M[e]
    m.add_constraints(
        duration.isel({dim: slice(1, None)})
        >= duration.isel({dim: slice(None, -1)})
        + dt.isel({dim: slice(None, -1)})
        + (state.isel({dim: slice(1, None)}) - 1) * mega,
        name=f'{name}|bwd',
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
            )

    # Minimum duration: duration[t] >= min * (state[t] - state[t+1])
    if minimum is not None:
        _add_minimum_constraints(m, state, duration, minimum, name, dim, element_dim)

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
) -> None:
    """Add initial duration constraints from previous period.

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
    """
    ids = list(previous.coords[element_dim].values)
    state_0 = state.sel({element_dim: ids}).isel({dim: 0})
    dur_0 = duration.sel({element_dim: ids}).isel({dim: 0})
    dt_0 = dt.isel({dim: 0})

    # duration[0] = state[0] * (previous + dt[0])
    m.add_constraints(dur_0 == state_0 * (previous + dt_0), name=f'{name}|init')

    # Force continuation if previous duration < minimum
    if minimum is not None:
        needs_cont = (previous > 0) & minimum.notnull() & (previous < minimum)
        if needs_cont.any():
            cont_ids = list(previous.coords[element_dim].values[needs_cont.values])
            m.add_constraints(
                state.sel({element_dim: cont_ids}).isel({dim: 0}) >= 1,
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
    """
    has_min = minimum.notnull()
    if not has_min.any():
        return

    min_ids = minimum.coords[element_dim].values[has_min.values]
    min_sub = minimum.sel({element_dim: min_ids})
    state_sub = state.sel({element_dim: min_ids})
    dur_sub = duration.sel({element_dim: min_ids})

    # duration[t] >= min * (state[t] - state[t+1])
    state_diff = state_sub.isel({dim: slice(None, -1)}) - state_sub.isel({dim: slice(1, None)})
    m.add_constraints(
        dur_sub.isel({dim: slice(None, -1)}) >= min_sub * state_diff,
        name=f'{name}|min',
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
) -> None:
    """Add startup/shutdown transition constraints.

    Links status changes to startup and shutdown indicator variables:
    ``startup[t] - shutdown[t] == status[t] - status[t-1]``.

    Args:
        m: Linopy model.
        status: Binary on/off variable.
        startup: Binary startup indicator variable.
        shutdown: Binary shutdown indicator variable.
        name: Base name for constraints.
        element_dim: Element dimension name in status.
        dim: Temporal dimension name.
        previous_state: Previous on/off per element (pre-filtered, no NaN).
    """
    # Transition for t > 0
    m.add_constraints(
        startup.isel({dim: slice(1, None)}) - shutdown.isel({dim: slice(1, None)})
        == status.isel({dim: slice(1, None)}) - status.isel({dim: slice(None, -1)}),
        name=f'{name}|transition',
    )

    # Initial transition from previous state
    if previous_state is not None:
        ids = list(previous_state.coords[element_dim].values)
        m.add_constraints(
            startup.sel({element_dim: ids}).isel({dim: 0}) - shutdown.sel({element_dim: ids}).isel({dim: 0})
            == status.sel({element_dim: ids}).isel({dim: 0}) - previous_state,
            name=f'{name}|init',
        )
