"""Storage constraint helpers.

Module-level functions that add accumulation balance constraints
to a linopy Model. Used by FlowSystemModel to build storage features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from linopy import Constraint, Model, Variable
    from linopy.expressions import LinearExpression

    from fluxopt.constraints.episodes import Episodes

__all__ = ['add_accumulation_constraints']


def _slice_dim(obj: Any, dim: str, slc: slice | int | list[int]) -> Any:
    """Slice along dim if present, otherwise return as-is."""
    if isinstance(obj, xr.DataArray) and dim in obj.dims:
        return obj.isel({dim: slc})
    if hasattr(obj, 'dims') and dim in obj.dims:
        return obj.isel({dim: slc})
    return obj


def add_accumulation_constraints(
    m: Model,
    variable: Variable,
    *,
    inflow: LinearExpression | Variable | xr.DataArray | float = 0,
    outflow: LinearExpression | Variable | xr.DataArray | float = 0,
    decay: xr.DataArray | float = 1.0,
    initial: LinearExpression | Variable | xr.DataArray | float | None = None,
    dim: str = 'time',
    name: str = 'accumulation',
    episodes: Episodes,
) -> Constraint | tuple[Constraint, Constraint]:
    """Add state accumulation balance constraints.

    Uses end-of-timestep convention: ``variable[t]`` is the state at the
    END of timestep t. The balance reads::

        variable[t] = variable[t - 1] * decay[t] + inflow[t] - outflow[t]

    The recursion never links across episode boundaries: at each episode
    start the ``initial`` parameter replaces ``variable[t - 1]``. With a
    single episode this is the classic t=0 initial condition; in
    multi-period models each period is an independent episode.

    Args:
        m: Linopy model.
        variable: State variable with a ``dim`` dimension.
        inflow: Additive inflow per timestep (aligned to variable dims).
        outflow: Additive outflow per timestep (aligned to variable dims).
        decay: Multiplicative retention factor per timestep (1 = no loss).
        initial: State before each episode start. If None, episode starts are
            unconstrained. May carry a ``period`` dim — matched to episodes
            in order, which requires exactly one episode per period.
        dim: Temporal dimension name.
        name: Base name for constraints.
        episodes: Episode partition of the ``dim`` axis — the recursion never
            chains across its boundaries. Pass ``Episodes.single(...)`` for
            one uninterrupted chain.

    Returns:
        Single balance constraint if initial is None, otherwise a tuple
        of (initial_constraint, balance_constraint).
    """
    labels = variable.coords[dim].values
    n = len(labels)
    starts = episodes.check(dim, n).flags

    # Balance for t >= 1: variable[t] = variable[t-1] * decay[t] + inflow[t] - outflow[t]
    coords_from_1 = labels[1:]
    curr = variable.isel({dim: slice(1, None)})
    # Reassign prev's coordinates to match curr so linopy can align them
    prev = variable.isel({dim: slice(None, -1)}).assign_coords({dim: coords_from_1})

    decay_t = _slice_dim(decay, dim, slice(1, None))
    inflow_t = _slice_dim(inflow, dim, slice(1, None))
    outflow_t = _slice_dim(outflow, dim, slice(1, None))

    chain_mask = xr.DataArray(~starts[1:], dims=[dim], coords={dim: coords_from_1})
    balance = m.add_constraints(
        curr == prev * decay_t + inflow_t - outflow_t,
        name=f'{name}|balance',
        mask=chain_mask,
    )

    if initial is None:
        return balance

    # Initial constraint at each episode start:
    # variable[s] = initial * decay[s] + inflow[s] - outflow[s]
    start_pos = episodes.start_positions.tolist()
    start_labels = labels[start_pos]
    var_0 = variable.isel({dim: start_pos})
    decay_0 = _slice_dim(decay, dim, start_pos)
    inflow_0 = _slice_dim(inflow, dim, start_pos)
    outflow_0 = _slice_dim(outflow, dim, start_pos)

    init = initial
    if not isinstance(init, (int, float)) and 'period' in init.dims:
        if init.sizes['period'] != episodes.n_episodes:
            raise ValueError(
                f'initial carries {init.sizes["period"]} period entries but the axis has '
                f'{episodes.n_episodes} episodes — per-period initial values require '
                'exactly one episode per period'
            )
        init = init.rename({'period': dim}).assign_coords({dim: start_labels})

    # Put linopy terms (var_0, inflow_0, outflow_0) before pure DataArray
    # terms (initial * decay_0) so linopy's operators handle type coercion.
    init_con = m.add_constraints(
        var_0 == inflow_0 - outflow_0 + init * decay_0,
        name=f'{name}|init',
    )

    return init_con, balance
