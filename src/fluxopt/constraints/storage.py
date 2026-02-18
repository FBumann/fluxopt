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

__all__ = ['add_accumulation_constraints']


def _slice_dim(obj: Any, dim: str, slc: slice | int) -> Any:
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
) -> Constraint | tuple[Constraint, Constraint]:
    """Add state accumulation balance constraints.

    Uses end-of-period convention: ``variable[t]`` is the state at the
    END of period t. The balance reads::

        variable[t] = variable[t - 1] * decay[t] + inflow[t] - outflow[t]

    For t=0 the ``initial`` parameter replaces ``variable[t-1]``.

    Args:
        m: Linopy model.
        variable: State variable with a ``dim`` dimension.
        inflow: Additive inflow per period (aligned to variable dims).
        outflow: Additive outflow per period (aligned to variable dims).
        decay: Multiplicative retention factor per period (1 = no loss).
        initial: State before period 0. If None, t=0 is unconstrained.
        dim: Temporal dimension name.
        name: Base name for constraints.

    Returns:
        Single balance constraint if initial is None, otherwise a tuple
        of (initial_constraint, balance_constraint).
    """
    # Coordinates for t >= 1 positions
    coords_from_1 = variable.coords[dim].values[1:]

    # Balance for t >= 1: variable[t] = variable[t-1] * decay[t] + inflow[t] - outflow[t]
    curr = variable.isel({dim: slice(1, None)})
    # Reassign prev's coordinates to match curr so linopy can align them
    prev = variable.isel({dim: slice(None, -1)}).assign_coords({dim: coords_from_1})

    decay_t = _slice_dim(decay, dim, slice(1, None))
    inflow_t = _slice_dim(inflow, dim, slice(1, None))
    outflow_t = _slice_dim(outflow, dim, slice(1, None))

    balance = m.add_constraints(
        curr == prev * decay_t + inflow_t - outflow_t,
        name=f'{name}|balance',
    )

    if initial is None:
        return balance

    # Initial constraint at t=0: variable[0] = initial * decay[0] + inflow[0] - outflow[0]
    var_0 = variable.isel({dim: 0})
    decay_0 = _slice_dim(decay, dim, 0)
    inflow_0 = _slice_dim(inflow, dim, 0)
    outflow_0 = _slice_dim(outflow, dim, 0)

    # Put linopy terms (var_0, inflow_0, outflow_0) before pure DataArray
    # terms (initial * decay_0) so linopy's operators handle type coercion.
    init_con = m.add_constraints(
        var_0 == inflow_0 - outflow_0 + initial * decay_0,
        name=f'{name}|init',
    )

    return init_con, balance
