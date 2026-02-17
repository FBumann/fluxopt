"""Reusable constraint patterns for linopy models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray import DataArray

if TYPE_CHECKING:
    from linopy import Constraint, LinearExpression, Model, Variable


def _isel_if(obj: Any, dim: str, indexer: int | slice) -> Any:
    """Slice along dim if it exists, otherwise return as-is (broadcasts).

    Args:
        obj: A Variable, LinearExpression, DataArray, or scalar.
        dim: Dimension name to slice along.
        indexer: Index or slice to apply.
    """
    if hasattr(obj, 'dims') and dim in obj.dims:
        return obj.isel({dim: indexer})
    return obj


def add_accumulation_constraints(
    model: Model,
    variable: Variable,
    inflow: Variable | LinearExpression,
    outflow: Variable | LinearExpression,
    *,
    loss_factor: float | DataArray = 1.0,
    initial: float | DataArray | Variable | LinearExpression | None = None,
    time_dim: str = 'time',
    name: str = 'accumulation',
) -> Constraint | tuple[Constraint, Constraint]:
    """Add an accumulation (state balance) constraint to a linopy model.

    Uses end-of-period convention: ``variable[t]`` is the state at the
    **end** of period ``t``, after all flows in that period. ``initial``
    is the state at the start of period 0 (before any flows)::

        variable[t] = loss_factor * variable[t - 1] + inflow[t] - outflow[t]

    Args:
        model: The linopy Model to add constraints to.
        variable: State variable (e.g. storage level) with a time dimension.
        inflow: Inflow variable or expression.
        outflow: Outflow variable or expression.
        loss_factor: Proportional retention factor (1.0 = no loss).
        initial: State before period 0 (seeds the recursion at t=0).
            If None, no initial constraint is added.
        time_dim: Name of the time dimension on variable.
        name: Base name for the constraints.

    Returns:
        Single Constraint if initial is None, otherwise a tuple of
        (initial_constraint, balance_constraint).

    Raises:
        ValueError: If time_dim is not in variable.dims.

    Example:
        Storage with charge/discharge efficiencies and standing loss::

            eta_c = 0.95  # charge efficiency
            eta_d = 0.90  # discharge efficiency
            delta = 0.01  # standing loss per timestep

            add_accumulation_constraints(
                model,
                charge_state,
                inflow=eta_c * charge_rate,
                outflow=(1 / eta_d) * discharge_rate,
                loss_factor=1 - delta,
                initial=0.0,
            )
    """
    if time_dim not in variable.dims:
        raise ValueError(f'time_dim {time_dim!r} not in variable.dims {variable.dims}')

    time_vals = variable.coords[time_dim].values
    time_bal = time_vals[1:]

    # Align loss_factor coords with the variable so slicing produces matching coords
    if isinstance(loss_factor, DataArray) and time_dim in loss_factor.dims:
        loss_factor = loss_factor.assign_coords({time_dim: time_vals})

    # Balance constraint: t >= 1
    var_next = variable.isel({time_dim: slice(1, None)})
    var_prev = variable.isel({time_dim: slice(None, -1)})
    var_prev = var_prev.assign_coords({time_dim: time_bal})

    in_bal = _isel_if(inflow, time_dim, slice(1, None))
    out_bal = _isel_if(outflow, time_dim, slice(1, None))
    lf_bal = _isel_if(loss_factor, time_dim, slice(1, None))

    balance = model.add_constraints(var_next == lf_bal * var_prev + in_bal - out_bal, name=name)

    if initial is None:
        return balance

    # Initial constraint: t = 0
    var_0 = variable.isel({time_dim: 0})
    in_0 = _isel_if(inflow, time_dim, 0)
    out_0 = _isel_if(outflow, time_dim, 0)
    lf_0 = _isel_if(loss_factor, time_dim, 0)

    # Put linopy objects first so their operators handle mixed types
    init_con = model.add_constraints(var_0 == in_0 - out_0 + lf_0 * initial, name=f'{name}_initial')

    return init_con, balance
