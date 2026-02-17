"""Reusable constraint patterns for linopy models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray import DataArray, full_like

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


def _add_min_duration_constraints(
    model: Model,
    on: Variable,
    transition: Variable | LinearExpression,
    periods: int | DataArray,
    initial_periods: int | DataArray,
    time_dim: str,
    name: str,
    *,
    is_uptime: bool,
) -> list[Constraint]:
    """Shared implementation for min uptime/downtime rolling-window constraints.

    Args:
        model: The linopy Model to add constraints to.
        on: Binary status variable (1=running).
        transition: Startup (uptime) or shutdown (downtime) indicator.
        periods: Minimum consecutive periods. Scalar or per-element DataArray.
        initial_periods: Periods already in state before t=0.
        time_dim: Name of the time dimension.
        name: Base name for the constraints.
        is_uptime: True for uptime (rolling <= on), False for downtime
            (rolling <= 1 - on).
    """
    time_vals = on.coords[time_dim].values
    constraints: list[Constraint] = []

    if isinstance(periods, int):
        # Uniform: single rolling window over all elements
        rolling = transition.rolling({time_dim: periods}).sum()
        valid = time_vals[periods - 1 :]
        on_valid = on.sel({time_dim: valid})
        if is_uptime:
            constraints.append(
                model.add_constraints(
                    rolling.sel({time_dim: valid}) <= on_valid,
                    name=name,
                )
            )
        else:
            constraints.append(
                model.add_constraints(
                    rolling.sel({time_dim: valid}) + on_valid <= 1,
                    name=name,
                )
            )

        # Initial pinning (scalar)
        if isinstance(initial_periods, int):
            remain = max(0, periods - initial_periods)
            if remain > 0 and initial_periods > 0:
                on_pin = on.sel({time_dim: time_vals[:remain]})
                if is_uptime:
                    constraints.append(
                        model.add_constraints(
                            on_pin >= 1,
                            name=f'{name}_initial',
                        )
                    )
                else:
                    constraints.append(
                        model.add_constraints(
                            on_pin <= 0,
                            name=f'{name}_initial',
                        )
                    )
    else:
        # Heterogeneous batched: group by period value
        element_dim = next(iter(periods.dims))
        unique_windows = sorted({int(v) for v in periods.values if v > 0})
        suffix = len(unique_windows) > 1

        for window in unique_windows:
            elem_ids = periods.coords[element_dim].values[periods.values == window]
            rolling = transition.sel({element_dim: elem_ids}).rolling({time_dim: window}).sum()
            valid = time_vals[window - 1 :]
            on_valid = on.sel({element_dim: elem_ids, time_dim: valid})
            con_name = f'{name}_{window}' if suffix else name
            if is_uptime:
                constraints.append(
                    model.add_constraints(
                        rolling.sel({time_dim: valid}) <= on_valid,
                        name=con_name,
                    )
                )
            else:
                constraints.append(
                    model.add_constraints(
                        rolling.sel({time_dim: valid}) + on_valid <= 1,
                        name=con_name,
                    )
                )

        # Initial pinning (per-element)
        init_da = full_like(periods, initial_periods) if isinstance(initial_periods, int) else initial_periods
        remain_da = (periods - init_da).clip(min=0)
        needs_pin = (remain_da > 0) & (init_da > 0)

        if needs_pin.any().item():
            unique_r = sorted({int(v) for v in remain_da.values[needs_pin.values]})
            pin_suffix = len(unique_r) > 1
            for r in unique_r:
                elem_ids = periods.coords[element_dim].values[(remain_da.values == r) & needs_pin.values]
                on_pin = on.sel(
                    {
                        element_dim: elem_ids,
                        time_dim: time_vals[:r],
                    }
                )
                con_name = f'{name}_initial_{r}' if pin_suffix else f'{name}_initial'
                if is_uptime:
                    constraints.append(
                        model.add_constraints(
                            on_pin >= 1,
                            name=con_name,
                        )
                    )
                else:
                    constraints.append(
                        model.add_constraints(
                            on_pin <= 0,
                            name=con_name,
                        )
                    )

    return constraints


def add_min_uptime_constraint(
    model: Model,
    on: Variable,
    startup: Variable | LinearExpression,
    periods: int | DataArray,
    *,
    initial_periods: int | DataArray = 0,
    time_dim: str = 'time',
    name: str = 'min_uptime',
) -> list[Constraint]:
    """Add minimum uptime constraints to a linopy model.

    If the unit starts up at time t, it must remain on for at least
    ``periods`` consecutive timesteps (rolling-window formulation)::

        sum(startup[τ] for τ in [t-L+1, t]) <= on[t]    ∀ t ≥ L-1

    Supports batched elements with per-element periods when ``periods``
    is a DataArray indexed by an element dimension.

    Args:
        model: The linopy Model to add constraints to.
        on: Binary status variable (1=running).
        startup: Startup indicator variable or expression.
        periods: Minimum consecutive on-periods. Scalar int for uniform,
            or DataArray indexed by element dim for per-element values.
        initial_periods: Periods already on before t=0.
        time_dim: Name of the time dimension.
        name: Base name for the constraints.

    Returns:
        List of constraints added to the model.
    """
    return _add_min_duration_constraints(
        model,
        on,
        startup,
        periods,
        initial_periods,
        time_dim,
        name,
        is_uptime=True,
    )


def add_min_downtime_constraint(
    model: Model,
    on: Variable,
    shutdown: Variable | LinearExpression,
    periods: int | DataArray,
    *,
    initial_periods: int | DataArray = 0,
    time_dim: str = 'time',
    name: str = 'min_downtime',
) -> list[Constraint]:
    """Add minimum downtime constraints to a linopy model.

    If the unit shuts down at time t, it must remain off for at least
    ``periods`` consecutive timesteps (rolling-window formulation)::

        sum(shutdown[τ] for τ in [t-L+1, t]) <= 1 - on[t]    ∀ t ≥ L-1

    Supports batched elements with per-element periods when ``periods``
    is a DataArray indexed by an element dimension.

    Args:
        model: The linopy Model to add constraints to.
        on: Binary status variable (1=running).
        shutdown: Shutdown indicator variable or expression.
        periods: Minimum consecutive off-periods. Scalar int for uniform,
            or DataArray indexed by element dim for per-element values.
        initial_periods: Periods already off before t=0.
        time_dim: Name of the time dimension.
        name: Base name for the constraints.

    Returns:
        List of constraints added to the model.
    """
    return _add_min_duration_constraints(
        model,
        on,
        shutdown,
        periods,
        initial_periods,
        time_dim,
        name,
        is_uptime=False,
    )
