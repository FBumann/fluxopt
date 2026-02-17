"""Tests for fluxopt.constraints.

Strategy: Each constraint helper is tested by building a minimal linopy
model, fixing inputs to known values, solving, and checking that the
solution matches hand-computed expectations. This verifies the constraint
wiring (slicing, coordinate alignment, operator ordering) end-to-end.
Tests vary one parameter at a time against a simple baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from linopy import Model

from fluxopt.constraints2 import (
    add_accumulation_constraints,
    add_min_downtime_constraint,
    add_min_uptime_constraint,
)


@pytest.fixture
def model_with_stock():
    """Minimal linopy model with stock, inflow, and outflow variables."""
    m = Model()
    time = xr.DataArray([0, 1, 2, 3], dims=['time'])
    stock = m.add_variables(lower=0, coords=[time], name='stock')
    inflow = m.add_variables(lower=0, coords=[time], name='inflow')
    outflow = m.add_variables(lower=0, coords=[time], name='outflow')
    return m, stock, inflow, outflow


class TestAddAccumulationConstraint:
    def test_with_initial_and_loss(self, model_with_stock):
        """Solve a known accumulation and check stock values."""
        m, stock, inflow, outflow = model_with_stock

        # Fix inflow=10 every step, outflow=3 every step
        m.add_constraints(inflow == 10, name='fix_in')
        m.add_constraints(outflow == 3, name='fix_out')

        loss_factor = 0.9
        initial = 100.0

        init_con, balance = add_accumulation_constraints(
            m,
            stock,
            inflow,
            outflow,
            loss_factor=loss_factor,
            initial=initial,
            name='stock_bal',
        )

        assert init_con is not None
        assert balance is not None

        m.add_objective(stock.sum())
        m.solve(solver_name='highs', io_api='direct')

        s = stock.solution.values
        # t=0: 0.9 * 100 + 10 - 3 = 97
        # t=1: 0.9 * 97  + 10 - 3 = 94.3
        # t=2: 0.9 * 94.3 + 10 - 3 = 91.87
        # t=3: 0.9 * 91.87 + 10 - 3 = 89.683
        expected = [97.0, 94.3, 91.87, 89.683]
        np.testing.assert_allclose(s, expected, atol=1e-6)

    def test_without_initial(self, model_with_stock):
        """Without initial, only balance constraint is returned (t >= 1)."""
        m, stock, inflow, outflow = model_with_stock

        m.add_constraints(inflow == 5, name='fix_in')
        m.add_constraints(outflow == 0, name='fix_out')

        result = add_accumulation_constraints(
            m,
            stock,
            inflow,
            outflow,
            name='acc',
        )

        # Should be a single Constraint, not a tuple
        from linopy import Constraint

        assert isinstance(result, Constraint)

        # Pin stock[0] externally
        m.add_constraints(stock.isel(time=0) == 10, name='pin_start')
        m.add_objective(stock.sum())
        m.solve(solver_name='highs', io_api='direct')

        s = stock.solution.values
        # t=0: 10 (pinned), t=1: 10+5=15, t=2: 15+5=20, t=3: 20+5=25
        np.testing.assert_allclose(s, [10, 15, 20, 25], atol=1e-6)

    def test_no_loss(self, model_with_stock):
        """Default loss_factor=1.0 means no loss."""
        m, stock, inflow, outflow = model_with_stock

        m.add_constraints(inflow == 4, name='fix_in')
        m.add_constraints(outflow == 1, name='fix_out')

        add_accumulation_constraints(
            m,
            stock,
            inflow,
            outflow,
            initial=0.0,
            name='bal',
        )

        m.add_objective(stock.sum())
        m.solve(solver_name='highs', io_api='direct')

        s = stock.solution.values
        # Each step adds net 3: [3, 6, 9, 12]
        np.testing.assert_allclose(s, [3, 6, 9, 12], atol=1e-6)

    def test_time_varying_loss_factor(self, model_with_stock):
        """Loss factor as a DataArray varying over time."""
        m, stock, inflow, outflow = model_with_stock

        m.add_constraints(inflow == 10, name='fix_in')
        m.add_constraints(outflow == 0, name='fix_out')

        lf = xr.DataArray([1.0, 0.5, 1.0, 0.5], dims=['time'])

        add_accumulation_constraints(
            m,
            stock,
            inflow,
            outflow,
            loss_factor=lf,
            initial=20.0,
            name='bal',
        )

        m.add_objective(stock.sum())
        m.solve(solver_name='highs', io_api='direct')

        s = stock.solution.values
        # t=0: 1.0*20 + 10 = 30
        # t=1: 0.5*30 + 10 = 25
        # t=2: 1.0*25 + 10 = 35
        # t=3: 0.5*35 + 10 = 27.5
        np.testing.assert_allclose(s, [30, 25, 35, 27.5], atol=1e-6)

    def test_invalid_time_dim(self, model_with_stock):
        """Raise ValueError if time_dim not in variable.dims."""
        m, stock, inflow, outflow = model_with_stock

        with pytest.raises(ValueError, match=r"time_dim 'wrong' not in variable\.dims"):
            add_accumulation_constraints(
                m,
                stock,
                inflow,
                outflow,
                time_dim='wrong',
            )

    def test_initial_as_variable(self):
        """Initial can be a linopy Variable."""
        m = Model()
        time = xr.DataArray([0, 1, 2], dims=['time'])
        stock = m.add_variables(lower=0, coords=[time], name='stock')
        inflow = m.add_variables(lower=0, coords=[time], name='inflow')
        outflow = m.add_variables(lower=0, coords=[time], name='outflow')
        init_var = m.add_variables(lower=50, upper=50, name='init_level')

        m.add_constraints(inflow == 0, name='fix_in')
        m.add_constraints(outflow == 0, name='fix_out')

        add_accumulation_constraints(
            m,
            stock,
            inflow,
            outflow,
            initial=init_var,
            name='bal',
        )

        m.add_objective(stock.sum())
        m.solve(solver_name='highs', io_api='direct')

        s = stock.solution.values
        np.testing.assert_allclose(s, [50, 50, 50], atol=1e-6)


class TestMinUptime:
    def test_basic_min_uptime(self):
        """Unit starts off, forced startup at t=2, must stay on for 3 periods."""
        m = Model()
        time = xr.DataArray(range(6), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')

        # Transition: on[t] = on[t-1] + startup[t] - shutdown[t], initial=0
        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=0,
            name='transition',
        )

        # Force startup at t=2
        m.add_constraints(startup.sel(time=2) >= 1, name='force_start')

        # Min uptime = 3
        add_min_uptime_constraint(m, on, startup, periods=3)

        # Minimize on-time → unit turns off as soon as allowed
        m.add_objective(on.sum())
        m.solve(solver_name='highs', io_api='direct')

        np.testing.assert_allclose(
            on.solution.values,
            [0, 0, 1, 1, 1, 0],
            atol=1e-6,
        )

    def test_initial_periods(self):
        """Unit was on for 1 period, min_up=3, must stay on for 2 more."""
        m = Model()
        time = xr.DataArray(range(4), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=1,
            name='transition',
        )

        cons = add_min_uptime_constraint(
            m,
            on,
            startup,
            periods=3,
            initial_periods=1,
        )

        assert len(cons) == 2  # rolling + initial pin

        m.add_objective(on.sum())
        m.solve(solver_name='highs', io_api='direct')

        # Must stay on for 2 more periods (remain = 3 - 1 = 2)
        np.testing.assert_allclose(
            on.solution.values,
            [1, 1, 0, 0],
            atol=1e-6,
        )

    def test_no_initial_pinning(self):
        """Default initial_periods=0 returns a single constraint."""
        m = Model()
        time = xr.DataArray(range(4), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')

        result = add_min_uptime_constraint(m, on, startup, periods=2)

        assert len(result) == 1

    def test_batched_heterogeneous(self):
        """Two elements with different min uptime: A=2, B=3."""
        m = Model()
        time = xr.DataArray(range(8), dims=['time'])
        elem = xr.DataArray(['A', 'B'], dims=['elem'])
        on = m.add_variables(binary=True, coords=[elem, time], name='on')
        startup = m.add_variables(binary=True, coords=[elem, time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[elem, time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=0,
            name='transition',
        )
        m.add_constraints(startup.sel(time=2) >= 1, name='force_start')

        periods = xr.DataArray([2, 3], dims=['elem'], coords={'elem': ['A', 'B']})
        add_min_uptime_constraint(m, on, startup, periods)

        m.add_objective(on.sum())
        m.solve(solver_name='highs', io_api='direct')

        # A stays on for 2 periods (t=2,3), B for 3 (t=2,3,4)
        np.testing.assert_allclose(
            on.solution.sel(elem='A').values,
            [0, 0, 1, 1, 0, 0, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            on.solution.sel(elem='B').values,
            [0, 0, 1, 1, 1, 0, 0, 0],
            atol=1e-6,
        )

    def test_batched_initial_periods(self):
        """Batched with per-element initial_periods: A=1, B=1; min_up A=3, B=4."""
        m = Model()
        time = xr.DataArray(range(6), dims=['time'])
        elem = xr.DataArray(['A', 'B'], dims=['elem'])
        on = m.add_variables(binary=True, coords=[elem, time], name='on')
        startup = m.add_variables(binary=True, coords=[elem, time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[elem, time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=1,
            name='transition',
        )

        periods = xr.DataArray([3, 4], dims=['elem'], coords={'elem': ['A', 'B']})
        initial = xr.DataArray([1, 1], dims=['elem'], coords={'elem': ['A', 'B']})
        add_min_uptime_constraint(m, on, startup, periods, initial_periods=initial)

        m.add_objective(on.sum())
        m.solve(solver_name='highs', io_api='direct')

        # A: remain=2 → on[0],on[1] pinned; B: remain=3 → on[0],on[1],on[2] pinned
        np.testing.assert_allclose(
            on.solution.sel(elem='A').values,
            [1, 1, 0, 0, 0, 0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            on.solution.sel(elem='B').values,
            [1, 1, 1, 0, 0, 0],
            atol=1e-6,
        )


class TestMinDowntime:
    def test_basic_min_downtime(self):
        """Unit starts on, forced shutdown at t=2, must stay off for 3 periods."""
        m = Model()
        time = xr.DataArray(range(6), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=1,
            name='transition',
        )

        m.add_constraints(shutdown.sel(time=2) >= 1, name='force_stop')

        add_min_downtime_constraint(m, on, shutdown, periods=3)

        # Maximize on-time → unit turns back on as soon as allowed
        m.add_objective(-on.sum())
        m.solve(solver_name='highs', io_api='direct')

        np.testing.assert_allclose(
            on.solution.values,
            [1, 1, 0, 0, 0, 1],
            atol=1e-6,
        )

    def test_initial_periods(self):
        """Unit was off for 1 period, min_down=3, must stay off for 2 more."""
        m = Model()
        time = xr.DataArray(range(4), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=0,
            name='transition',
        )

        cons = add_min_downtime_constraint(
            m,
            on,
            shutdown,
            periods=3,
            initial_periods=1,
        )

        assert len(cons) == 2  # rolling + initial pin

        # Maximize on-time
        m.add_objective(-on.sum())
        m.solve(solver_name='highs', io_api='direct')

        np.testing.assert_allclose(
            on.solution.values,
            [0, 0, 1, 1],
            atol=1e-6,
        )


class TestTransitionViaAccumulation:
    def test_transition_is_accumulation(self):
        """Transition on[t] = on[t-1] + startup[t] - shutdown[t] via accumulation."""
        m = Model()
        time = xr.DataArray(range(4), dims=['time'])
        on = m.add_variables(binary=True, coords=[time], name='on')
        startup = m.add_variables(binary=True, coords=[time], name='startup')
        shutdown = m.add_variables(binary=True, coords=[time], name='shutdown')

        add_accumulation_constraints(
            m,
            on,
            startup,
            shutdown,
            initial=0,
            name='transition',
        )

        # Mutual exclusion: can't start and stop in the same period
        m.add_constraints(startup + shutdown <= 1, name='mutex')

        # Fix startup at t=1, shutdown at t=3
        m.add_constraints(startup.sel(time=1) >= 1, name='start_t1')
        m.add_constraints(shutdown.sel(time=3) >= 1, name='stop_t3')

        m.add_objective(on.sum())
        m.solve(solver_name='highs', io_api='direct')

        # on = [0, 1, 1, 0]
        np.testing.assert_allclose(
            on.solution.values,
            [0, 1, 1, 0],
            atol=1e-6,
        )
