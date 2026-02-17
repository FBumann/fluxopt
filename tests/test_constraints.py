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

from fluxopt.constraints import add_accumulation_constraints


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
