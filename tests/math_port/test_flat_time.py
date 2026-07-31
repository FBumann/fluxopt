"""Mathematical correctness tests for the flat time index.

Multi-period models run on one flat time axis with a ``time_period``
coordinate (docs/design/time-index.md). Each period is an independent
operational episode: temporal coupling (storage levels, ramps, status)
never crosses period boundaries, and per-period grids may differ in
resolution and length (ragged periods).
"""

import numpy as np
import pandas as pd
import xarray as xr
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Status, Storage


class TestRaggedPeriods:
    def test_ragged_resolutions_use_per_period_dt(self, optimize):
        """Proves: energy accounting uses each period's own timestep duration.

        2030: 3 hourly steps at 10 MW → 30 MWh. 2040: 2 four-hourly steps at
        5 MW → 40 MWh. Grid @1/MWh, weights [1, 1]. Objective = 30 + 40 = 70.

        Sensitivity: with a shared dt of 1 h, 2040 would count 10 MWh and the
        objective would be 40.
        """
        timesteps = {
            2030: pd.date_range('2030-01-01', periods=3, freq='h'),
            2040: pd.date_range('2040-01-01', periods=2, freq='4h'),
        }
        result = optimize(
            timesteps=timesteps,
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(
                            carrier='Heat',
                            size=1,
                            fixed_relative_profile={2030: [10.0, 10.0, 10.0], 2040: [5.0, 5.0]},
                        ),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
            ],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 70.0, rtol=1e-5)

    def test_ragged_flow_rates_carry_time_period(self, optimize):
        """Proves: solution arrays expose the time_period coordinate for slicing."""
        timesteps = {
            2030: pd.date_range('2030-01-01', periods=3, freq='h'),
            2040: pd.date_range('2040-01-01', periods=2, freq='4h'),
        }
        result = optimize(
            timesteps=timesteps,
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(
                            carrier='Heat',
                            size=1,
                            fixed_relative_profile={2030: [10.0, 10.0, 10.0], 2040: [5.0, 5.0]},
                        ),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
            ],
            period_weights=[1, 1],
        )
        grid = result.flow_rate('Grid(Heat)')
        assert 'time_period' in grid.coords
        rates_2040 = grid.where(grid.time_period == 2040, drop=True)
        assert len(rates_2040) == 2
        assert_allclose(rates_2040.values, [5.0, 5.0], rtol=1e-5)


class TestPeriodBoundaryIsolation:
    def test_storage_never_leaks_across_periods(self, optimize):
        """Proves: storage state does not carry from one period into the next.

        2 uniform periods x 2 h. Grid price 1 in 2030, 100 in 2040 (per-period
        scalar). Demand [0, 10] within each period. Storage available.
        Each period must self-supply: 2030 → 10 MWh @1, 2040 → 10 MWh @100.
        Objective = 10 + 1000 = 1010.

        Sensitivity: if the level chained across the boundary, 2040's demand
        could be served with energy bought in 2030 (objective ≈ 20).
        """
        price = xr.DataArray([1.0, 100.0], dims=['period'], coords={'period': [2030, 2040]})
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[0, 10])]),
                Port(id='Grid', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': price})]),
            ],
            storages=[
                Storage(
                    id='store',
                    charging=Flow(carrier='Heat'),
                    discharging=Flow(carrier='Heat'),
                    capacity=100,
                ),
            ],
            periods=[2030, 2040],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 1010.0, rtol=1e-5)

    def test_ramp_resets_at_period_boundary(self, optimize):
        """Proves: ramp limits do not bind across the period boundary.

        2 uniform periods x 2 h. Demand 0 MW in 2030, 50 MW in 2040. Source
        size=100 with ramp_up 0.1 (Δmax 10 MW/h) — the 0→50 jump at the 2040
        start must be free; within-period steps are flat. @1/MWh the 2040
        energy is 100 MWh. Objective = 100.

        Sensitivity: a ramp binding across the boundary would make the model
        infeasible (50 MW jump ≫ 10 MW/h allowance).
        """
        timesteps = ts(2)
        demand = pd.DataFrame(
            np.array([[0.0, 50.0], [0.0, 50.0]]),
            index=pd.DatetimeIndex(timesteps, name='time'),
            columns=pd.Index([2030, 2040], name='period'),
        )
        result = optimize(
            timesteps=timesteps,
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)]),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', size=100, ramp_up_per_hour=0.1, effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            periods=[2030, 2040],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-5)

    def test_status_prior_state_applies_at_each_period_start(self, optimize):
        """Proves: each period starts from the flow's pre-horizon state.

        prior_rates=[0] → OFF before each period. Demand 10 MW at both steps
        forces ON from the first step, costing one startup per period.
        Startup cost 100, weights [1, 1]. Objective = 2 x 100 = 200.

        Sensitivity: if the ON state chained across the boundary, the 2040
        startup would be free (objective 100).
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=10,
                            relative_rate_min=0.5,
                            prior_rates=[0.0],
                            status=Status(effects_per_startup={'cost': 100}),
                        ),
                    ],
                ),
            ],
            periods=[2030, 2040],
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 200.0, rtol=1e-4)


class TestUniformCalendarShift:
    def test_flat_labels_shift_into_period_years(self, optimize):
        """Proves: uniform replication gives each period real calendar labels.

        ts(2) starts 2024-01-01; periods [2030, 2040] shift the replicas by
        the year gap to the first period: 2030 keeps 2024 labels, 2040 gets
        2034 labels. Datetime features work on the flat axis.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10])]),
                Port(id='Grid', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
            ],
            periods=[2030, 2040],
            period_weights=[1, 1],
        )
        time = pd.DatetimeIndex(result.flow_rates.coords['time'].values)
        assert list(time.year) == [2024, 2024, 2034, 2034]
        assert time.is_monotonic_increasing
        # datetime accessors work directly on solution arrays
        by_year = result.flow_rate('Grid(Heat)').groupby('time.year').sum()
        assert_allclose(by_year.values, [20.0, 20.0], rtol=1e-5)


class TestEpisodeBigM:
    def test_duration_big_m_is_per_episode(self):
        """The duration-tracking big-M spans one episode, not the flat axis.

        2 periods x 3 h: the uptime duration variable's upper bound (= M when
        no uptime_max is set) must be the longest episode (3 h), not the
        6 h flat-axis total — an inflated M loosens the MIP relaxation.
        """
        from fluxopt import FlowSystemModel, ModelData

        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10, 10, 10])]),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', size=10, relative_rate_min=0.5, status=Status(uptime_min=2)),
                    ],
                ),
            ],
            periods=[2030, 2040],
            period_weights=[1, 1],
        )
        fs = FlowSystemModel(data, objective='cost')
        fs.build()
        assert float(fs.m.variables['uptime'].upper.max()) == 3.0
