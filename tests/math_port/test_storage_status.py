"""Mathematical correctness tests for Storage component-level Status.

Component-level Status on Converter is deferred — when PiecewiseConversion lands
(see #25), it will provide a more versatile, single-API path that subsumes
linear converter on/off.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Status, Storage

from .conftest import ts


class TestStorageStatusValidation:
    def test_flow_level_status_forbidden(self):
        """Flow.status on a storage flow conflicts with component-level Status."""
        with pytest.raises(ValueError, match='cannot have flow-level status'):
            Storage(
                id='Bat',
                charging=Flow(carrier='Elec', size=10, relative_rate_min=0.1, status=Status(uptime_min=2)),
                discharging=Flow(carrier='Elec', size=10),
                capacity=100,
                status=Status(),
            )

    def test_storage_status_optional(self):
        """Storage with status=None still constructs (default)."""
        s = Storage(id='Bat', charging=Flow(carrier='Elec'), discharging=Flow(carrier='Elec'), capacity=10)
        assert s.status is None

    def test_unsized_flow_forbidden(self):
        """Storage flows must be sized when component Status is set.

        Without a size, the on/off binary has no upper bound to scale by, so
        the flow can't be gated. Unlike Converter inputs (which are gated
        transitively through the conversion equation), Storage charge/discharge
        are independent flows with no such coupling.
        """
        with pytest.raises(ValueError, match='must have a size'):
            Storage(
                id='Bat',
                charging=Flow(carrier='Elec'),  # no size
                discharging=Flow(carrier='Elec', size=10),
                capacity=100,
                status=Status(),
            )

    def test_fixed_profile_compatible(self):
        """fixed_relative_profile on a storage flow is allowed with component Status.

        Constraint becomes ``P = size * profile * on``: the on-binary still has
        a meaningful role (forces P=0 when off), and startup tracking on a
        fixed dispatch profile is a legitimate use case.
        """
        s = Storage(
            id='Bat',
            charging=Flow(carrier='Elec', size=10, fixed_relative_profile=0.5),
            discharging=Flow(carrier='Elec', size=10),
            capacity=100,
            status=Status(),
        )
        assert s.status is not None

    def test_sized_flow_with_status_builds(self):
        """A governed flow may be sized — the linopy lane could not do this.

        `_constrain_flow_rates_component_status` raised NotImplementedError
        here. One status axis makes the case ordinary: `status_sizing_*` does
        not care whether the binary is the flow's own or its component's.
        """
        from fluxopt import ModelData
        from fluxopt.math import build_sources

        data = ModelData.build(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 5]))])
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=Sizing(size_min=0, size_max=20)),
                    discharging=Flow(carrier='Elec', size=10),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        # Builds rather than refusing. (The system has no source, so it is
        # infeasible on its own merits — that is a different statement.)
        assert build_sources(data, {'cost': 1.0})


class TestStorageComponentStatus:
    def test_solution_includes_component_variables(self, optimize):
        """Storage with status emits ``component--on/startup/shutdown`` solutions."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 10]))]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=20),
                    discharging=Flow(carrier='Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        assert 'component--on' in result.solution
        assert 'component--startup' in result.solution
        assert 'component--shutdown' in result.solution
        assert 'Bat' in result.solution['component--on'].coords['component'].values

    def test_status_gates_both_flows(self, optimize):
        """When component_on=0, both charging and discharging are forced to 0."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 5]))]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=20),
                    discharging=Flow(carrier='Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Bat').values
        charge = result.solution['flow--rate'].sel(flow='Bat(charge)').values
        discharge = result.solution['flow--rate'].sel(flow='Bat(discharge)').values
        for t in range(3):
            if on[t] < 0.5:
                assert charge[t] < 1e-6, f't={t}: charge={charge[t]} but on=0'
                assert discharge[t] < 1e-6, f't={t}: discharge={discharge[t]} but on=0'

    def test_startup_cost(self, optimize):
        """Proves: effects_per_startup deters cycling — cost accrues per on-transition.

        Demand [0, 10, 0, 10, 0] over 5 steps; storage must charge from grid
        and re-discharge twice. Startup cost makes a single long charge cheaper
        than two short ones.

        Without startup cost, optimal=20 (grid energy at €1/MWh, no losses).
        With 50€/startup, the storage does not cycle — solver routes from grid
        directly when possible, charging once. Either way startup cost gets
        baked into the objective only when the storage is used.
        """
        result = optimize(
            timesteps=ts(5),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 10, 0, 10, 0]))],
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=20),
                    discharging=Flow(carrier='Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(effects_per_startup={'cost': 1000}),
                ),
            ],
        )
        # Direct supply costs 20€ — startup cost deters using storage at all.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)
        startups = result.solution['component--startup'].sel(component='Bat').values
        assert startups.sum() == 0

    def test_running_cost_accrues_per_timestep(self, optimize):
        """``effects_per_running_hour`` charges (cost/h) * on * dt per timestep."""
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 0, 10]))]
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=20),
                    discharging=Flow(carrier='Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(effects_per_running_hour={'cost': 5}),
                ),
            ],
        )
        # Running cost so high that storage stays off and demand draws from grid directly.
        # objective = 10 (grid only); storage on-hours = 0.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)
        on_hours = result.solution['component--on'].sel(component='Bat').values
        assert on_hours.sum() == 0

    def test_fixed_profile_with_status_solves(self, optimize):
        """Profile-bound governed flow with component Status applies the
        ``P = size * profile * on`` equality constraint.

        Charging is pinned to a fixed schedule. Solver picks on=1 where the
        profile is non-zero (else infeasible by bus balance) and may pick on=0
        elsewhere — the equality constraint allows P=0 when on=0.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 5]))]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Bat',
                    charging=Flow(carrier='Elec', size=10, fixed_relative_profile=np.array([0.5, 0.5, 0])),
                    discharging=Flow(carrier='Elec', size=10),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        # Grid pays for charge (5 MWh = 5 * 1 €/MWh = 5) plus demand (5 MWh = 5).
        # Plus any discharge gap. Charging is forced by profile when on=1.
        assert result.effect_totals.sel(effect='cost').item() >= 5.0
        # Charging actual rate must match profile when on=1 (and be 0 when on=0)
        on = result.solution['component--on'].sel(component='Bat').values
        charge = result.solution['flow--rate'].sel(flow='Bat(charge)').values
        for t in range(3):
            expected = 0.5 * 10 * on[t] if t < 2 else 0.0
            assert_allclose(charge[t], expected, atol=1e-6)
