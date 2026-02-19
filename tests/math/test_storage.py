from __future__ import annotations

from datetime import datetime

import pytest

from fluxopt import Bus, Effect, Flow, Port, Storage, optimize


class TestStorage:
    def test_charge_in_cheap_discharge_in_expensive(self, timesteps_4):
        """Battery charges in cheap hours, discharges in expensive hours."""
        prices = [0.02, 0.08, 0.02, 0.08]

        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': prices})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        charge_flow = Flow(bus='elec', size=50)
        discharge_flow = Flow(bus='elec', size=50)
        battery = Storage('battery', charging=charge_flow, discharging=discharge_flow, capacity=100.0)

        result = optimize(
            timesteps=timesteps_4,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        charge = result.flow_rate('battery(charge)').values
        discharge = result.flow_rate('battery(discharge)').values

        # Should charge in cheap hours (t0, t2) and discharge in expensive (t1, t3)
        assert charge[0] > 0  # t0: cheap
        assert charge[1] == pytest.approx(0.0, abs=1e-6)  # t1: expensive
        assert charge[2] > 0  # t2: cheap
        assert charge[3] == pytest.approx(0.0, abs=1e-6)  # t3: expensive

        assert discharge[0] == pytest.approx(0.0, abs=1e-6)  # t0: cheap
        assert discharge[1] > 0  # t1: expensive
        assert discharge[2] == pytest.approx(0.0, abs=1e-6)  # t2: cheap
        assert discharge[3] > 0  # t3: expensive

    def test_level_starts_at_prior(self, timesteps_4):
        """Prior level feeds into the balance at t=0."""
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        charge_flow = Flow(bus='elec', size=50)
        discharge_flow = Flow(bus='elec', size=50)
        battery = Storage(
            'battery',
            charging=charge_flow,
            discharging=discharge_flow,
            capacity=100.0,
            prior_level=0.0,
            cyclic=False,
        )

        result = optimize(
            timesteps=timesteps_4,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        cs = result.storage_level('battery')
        charge_t0 = float(result.flow_rate('battery(charge)').values[0])
        discharge_t0 = float(result.flow_rate('battery(discharge)').values[0])
        # End-of-period: level[0] = prior * decay + charge[0] * eta_c * dt - discharge[0] * dt / eta_d
        # With prior=0, dt=1, eta=1, loss=0: level[0] = charge[0] - discharge[0]
        expected = 0.0 + charge_t0 - discharge_t0
        assert float(cs.values[0]) == pytest.approx(expected, abs=1e-6)

    def test_cyclic_storage(self):
        """Cyclic constraint: initial for period 0 equals level at end of last period."""
        timesteps = [datetime(2024, 1, 1, h) for h in range(2)]
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': [0.02, 0.08]})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5])

        charge_flow = Flow(bus='elec', size=100)
        discharge_flow = Flow(bus='elec', size=100)
        battery = Storage(
            'battery',
            charging=charge_flow,
            discharging=discharge_flow,
            capacity=100.0,
        )

        result = optimize(
            timesteps=timesteps,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        cs = result.storage_level('battery')
        charge = result.flow_rate('battery(charge)').values
        discharge = result.flow_rate('battery(discharge)').values

        # End-of-period: level[0] = initial * decay + charge[0] - discharge[0]
        # Cyclic: initial = level[-1]. With dt=1, eta=1, loss=0:
        # level[0] = level[-1] + charge[0] - discharge[0]
        level_0 = float(cs.values[0])
        level_last = float(cs.values[-1])
        expected = level_last + float(charge[0]) - float(discharge[0])
        assert level_0 == pytest.approx(expected, abs=1e-6)

    def test_storage_with_efficiency(self, timesteps_3):
        """With eta_charge < 1, more energy is drawn from bus than stored."""
        eta_c = 0.8
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        charge_flow = Flow(bus='elec', size=100)
        discharge_flow = Flow(bus='elec', size=100)
        battery = Storage(
            'battery',
            charging=charge_flow,
            discharging=discharge_flow,
            capacity=200.0,
            eta_charge=eta_c,
        )

        result = optimize(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        # With charging efficiency, stored energy = charge_rate * eta_c
        cs = result.storage_level('battery')
        # Check balance between period 1 and period 2 (both observable):
        # level[2] = level[1] * decay + charge[2] * eta_c * dt - discharge[2] * dt / eta_d
        # With dt=1, loss=0, eta_d=1: level[2] = level[1] + charge[2] * eta_c - discharge[2]
        charge_t2 = float(result.flow_rate('battery(charge)').values[2])
        discharge_t2 = float(result.flow_rate('battery(discharge)').values[2])
        cs_t1 = float(cs.values[1])
        cs_t2 = float(cs.values[2])
        expected_cs_t2 = cs_t1 + charge_t2 * eta_c - discharge_t2
        assert cs_t2 == pytest.approx(expected_cs_t2, abs=1e-6)
