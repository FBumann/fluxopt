from __future__ import annotations

from datetime import datetime

import pytest

from fluxopt import Bus, Effect, Flow, Port, Storage, solve


class TestStorage:
    def test_charge_in_cheap_discharge_in_expensive(self, timesteps_4):
        """Battery charges in cheap hours, discharges in expensive hours."""
        prices = [0.02, 0.08, 0.02, 0.08]

        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': prices})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        charge_flow = Flow(bus='elec', size=50)
        discharge_flow = Flow(bus='elec', size=50)
        battery = Storage('battery', charging=charge_flow, discharging=discharge_flow, capacity=100.0)

        result = solve(
            timesteps=timesteps_4,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        charge = result.flow_rate('battery(charge)')['value'].to_list()
        discharge = result.flow_rate('battery(discharge)')['value'].to_list()

        # Should charge in cheap hours (t0, t2) and discharge in expensive (t1, t3)
        assert charge[0] > 0  # t0: cheap
        assert charge[1] == pytest.approx(0.0, abs=1e-6)  # t1: expensive
        assert charge[2] > 0  # t2: cheap
        assert charge[3] == pytest.approx(0.0, abs=1e-6)  # t3: expensive

        assert discharge[0] == pytest.approx(0.0, abs=1e-6)  # t0: cheap
        assert discharge[1] > 0  # t1: expensive
        assert discharge[2] == pytest.approx(0.0, abs=1e-6)  # t2: cheap
        assert discharge[3] > 0  # t3: expensive

    def test_charge_state_starts_at_zero(self, timesteps_4):
        """Initial charge state defaults to 0."""
        source_flow = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        demand_flow = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

        charge_flow = Flow(bus='elec', size=50)
        discharge_flow = Flow(bus='elec', size=50)
        battery = Storage(
            'battery', charging=charge_flow, discharging=discharge_flow, capacity=100.0, initial_charge_state=0.0
        )

        result = solve(
            timesteps=timesteps_4,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        cs = result.charge_state('battery')
        # First row is the initial charge state (first timestep)
        first_time = cs['time'][0]
        assert cs.filter(cs['time'] == first_time)['value'][0] == pytest.approx(0.0, abs=1e-6)

    def test_cyclic_storage(self):
        """Cyclic constraint: charge state at end == start."""
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
            initial_charge_state='cyclic',
        )

        result = solve(
            timesteps=timesteps,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        cs = result.charge_state('battery')
        first = cs['value'][0]
        last = cs['value'][-1]
        assert last == pytest.approx(first, abs=1e-6)

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

        result = solve(
            timesteps=timesteps_3,
            buses=[Bus('elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[Port('grid', imports=[source_flow]), Port('demand', exports=[demand_flow])],
            storages=[battery],
        )

        # With charging efficiency, stored energy = charge_rate * eta_c
        cs = result.charge_state('battery')
        charge_t0 = result.flow_rate('battery(charge)')['value'][0]
        cs_t1 = cs['value'][1]
        cs_t0 = cs['value'][0]
        # cs[t1] = cs[t0] + charge[t0] * eta_c - discharge[t0] / eta_d
        discharge_t0 = result.flow_rate('battery(discharge)')['value'][0]
        expected_cs_t1 = cs_t0 + charge_t0 * eta_c - discharge_t0
        assert cs_t1 == pytest.approx(expected_cs_t1, abs=1e-6)
