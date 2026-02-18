"""Mathematical correctness tests for storage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Bus, Effect, Flow, Port, Sizing, Storage

from .conftest import ts


class TestStorage:
    def test_storage_shift_saves_money(self, optimize):
        """Proves: Storage enables temporal arbitrage — charge cheap, discharge when expensive.

        Sensitivity: Without storage, demand at t=2 must be bought at 10€/kWh → cost=200.
        With working storage, buy at t=1 for 1€/kWh → cost=20. A 10x difference.
        """
        result = optimize(
            timesteps=ts(3),
            buses=[Bus('Elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Elec', size=1, fixed_relative_profile=np.array([0, 0, 20])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': np.array([10, 1, 10])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=100),
                    discharging=Flow(bus='Elec', size=100),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

    def test_storage_losses(self, optimize):
        """Proves: relative_loss_per_hour correctly reduces stored energy over time.

        Sensitivity: If losses were ignored (0%), only 90 would be charged → cost=90.
        With 10% loss, must charge 100 to have 90 after 1h → cost=100.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Elec', size=1, fixed_relative_profile=np.array([0, 90])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': np.array([1, 1000])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=200,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0.1,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self, optimize):
        """Proves: eta_charge and eta_discharge are both applied to the energy flow.

        Sensitivity: If eta_charge broken (1.0), cost=90. If eta_discharge broken (1.0),
        cost=80. If both broken, cost=72. Only both correct yields cost=100.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Elec', size=1, fixed_relative_profile=np.array([0, 72])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': np.array([1, 1000])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=200,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=0.9,
                    eta_discharge=0.8,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self, optimize):
        """Proves: relative_maximum_level caps how much energy can be stored.

        Sensitivity: If level bound were ignored, all 60 stored cheaply → cost=60.
        With the bound enforced, cost=1050 (50*1 + 10*100).
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Elec', size=1, fixed_relative_profile=np.array([0, 60])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': np.array([1, 100])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    relative_maximum_level=0.5,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 1050.0, rtol=1e-5)

    def test_storage_invest_capacity(self, optimize):
        """Proves: Sizing on capacity correctly sizes the storage.

        Sensitivity: If invest cost were 100€/kWh (>9 saving), no storage built → cost=500.
        At 1€/kWh, storage built → cost=50*1 (buy) + 50*1 (invest) = 100.
        """
        result = optimize(
            timesteps=ts(2),
            buses=[Bus('Elec')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow(bus='Elec', size=1, fixed_relative_profile=np.array([0, 50])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(bus='Elec', effects_per_flow_hour={'cost': np.array([1, 10])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    'Battery',
                    charging=Flow(bus='Elec', size=200),
                    discharging=Flow(bus='Elec', size=200),
                    capacity=Sizing(min_size=0, max_size=200, effects_per_size={'cost': 1}),
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.storage_capacities.sel(storage='Battery').item(), 50.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    @pytest.mark.skip(reason='prior_level is absolute in fluxopt, not relative')
    def test_storage_relative_minimum_level(self, optimize):
        """Proves: relative_minimum_level enforces a minimum SOC at all times.

        Sensitivity: Without min SOC, discharge all 100 → no grid → cost=50.
        With min SOC=0.3, max discharge=70 → grid covers 10 @100€ → cost=1050.
        """
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(3)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 80, 0]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100, 1]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=50,
                relative_minimum_charge_state=0.3,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 1050.0, rtol=1e-5)

    @pytest.mark.skip(reason='flixopt-specific legacy test')
    def test_storage_cyclic_level(self, optimize):
        """Proves: cyclic=True forces the storage to end at the
        same level it started, preventing free energy extraction."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 50]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state='equals_final',
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 50.0, rtol=1e-5)

    @pytest.mark.skip(reason='absolute final level not supported in fluxopt')
    def test_storage_minimal_final_level(self, optimize):
        """Proves: minimal_final_level forces the storage to retain at least the
        specified absolute energy at the end."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 20]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=0,
                minimal_final_charge_state=60,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 80.0, rtol=1e-5)

    @pytest.mark.skip(reason='prevent_simultaneous not supported in fluxopt')
    def test_prevent_simultaneous_charge_and_discharge(self, optimize):
        """Proves: prevent_simultaneous_charge_and_discharge=True prevents the storage
        from charging and discharging in the same timestep."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(3)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([10, 20, 10]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 10, 1]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=100),
                discharging=fx.Flow(bus='Elec', size=100),
                capacity_in_flow_hours=100,
                initial_charge_state=0,
                eta_charge=0.9,
                eta_discharge=0.9,
                relative_loss_per_hour=0,
                prevent_simultaneous_charge_and_discharge=True,
            ),
        )
        fs = optimize(fs)

    @pytest.mark.skip(reason='absolute final level + imbalance penalty not supported in fluxopt')
    def test_storage_maximal_final_level(self, optimize):
        """Proves: maximal_final_level caps the storage level at the end."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([50, 0]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([100, 1]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                maximal_final_charge_state=20,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    @pytest.mark.skip(reason='relative final level not supported in fluxopt')
    def test_storage_relative_minimum_final_level(self, optimize):
        """Proves: relative_minimum_final_level forces a minimum final SOC
        as a fraction of capacity."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 80]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=50,
                relative_minimum_charge_state=np.array([0, 0]),
                relative_minimum_final_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 3050.0, rtol=1e-5)

    @pytest.mark.skip(reason='relative final level + imbalance penalty not supported in fluxopt')
    def test_storage_relative_maximum_final_level(self, optimize):
        """Proves: relative_maximum_final_level caps the storage at end
        as a fraction of capacity."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([50, 0]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([100, 1]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                relative_maximum_charge_state=np.array([1.0, 1.0]),
                relative_maximum_final_charge_state=0.2,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    @pytest.mark.skip(reason='relative final level not supported in fluxopt')
    def test_storage_relative_minimum_final_level_scalar(self, optimize):
        """Proves: relative_minimum_final_level works when relative_minimum_level
        is a scalar (default=0, no time dimension)."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 80]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=50,
                relative_minimum_final_charge_state=0.5,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['costs'].item(), 3050.0, rtol=1e-5)

    @pytest.mark.skip(reason='relative final level + imbalance penalty not supported in fluxopt')
    def test_storage_relative_maximum_final_level_scalar(self, optimize):
        """Proves: relative_maximum_final_level works when relative_maximum_level
        is a scalar (default=1, no time dimension)."""
        import flixopt as fx

        from .conftest import make_flow_system

        fs = make_flow_system(2)
        fs.add(
            fx.Bus('Elec', imbalance_penalty_per_flow_hour=5),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([50, 0]))],
            ),
            fx.Port('Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([100, 1]))]),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=200),
                discharging=fx.Flow(bus='Elec', size=200),
                capacity_in_flow_hours=100,
                initial_charge_state=80,
                relative_maximum_final_charge_state=0.2,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        assert_allclose(fs.solution['objective'].item(), 50.0, rtol=1e-5)

    @pytest.mark.skip(reason='balanced invest not supported in fluxopt')
    def test_storage_balanced_invest(self, optimize):
        """Proves: balanced=True forces charge and discharge invest sizes to be equal."""
        import flixopt as fx
        from flixopt import InvestParameters

        from .conftest import make_flow_system

        fs = make_flow_system(3)
        fs.add(
            fx.Bus('Elec'),
            fx.Effect('costs', '€', is_standard=True, is_objective=True),
            fx.Port(
                'Demand',
                exports=[fx.Flow(bus='Elec', flow_id='elec', size=1, fixed_relative_profile=np.array([0, 80, 80]))],
            ),
            fx.Port(
                'Grid', imports=[fx.Flow(bus='Elec', flow_id='elec', effects_per_flow_hour=np.array([1, 100, 100]))]
            ),
            fx.Storage(
                'Battery',
                charging=fx.Flow(bus='Elec', size=InvestParameters(maximum_size=200, effects_of_investment_per_size=1)),
                discharging=fx.Flow(
                    bus='Elec', size=InvestParameters(maximum_size=200, effects_of_investment_per_size=1)
                ),
                capacity_in_flow_hours=200,
                initial_charge_state=0,
                balanced=True,
                eta_charge=1,
                eta_discharge=1,
                relative_loss_per_hour=0,
            ),
        )
        fs = optimize(fs)
        charge_size = fs.solution['Battery(charging)|size'].item()
        discharge_size = fs.solution['Battery(discharging)|size'].item()
        assert_allclose(charge_size, discharge_size, rtol=1e-5)
