from __future__ import annotations

import pandas as pd
import pytest
from conftest import ts

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port, optimize


class TestEditingTheMath:
    """Extending the model by editing the math, rather than by callback.

    `customize` used to hand a caller the built linopy model to poke. There is
    no such object now: the math is a file, so extending it means adding a
    declaration to that file and binding data for whatever it names.
    """

    @pytest.fixture
    def simple_system(self):
        """Single-bus system: grid source (size=100) feeding a fixed 50 MW demand."""
        return {
            'timesteps': ts(3),
            'carriers': [Carrier(id='elec')],
            'effects': [Effect(id='cost')],
            'objective': 'cost',
            'ports': [
                Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
            ],
        }

    def test_math_reads_without_data_or_solver(self, simple_system):
        """The equations are an artefact before anything is bound to them."""
        math = FlowSystem(**simple_system).math()

        assert 'carrier_balance' in math.constraints
        assert math.objective.sense == 'minimize'
        # It round-trips as the file a reviewer reads.
        assert 'carrier_balance' in math.to_yaml()

    def test_added_constraint_changes_the_answer(self, simple_system):
        """A row the caller wrote, in the same language, with its own data."""
        # A dearer second source, so capping the cheap one has somewhere to go
        # rather than making a fixed demand infeasible.
        simple_system['ports'].append(
            Port(id='backup', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 5.0})])
        )
        base = optimize(**simple_system)
        assert base.flow_rate('grid(elec)').values == pytest.approx([50.0] * 3, abs=1e-6)

        system = FlowSystem(**simple_system)
        math = system.math()
        math.parameters['grid_cap'] = type(math.parameters['carrier_sign'])(dims=['time'])
        math.constraints['grid_cap_row'] = type(math.constraints['carrier_balance'])(
            foreach=['flow', 'time', 'period'],
            where='is_grid',
            expression='rate <= grid_cap',
        )
        math.parameters['is_grid'] = type(math.parameters['carrier_sign'])(dims=['flow'], dtype='bool')

        result = system.optimize(
            math=math,
            parameters={
                'grid_cap': pd.DataFrame({'time': [0, 1, 2], 'value': [30.0, 30.0, 30.0]}),
                'is_grid': pd.DataFrame({'flow': ['grid(elec)'], 'value': [True]}),
            },
        )
        # The cap binds: 50 was the unconstrained answer, 30 is the cap, and
        # the dearer source picks up the rest.
        assert result.flow_rate('grid(elec)').values == pytest.approx([30.0] * 3, abs=1e-6)
        assert result.flow_rate('backup(elec)').values == pytest.approx([20.0] * 3, abs=1e-6)
        assert result.objective > base.objective

    def test_a_caller_may_not_overwrite_the_program_s_own_data(self, simple_system):
        """Silently replacing `rate_max` would change the model without editing it."""
        system = FlowSystem(**simple_system)
        with pytest.raises(ValueError, match='cannot be supplied'):
            system.optimize(parameters={'rate_max': pd.DataFrame({'flow': [], 'value': []})})

    def test_unedited_math_answers_what_the_shipped_program_answers(self, simple_system):
        """Passing `math=` unchanged is not a different model."""
        system = FlowSystem(**simple_system)
        assert system.optimize(math=system.math()).objective == pytest.approx(
            optimize(**simple_system).objective, abs=1e-9
        )
