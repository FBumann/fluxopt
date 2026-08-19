"""The two backends must agree.

Both consume the same `ModelData`: one builds a linopy model, the other binds
the YAML program and streams it. Their objectives must match, and — the check
that actually catches a dropped constraint — the relational lane must never
*beat* the linopy lane's proven optimum, since a minimisation cannot do better
than the optimum without having lost a constraint.

Both lanes are driven to `mip_rel_gap=1e-9`. Comparing two loose MIP solves
would report differences that are only different incumbents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    Investment,
    ModelData,
    PiecewiseConversion,
    Port,
    Sizing,
    Status,
    Storage,
)
from fluxopt.contract import Var
from fluxopt.model import FlowSystemModel
from fluxopt.relational import UnsupportedFeatureError, build_sources, solve
from fluxopt.results import Result

OBJECTIVE = {'cost': 1.0}
SOLVER_OPTIONS = {'mip_rel_gap': 1e-9}
#: Generous by default: the short horizons used for parity have no co2 slack
#: (their cost optimum *is* their minimum co2), so a tighter cap would be
#: infeasible rather than binding. The limit rows are still emitted and must
#: not perturb the solution. `test_effect_limit_binds` uses a horizon with room.
CO2_CAP = 1.0e6
#: Between the 168-step optimum's co2 (213.1) and the tightest feasible
#: cap (~202.5), so it changes the solution instead of sitting slack.
BINDING_CO2_CAP = 205.0


def _system(n: int, periods: list[int] | None = None, co2_cap: float | None = CO2_CAP) -> dict:
    """A system exercising every feature the relational program expresses."""
    rng = np.random.default_rng(0)
    hours = np.arange(n)
    demand = np.clip(0.5 + 0.3 * np.cos(2 * np.pi * hours / 24) + 0.05 * rng.standard_normal(n), 0.05, 1.0)
    gas_price = 30 + 5 * np.sin(2 * np.pi * hours / 168)
    elec_price = 60 + 25 * np.sin(2 * np.pi * hours / 24) + 3 * rng.standard_normal(n)
    return {
        **({'periods': periods} if periods else {}),
        'timesteps': pd.date_range('2025-01-01', periods=n, freq='h'),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='heat'), Carrier(id='ambient')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': 45.0}),
            Effect(id='co2', unit='kg', total_max=co2_cap),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[
                    Flow(carrier='gas', size=60.0, effects_per_flow_hour={'cost': gas_price.tolist(), 'co2': 0.2})
                ],
            ),
            Port(
                id='power_exchange',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=30.0,
                        effects_per_flow_hour={'cost': elec_price.tolist(), 'co2': 0.35},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=30.0,
                        effects_per_flow_hour={'cost': (-0.9 * elec_price).tolist()},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='heat_network',
                exports=[Flow(carrier='heat', size=20.0, fixed_relative_profile=demand.tolist())],
            ),
        ],
        'converters': [
            Converter.boiler(
                'gas_boiler',
                0.92,
                Flow(carrier='gas'),
                Flow(carrier='heat', size=Sizing(size_min=2.0, size_max=15.0, effects_per_size={'cost': 240.0})),
            ),
            Converter.heat_pump(
                'heat_pump',
                3.2,
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(
                    carrier='heat',
                    size=Sizing(
                        size_min=1.0,
                        size_max=8.0,
                        mandatory=False,
                        effects_per_size={'cost': 400.0},
                        effects_fixed={'cost': 5000.0},
                    ),
                ),
            ),
            Converter.chp(
                'chp',
                0.38,
                0.45,
                Flow(
                    carrier='gas',
                    size=Sizing(
                        size_min=8.0,
                        size_max=25.0,
                        mandatory=False,
                        effects_per_size={'cost': 180.0},
                        effects_fixed={'cost': 3000.0},
                    ),
                    relative_rate_min=0.4,
                    ramp_up_per_hour=0.3,
                    ramp_down_per_hour=0.25,
                    status=Status(
                        uptime_min=24.0,
                        downtime_min=6.0,
                        effects_per_startup={'cost': 900.0},
                        effects_per_running_hour={'cost': 12.0},
                    ),
                    prior_rates=[18.0, 18.0, 18.0],
                ),
                Flow(carrier='elec'),
                Flow(carrier='heat'),
            ),
        ],
        'storages': [
            Storage(
                id='tank',
                charging=Flow(carrier='heat', size=10.0),
                discharging=Flow(carrier='heat', size=10.0),
                capacity=Sizing(
                    size_min=10.0,
                    size_max=200.0,
                    mandatory=False,
                    effects_per_size={'cost': 55.0},
                    effects_fixed={'cost': 1200.0},
                ),
                relative_loss_per_hour=0.003,
                final_level_min=20.0,
                prevent_simultaneous=True,
            ),
        ],
    }


def _invest_system(n: int, periods: list[int], lifetime: int) -> dict:
    """A multi-period system whose capacity is a build-timing decision."""
    rng = np.random.default_rng(1)
    hours = np.arange(n)
    demand = np.clip(0.5 + 0.3 * np.cos(2 * np.pi * hours / 24) + 0.05 * rng.standard_normal(n), 0.05, 1.0)
    return {
        'periods': periods,
        'timesteps': pd.date_range('2025-01-01', periods=n, freq='h'),
        'carriers': [Carrier(id='gas'), Carrier(id='heat')],
        'effects': [Effect(id='cost', unit='EUR')],
        'ports': [
            Port(id='gas_grid', imports=[Flow(carrier='gas', size=60.0, effects_per_flow_hour={'cost': 35.0})]),
            Port(
                id='heat_network',
                exports=[Flow(carrier='heat', size=20.0, fixed_relative_profile=demand.tolist())],
            ),
        ],
        'converters': [
            Converter.boiler(
                'gas_boiler',
                0.92,
                Flow(carrier='gas'),
                Flow(
                    carrier='heat',
                    size=Investment(
                        size_min=2.0,
                        size_max=20.0,
                        lifetime=lifetime,
                        effects_per_size_at_build={'cost': 900.0},
                        effects_fixed_at_build={'cost': 4000.0},
                        effects_per_size_recurring={'cost': 30.0},
                    ),
                ),
            ),
        ],
    }


def _linopy_optimum(data: ModelData) -> FlowSystemModel:
    model = FlowSystemModel(data, objective=OBJECTIVE)
    model.build()
    model.m.solve(solver_name='highs', output_flag=False, **SOLVER_OPTIONS)
    return model


@pytest.mark.parametrize(
    ('timesteps', 'periods', 'co2_cap'),
    # The cap is a weighted total across periods, so a multi-period horizon
    # carrying period weights would make this exact cap infeasible.
    [(48, None, CO2_CAP), (24, [2025, 2030], CO2_CAP)],
    ids=['single-period', 'multi-period'],
)
def test_objective_matches_linopy(timesteps: int, periods: list[int] | None, co2_cap: float | None) -> None:
    data = ModelData.build(**_system(timesteps, periods, co2_cap))
    reference = float(_linopy_optimum(data).m.objective.value)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    # A minimisation cannot beat the proven optimum: below it means a
    # constraint went missing on the relational side.
    assert result.objective >= reference - 1e-6
    assert result.objective == pytest.approx(reference, rel=1e-9)


def test_flow_rates_match_linopy() -> None:
    """Objectives can coincide while schedules differ — compare the primals."""
    data = ModelData.build(**_system(48))
    reference = _linopy_optimum(data)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    rates = result.flow_rates
    for flow_id in rates.coords['flow'].values:
        expected = reference.flow_rate.solution.sel(flow=flow_id).values
        assert rates.sel(flow=flow_id).values == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    ('periods', 'lifetime'),
    # Builds happen once, so the lifetime has to span the horizon or the last
    # periods have no capacity at all and the model is infeasible on both sides.
    [([2025, 2030], 2), ([2025, 2030, 2035], 3)],
    ids=['2-periods', '3-periods'],
)
def test_investment_matches_linopy(periods: list[int], lifetime: int) -> None:
    """Build timing, lifetime windows and the at-build effect terms."""
    data = ModelData.build(**_invest_system(24, periods, lifetime))
    reference = _linopy_optimum(data)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert result.objective >= float(reference.m.objective.value) - 1e-6
    assert result.objective == pytest.approx(float(reference.m.objective.value), rel=1e-9)

    # The build decision itself must agree, not only its cost.
    # Both sides read through `Result`, because a Dataset aligns its members
    # to the union of their coordinates — comparing one lane's Dataset entry
    # against the other lane's raw variable compares two different shapes.
    built = result.solution[Var.INVEST_BUILD]
    expected = Result.from_model(reference).solution[Var.INVEST_BUILD]
    assert list(built.coords['flow'].values) == list(expected.coords['flow'].values)
    assert np.nan_to_num(built.values).round() == pytest.approx(np.nan_to_num(expected.values).round())


@pytest.mark.slow
def test_effect_limit_binds() -> None:
    """A tight cap must actually hold — a dropped limit would solve cheaper."""
    data = ModelData.build(**_system(168, None, BINDING_CO2_CAP))
    reference = float(_linopy_optimum(data).m.objective.value)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert float(result.effect_totals.sel(effect='co2')) == pytest.approx(BINDING_CO2_CAP, rel=1e-9)
    assert result.objective == pytest.approx(reference, rel=1e-9)


def test_mandatory_storage_sizing_binds_the_right_dims() -> None:
    """An absent lump term must still be keyed by its own entity dim.

    A storage sized with `mandatory=True` has no build indicator, so the
    `cap_ind_coeff` table is empty — and an empty table keyed by `flow`
    instead of `storage` fails to bind. Regression for a real benchmark
    system (`green_city`).
    """
    elements = _system(24)
    elements['storages'] = [
        Storage(
            id='tank',
            charging=Flow(carrier='heat', size=10.0),
            discharging=Flow(carrier='heat', size=10.0),
            capacity=Sizing(size_min=10.0, size_max=200.0, effects_per_size={'cost': 55.0}),
            relative_loss_per_hour=0.003,
        )
    ]
    data = ModelData.build(**elements)
    reference = float(_linopy_optimum(data).m.objective.value)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert result.objective == pytest.approx(reference, rel=1e-9)


def test_component_status_matches_linopy() -> None:
    """One binary per component, read by every flow it governs.

    A storage with Status gates its charging and discharging flows together —
    the case `at(running, by=status_of)` exists for. The linopy lane builds it
    as a per-component binary and a constraint per governed flow; this one
    states it once and lets the lookup decide which rows read it.
    """
    elements = _system(48)
    elements['storages'] = [
        Storage(
            id='tank',
            charging=Flow(carrier='heat', size=10.0),
            discharging=Flow(carrier='heat', size=10.0),
            capacity=100.0,
            relative_loss_per_hour=0.003,
            status=Status(uptime_min=3.0, downtime_min=2.0, effects_per_startup={'cost': 40.0}),
        )
    ]
    data = ModelData.build(**elements)
    reference = _linopy_optimum(data)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert result.objective >= float(reference.m.objective.value) - 1e-6
    assert result.objective == pytest.approx(float(reference.m.objective.value), rel=1e-9)

    # The binary itself must agree, not only its cost — a dropped gate would
    # still solve, and cheaper.
    on = result.solution[Var.COMPONENT_ON].sel(component='tank')
    expected = reference.component_on.solution.sel(component='tank')
    assert on.values.round() == pytest.approx(expected.values.round())


def test_both_lanes_answer_with_the_same_result() -> None:
    """One `Result`, whichever lane built it — including the derived views.

    The point of the object being shared is that parity compares *answers*
    rather than shapes: effect contributions are reconstructed from the
    solution and the same ModelData either way, so agreeing here means the
    solution agrees everywhere it is read from.
    """
    data = ModelData.build(**_system(48))
    theirs = Result.from_model(_linopy_optimum(data))
    ours = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert ours.objective == pytest.approx(theirs.objective, rel=1e-9)
    assert ours.objective_weights == theirs.objective_weights

    for view in ('effect_totals', 'effects_lump'):
        mine, other = getattr(ours, view), getattr(theirs, view)
        assert list(mine.coords['effect'].values) == list(other.coords['effect'].values), view
        assert mine.values == pytest.approx(other.values, abs=1e-7), view

    # Reconstructed per-timestep effects: the accessor that goes through
    # `stats`, so this covers the contributions path as well.
    assert ours.effects_temporal.values == pytest.approx(theirs.effects_temporal.values, abs=1e-7)


def test_sparse_coefficients_are_not_materialised() -> None:
    """`effect_coeff` is declared dense but only live rows are emitted."""
    data = ModelData.build(**_system(48))
    sources, coords = build_sources(data, OBJECTIVE)

    dense = len(sources['flow']) * len(coords['effect']) * len(coords['time']) * len(coords['period'])
    assert len(sources['effects_per_flow_hour']) < dense / 2


def test_piecewise_matches_linopy() -> None:
    """A curve tying N flows through shared weights, N being data.

    lpspec's `piecewise:` block takes a static link list, so the program
    writes the lambda formulation out and keys a link on `flow` — which is
    what lets one declaration serve curves of different arity (fluxopt/lpspec#1101).
    """
    elements = _system(48)
    elements['converters'] = [
        Converter(
            id='pw_boiler',
            inputs=[Flow(carrier='gas', size=100.0)],
            outputs=[Flow(carrier='heat', size=70.0)],
            conversion=PiecewiseConversion(points={'gas': [0, 50, 100], 'heat': [0, 45, 70]}),
        ),
        Converter(
            id='pw_chp',
            inputs=[Flow(carrier='gas', size=120.0)],
            outputs=[Flow(carrier='heat', size=60.0), Flow(carrier='elec', size=40.0)],
            conversion=PiecewiseConversion(
                points={'gas': [0, 60, 120], 'heat': [0, 35, 60], 'elec': [0, 18, 40]},
            ),
        ),
    ]
    data = ModelData.build(**elements)
    reference = _linopy_optimum(data)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    assert result.objective >= float(reference.m.objective.value) - 1e-6
    assert result.objective == pytest.approx(float(reference.m.objective.value), rel=1e-9)

    # Two curves of different arity in one system is the case a static link
    # list cannot express, so assert both were actually built.
    for flow_id in ('pw_boiler(gas)', 'pw_chp(gas)', 'pw_chp(elec)'):
        expected = reference.flow_rate.solution.sel(flow=flow_id).values.ravel()
        assert result.flow_rates.sel(flow=flow_id).values.ravel() == pytest.approx(expected, abs=1e-7)


def test_piecewise_lp_method_raises_rather_than_answering_differently() -> None:
    """`method='lp'` is a relaxation this lane has no formulation for."""
    elements = _system(24)
    elements['converters'] = [
        Converter(
            id='pw_boiler',
            inputs=[Flow(carrier='gas', size=100.0)],
            outputs=[Flow(carrier='heat', size=70.0)],
            conversion=PiecewiseConversion(points=[('gas', [0, 50, 100]), ('heat', [0, 45, 70], '<=')], method='lp'),
        )
    ]
    data = ModelData.build(**elements)

    with pytest.raises(UnsupportedFeatureError, match='lp'):
        build_sources(data, OBJECTIVE)


def test_investment_requires_periods() -> None:
    """Investment is period-timed; without periods it must not build silently."""
    elements = _system(24)
    elements['converters'] = [
        Converter.boiler(
            'gas_boiler',
            0.92,
            Flow(carrier='gas'),
            Flow(carrier='heat', size=Investment(size_min=1.0, size_max=10.0, lifetime=20)),
        )
    ]
    data = ModelData.build(**elements)

    with pytest.raises(UnsupportedFeatureError, match='multi-period'):
        build_sources(data, OBJECTIVE)
