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

from fluxopt import Carrier, Converter, Effect, Flow, Investment, ModelData, Port, Sizing, Status, Storage
from fluxopt.model import FlowSystemModel
from fluxopt.relational import UnsupportedFeatureError, build_sources, solve

pytest.importorskip('farkas', reason='the relational backend needs the `relational` extra')

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

    rates = result.primal('rate').drop(columns='period')
    pivot = rates.pivot(index='time', columns='flow', values='value').sort_index()
    for flow_id in pivot.columns:
        expected = reference.flow_rate.solution.sel(flow=flow_id).values
        assert pivot[flow_id].to_numpy() == pytest.approx(expected, abs=1e-9)


@pytest.mark.slow
def test_effect_limit_binds() -> None:
    """A tight cap must actually hold — a dropped limit would solve cheaper."""
    data = ModelData.build(**_system(168, None, BINDING_CO2_CAP))
    reference = float(_linopy_optimum(data).m.objective.value)
    result = solve(data, OBJECTIVE, solver_options=SOLVER_OPTIONS)

    totals = result.primal('effect_total').set_index('effect')['value']
    assert totals['co2'] == pytest.approx(BINDING_CO2_CAP, rel=1e-9)
    assert result.objective == pytest.approx(reference, rel=1e-9)


def test_sparse_coefficients_are_not_materialised() -> None:
    """`effect_coeff` is declared dense but only live rows are emitted."""
    data = ModelData.build(**_system(48))
    sources, coords = build_sources(data, OBJECTIVE)

    dense = len(sources['flow']) * len(coords['effect']) * len(coords['time']) * len(coords['period'])
    assert len(sources['effect_coeff']) < dense / 2


def test_unsupported_feature_raises_rather_than_dropping() -> None:
    """A feature with no formulation must fail loudly, not solve to a wrong answer."""
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

    with pytest.raises(UnsupportedFeatureError, match='investment'):
        build_sources(data, OBJECTIVE)
