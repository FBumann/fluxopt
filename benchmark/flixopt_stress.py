"""The fluxopt ``stress`` reference system, ported 1:1 to flixopt.

Same structure, same fixed-seed rng draw order (seed 7), so both frameworks
build the same abstract multi-period MILP: ~190 flows over (time, period),
a 30-effect share graph, optional invest in bulk, 12 bus balances, one
piecewise unit with status, one status flow, a storage fleet with
time-varying charge-state caps.

Mapping (fluxopt -> flixopt):
    Carrier nodes          -> one Bus per carrier:node
    Effect.contribution_from -> share_from_temporal + share_from_periodic
    periodic_min/max, total_max -> minimum/maximum_periodic, maximum_total
    Sizing                 -> InvestParameters (fixed_size for min==max)
    Status                 -> StatusParameters
    Port                   -> Source / Sink
    Converter              -> LinearConverter
    PiecewiseConversion    -> PiecewiseConversion(Piecewise([Piece..]))
    Storage                -> Storage (prevent_simultaneous off, cyclic via
                              initial_charge_state='equals_final')

Not part of the pinned benchmark env — flixopt is not a fluxopt dependency.
Run against the pinned comparison version with uv:

    uv run --no-project --with 'flixopt==7.2.3' python benchmark/flixopt_stress.py --timesteps 547 --periods 16

The fluxopt side of the comparison is `python -m fluxopt.benchmark stress
--timesteps <n*16>` (its --timesteps budget is split across the 16 periods).
Measured numbers and methodology: docs/benchmark.md, "Comparison with flixopt".
"""

from __future__ import annotations

import argparse
import resource
import time as _time

import flixopt as fx
import numpy as np
import pandas as pd
import xarray as xr

GAP_PATTERN = [2, 1, 2, 2, 1, 3]


def _periods(n: int, start: int = 2025) -> list[int]:
    years = [start]
    for i in range(n - 1):
        years.append(years[-1] + GAP_PATTERN[i % len(GAP_PATTERN)])
    return years


def _clock(n: int):
    t = np.arange(n)
    return t % 24, t // 24, (t // 24 % 7) >= 5


def _winter(day):
    return 0.5 + 0.5 * np.cos(2 * np.pi * day / 365.0)


def build_system(n_timesteps: int = 240, n_periods: int = 16) -> fx.FlowSystem:
    rng = np.random.default_rng(7)
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h', name='time')
    years = _periods(n_periods)
    periods = pd.Index(years, name='period')

    hours = np.arange(n_timesteps)
    seasonal = 0.5 + 0.4 * np.cos(2 * np.pi * hours / 8760)
    daily = 0.1 * np.sin(2 * np.pi * hours / 24)
    escalation = xr.DataArray([1.025 ** (y - years[0]) for y in years], coords=(periods,))
    npv = np.array([float(1 / 1.035 ** (y - years[0])) for y in years])

    def tprofile(base: float, amp: float = 0.2) -> xr.DataArray:
        vals = base * (1 + amp * (seasonal - 0.5) + daily + 0.05 * rng.standard_normal(n_timesteps))
        return xr.DataArray(vals, coords=(timesteps,))

    def tp_price(lo: float = 20, hi: float = 100) -> xr.DataArray:
        return tprofile(float(rng.uniform(lo, hi))) * escalation  # (time, period)

    def availability(floor: float = 0.05) -> xr.DataArray:
        return xr.DataArray(
            np.clip(0.4 + seasonal + 0.02 * rng.standard_normal(n_timesteps), floor, 1.0), coords=(timesteps,)
        )

    def tp_coeff() -> xr.DataArray:
        return xr.DataArray(np.clip(tprofile(float(rng.uniform(1.5, 4.0)), 0.4).values, 1.2, 6.0), coords=(timesteps,))

    demand = xr.DataArray(
        np.clip(seasonal + daily + 0.03 * rng.standard_normal(n_timesteps), 0.05, 1.0), coords=(timesteps,)
    ) * xr.DataArray([0.985**i for i in range(n_periods)], coords=(periods,))
    peak = float(rng.uniform(400, 600))
    level_cap = xr.DataArray(np.clip(0.6 + 0.4 * seasonal, 0.3, 1.0), coords=(timesteps,))

    def window(y0: int, y1: int) -> xr.DataArray:
        return xr.DataArray([1.0 if y0 <= y <= y1 else 0.0 for y in years], coords=(periods,))

    def lump_effects(y0: int) -> dict:
        cost = float(rng.uniform(3e4, 3e6))
        funding = float(rng.uniform(0.2, 0.45))
        amort = int(rng.integers(10, 25))
        rate = 0.04
        af = rate * (1 + rate) ** amort / ((1 + rate) ** amort - 1)
        ann = window(y0, y0 + amort - 1) * cost * af
        one = xr.DataArray([cost if y == y0 else 0.0 for y in years], coords=(periods,))
        return {
            'leaf_ba': ann,
            'leaf_ga': ann * funding,
            'leaf_fi': ann * 0.08,
            'leaf_bt': one,
            'leaf_gt': one * funding,
            'cap_min': window(y0, y0 + amort + 4),
            'leaf_l': window(y0, y0 + amort + 4) * 0.02,
        }

    def invest(size_min, size_max, mandatory, per_size=None, fixed=None) -> fx.InvestParameters:
        """fluxopt Sizing -> InvestParameters; min==max & optional -> fixed_size."""
        if size_min == size_max and not mandatory:
            return fx.InvestParameters(
                fixed_size=size_min,
                mandatory=False,
                effects_of_investment_per_size=per_size,
                effects_of_investment=fixed,
            )
        return fx.InvestParameters(
            minimum_size=size_min,
            maximum_size=size_max,
            mandatory=mandatory,
            effects_of_investment_per_size=per_size,
            effects_of_investment=fixed,
        )

    def cap_charge_input(bus: str, sid: str, kind: str, extra: dict | None = None) -> fx.Flow:
        effects = {'leaf_f' if kind == 'fuel' else 'leaf_p': tp_price(), **(extra or {})}
        return fx.Flow(
            sid,
            bus=bus,
            effects_per_flow_hour=effects,
            size=invest(0, 1000, False, per_size={'leaf_om': float(rng.uniform(2e4, 1e5)) * escalation}),
        )

    sites = ['n1', 'n2', 'n3']
    # No k1_m1: flixopt rejects buses without flows (fluxopt tolerates the unused node).
    buses = [
        fx.Bus('k0'),
        fx.Bus('k1_m0'),
        *[fx.Bus(f'k2_{n}') for n in ('m0', *sites)],
        *[fx.Bus(f'k3_{n}') for n in ('m0', *sites)],
        fx.Bus('k7'),
    ]

    annual_demand = float(demand.isel(period=0).sum('time')) * peak
    per_period = lambda vals: xr.DataArray(np.asarray(vals, dtype=float), coords=(periods,))  # noqa: E731
    effects = [
        fx.Effect(
            'cost',
            'u',
            is_objective=True,
            period_weights=npv,
            share_from_temporal={'agg_fix': 1, 'agg_op': 1, 'agg_env': 0.0625},
            share_from_periodic={'agg_fix': 1, 'agg_op': 1, 'agg_env': 0.0625},
        ),
        fx.Effect(
            'agg_op',
            'u',
            share_from_temporal={'leaf_f': 1, 'leaf_p': 1, 'leaf_m': 1, 'leaf_r': -1, 'leaf_s': -1},
            share_from_periodic={'leaf_f': 1, 'leaf_p': 1, 'leaf_m': 1, 'leaf_r': -1, 'leaf_s': -1},
        ),
        fx.Effect('leaf_f', 'u'),
        fx.Effect('leaf_p', 'u'),
        fx.Effect('leaf_m', 'u'),
        fx.Effect('leaf_r', 'u'),
        fx.Effect('leaf_s', 'u'),
        fx.Effect(
            'agg_fix',
            'u',
            share_from_temporal={'leaf_ba': 1, 'leaf_om': 1, 'leaf_fi': 1, 'leaf_ga': -1},
            share_from_periodic={'leaf_ba': 1, 'leaf_om': 1, 'leaf_fi': 1, 'leaf_ga': -1},
        ),
        fx.Effect('leaf_ba', 'u'),
        fx.Effect('leaf_om', 'u'),
        fx.Effect('leaf_fi', 'u'),
        fx.Effect('leaf_ga', 'u'),
        fx.Effect(
            'agg_cap',
            'u',
            share_from_temporal={'leaf_bt': 1, 'leaf_gt': -1, 'agg_net': 1},
            share_from_periodic={'leaf_bt': 1, 'leaf_gt': -1, 'agg_net': 1},
        ),
        fx.Effect('leaf_bt', 'u'),
        fx.Effect('leaf_gt', 'u'),
        fx.Effect(
            'agg_net',
            'u',
            share_from_temporal={'leaf_nc': 1, 'leaf_ng': -1},
            share_from_periodic={'leaf_nc': 1, 'leaf_ng': -1},
        ),
        fx.Effect('leaf_nc', 'u'),
        fx.Effect('leaf_ng', 'u'),
        fx.Effect(
            'agg_env',
            'u',
            share_from_temporal={'net_x': 1.0, 'leaf_w': 0.25, 'leaf_l': 0.125},
            share_from_periodic={'net_x': 1.0, 'leaf_w': 0.25, 'leaf_l': 0.125},
        ),
        fx.Effect('leaf_xs', 'u'),
        fx.Effect(
            'net_x',
            'u',
            maximum_periodic=per_period([999_999.0] * n_periods),
            share_from_temporal={'leaf_x': 1, 'leaf_xs': -1},
            share_from_periodic={'leaf_x': 1, 'leaf_xs': -1},
        ),
        fx.Effect('leaf_x', 'u'),
        fx.Effect('leaf_w', 'u'),
        fx.Effect('leaf_l', 'u'),
        fx.Effect(
            'cap_min',
            'MW',
            minimum_periodic=per_period([peak * 1.05 * 0.985**i for i in range(n_periods)]),
        ),
        fx.Effect(
            'share_min',
            'MWh',
            minimum_periodic=per_period([annual_demand * min(0.6, 0.05 + 0.04 * i) * 0.1 for i in range(n_periods)]),
        ),
        fx.Effect('zone_max', 'MW', maximum_periodic=8.0),
        fx.Effect(
            'quota_a',
            'h',
            maximum_periodic=per_period([3000.0] * n_periods),
            maximum_total=20_000.0,
            period_weights=np.ones(n_periods),
        ),
        fx.Effect(
            'quota_b',
            'h',
            maximum_periodic=per_period([3000.0] * n_periods),
            maximum_total=10_000.0,
            period_weights=np.ones(n_periods),
        ),
        fx.Effect('pair_limit', '', minimum_periodic=-0.15, maximum_periodic=0.15),
    ]

    components: list = [
        fx.Sink('sink_0', inputs=[fx.Flow('load', bus='k0', size=peak, fixed_relative_profile=demand)]),
        fx.Source('src_k2', outputs=[fx.Flow('buy', bus='k2_m0', size=12_000)]),
        fx.Source('src_k3', outputs=[fx.Flow('buy', bus='k3_m0', size=invest(4000, 4000, False))]),
        fx.Source('hub_k1_buy', outputs=[fx.Flow('buy', bus='k1_m0', size=6000)]),
        fx.Sink(
            'hub_k1_sell',
            inputs=[fx.Flow('sell', bus='k1_m0', size=6000, effects_per_flow_hour={'leaf_r': tp_price()})],
        ),
        fx.Source('src_k7', outputs=[fx.Flow('buy', bus='k7', size=6000)]),
    ]

    components.extend(
        fx.LinearConverter(
            f'bridge_{site}',
            inputs=[fx.Flow('a_in', bus='k2_m0'), fx.Flow('b_in', bus='k3_m0')],
            outputs=[
                fx.Flow('a_out', bus=f'k2_{site}', size=invest(8_000, 8_000, False)),
                fx.Flow('b_out', bus=f'k3_{site}', size=invest(8_000, 8_000, False)),
            ],
            conversion_factors=[{'a_in': 1, 'a_out': -1}, {'b_in': 1, 'b_out': -1}],
        )
        for site in sites
    )

    for i in range(5):
        eff = float(rng.uniform(0.75, 0.95))
        components.append(
            fx.LinearConverter(
                f'conv_a{i}',
                inputs=[cap_charge_input(f'k2_{sites[i % len(sites)]}', 'fuel', 'fuel', extra={'leaf_x': 0.2})],
                outputs=[
                    fx.Flow(
                        'out',
                        bus='k0',
                        size=invest(
                            0,
                            float(rng.uniform(80, 260)),
                            True,
                            per_size={'leaf_om': float(rng.uniform(3e3, 3e4)) * escalation, **lump_effects(years[0])},
                        ),
                    )
                ],
                conversion_factors=[{'fuel': eff, 'out': -1}],
            )
        )

    for name, budget, site in (('conv_b0', 'quota_a', 'n1'), ('conv_b1', 'quota_b', 'n2')):
        components.append(
            fx.LinearConverter(
                name,
                conversion_factors=[
                    {'fuel': 0.5, 'out': -1},
                    {'fuel': 0.4, 'aux': -1, 'aux_q': -1},
                ],
                inputs=[cap_charge_input(f'k2_{site}', 'fuel', 'fuel', extra={'leaf_x': 0.4})],
                outputs=[
                    fx.Flow(
                        'out',
                        bus='k0',
                        size=invest(
                            0,
                            float(rng.uniform(60, 120)),
                            True,
                            per_size={'leaf_om': float(rng.uniform(1e4, 5e4)) * escalation, **lump_effects(years[0])},
                        ),
                    ),
                    fx.Flow('aux', bus='k1_m0', effects_per_flow_hour={'leaf_r': tp_price(), 'leaf_xs': 0.4}),
                    fx.Flow(
                        'aux_q',
                        bus='k1_m0',
                        effects_per_flow_hour={
                            'leaf_r': tp_price(),
                            'leaf_s': 30.0 * escalation,
                            budget: 1 / 100,
                            'leaf_xs': 0.4,
                        },
                    ),
                ],
            )
        )

    components.append(
        fx.LinearConverter(
            'conv_b2',
            conversion_factors=[{'fuel': 0.5, 'aux': -1}, {'fuel': 0.4, 'out': -1}],
            inputs=[cap_charge_input('k3_n1', 'fuel', 'fuel')],
            outputs=[
                fx.Flow('aux', bus='k1_m0', effects_per_flow_hour={'leaf_r': tp_price()}),
                fx.Flow(
                    'out',
                    bus='k0',
                    effects_per_flow_hour={'share_min': 1},
                    size=invest(0, 80, False, per_size=lump_effects(years[min(4, n_periods - 1)])),
                ),
            ],
        )
    )

    components.append(
        fx.LinearConverter(
            'conv_d',
            inputs=[cap_charge_input('k1_m0', 'drive', 'power')],
            outputs=[
                fx.Flow(
                    'out',
                    bus='k0',
                    effects_per_flow_hour={'share_min': 1},
                    load_factor_max=0.9,
                    size=invest(0, 70, False, per_size=lump_effects(years[min(6, n_periods - 1)])),
                )
            ],
            conversion_factors=[{'drive': 30, 'out': -1}],
        )
    )

    avail = np.ones(n_timesteps)
    for block_start in range(0, n_timesteps, max(1, n_timesteps // 5)):
        avail[block_start : block_start + max(1, n_timesteps // 50)] = 0.0
    components.append(
        fx.LinearConverter(
            'pw_unit',
            inputs=[fx.Flow('fuel', bus='k7', size=invest(0, 50, True))],
            outputs=[
                fx.Flow('aux', bus='k1_m0', size=6.0, effects_per_flow_hour={'leaf_r': tp_price()}),
                fx.Flow(
                    'out',
                    bus='k0',
                    effects_per_flow_hour={'share_min': 1},
                    relative_maximum=xr.DataArray(avail, coords=(timesteps,)),
                    size=invest(0, 43, True),
                ),
            ],
            status_parameters=fx.StatusParameters(),
            piecewise_conversion=fx.PiecewiseConversion(
                {
                    'fuel': fx.Piecewise([fx.Piece(50, 50)]),
                    'aux': fx.Piecewise([fx.Piece(6.0, 0.5)]),
                    'out': fx.Piecewise([fx.Piece(37, 43)]),
                }
            ),
        )
    )

    components.append(
        fx.LinearConverter(
            'pair_unit',
            inputs=[
                cap_charge_input('k1_m0', 'drive', 'power'),
                cap_charge_input('k1_m0', 'drive_2', 'power'),
            ],
            outputs=[
                fx.Flow(
                    'out',
                    bus='k0',
                    effects_per_flow_hour={'share_min': 1, 'pair_limit': 1},
                    size=invest(
                        0,
                        8.0,
                        True,
                        per_size=lump_effects(years[1] if n_periods > 1 else years[0]),
                        fixed={'pair_limit': window(years[0], years[-1])},
                    ),
                ),
                fx.Flow(
                    'out_2',
                    bus='k0',
                    effects_per_flow_hour={'share_min': 1},
                    relative_minimum=0.02,
                    status_parameters=fx.StatusParameters(effects_per_active_hour={'pair_limit': 4.0}),
                    size=invest(2.5, 2.5, False, fixed={'pair_limit': -window(years[0], years[-1])}),
                ),
            ],
            conversion_factors=[
                {'drive': 3.0, 'out': -1},
                {'drive_2': tp_coeff(), 'out_2': -1},
            ],
        )
    )

    for i in range(7):
        coeff = tp_coeff()
        components.append(
            fx.LinearConverter(
                f'conv_e{i}',
                inputs=[cap_charge_input('k1_m0', 'drive', 'power', extra={'leaf_s': tp_price(10, 40)})],
                outputs=[
                    fx.Flow(
                        'out',
                        bus='k0',
                        effects_per_flow_hour={'share_min': 1, 'leaf_m': 4 * (coeff - 1) / coeff * escalation},
                        relative_maximum=availability(),
                        size=invest(
                            0,
                            float(rng.uniform(10, 60)),
                            (i % 2 == 0),
                            per_size={
                                'leaf_om': float(rng.uniform(2e4, 6e4)) * escalation,
                                'zone_max': 0.5 if i >= 4 else 0.0,
                                **lump_effects(years[min(i, n_periods - 1)]),
                            },
                        ),
                    )
                ],
                conversion_factors=[{'drive': coeff, 'out': -1}],
            )
        )

    coeff_fleet = tp_coeff()
    for pi, y in enumerate(years):
        site = sites[pi % len(sites)]
        if pi >= 2:
            components.append(
                fx.LinearConverter(
                    f'gen_a{y}',
                    conversion_factors=[
                        {'b_in': 0.9, 'out_b': -1},
                        {'a_in': 0.9, 'out_a': -1},
                    ],
                    inputs=[
                        cap_charge_input(f'k2_{site}', 'a_in', 'fuel', extra={'leaf_x': 0.2}),
                        cap_charge_input(f'k3_{site}', 'b_in', 'fuel'),
                    ],
                    outputs=[
                        fx.Flow(
                            'out_b',
                            bus='k0',
                            effects_per_flow_hour={'share_min': 1},
                            size=invest(
                                5,
                                120,
                                False,
                                per_size={
                                    'leaf_om': float(rng.uniform(2e3, 5e3)) * escalation,
                                    **lump_effects(y),
                                },
                            ),
                        ),
                        fx.Flow('out_a', bus='k0', size=invest(0, 120, False)),
                    ],
                )
            )
            components.append(
                fx.LinearConverter(
                    f'gen_b{y}',
                    inputs=[cap_charge_input('k1_m0', 'drive', 'power', extra={'leaf_s': tp_price(10, 40)})],
                    outputs=[
                        fx.Flow(
                            'out',
                            bus='k0',
                            effects_per_flow_hour={'share_min': 1},
                            relative_maximum=availability(),
                            size=invest(
                                5,
                                60,
                                False,
                                per_size={
                                    'leaf_om': float(rng.uniform(3e4, 6e4)) * escalation,
                                    **lump_effects(y),
                                },
                            ),
                        )
                    ],
                    conversion_factors=[{'drive': coeff_fleet, 'out': -1}],
                )
            )
        if pi >= 1:
            components.append(
                fx.LinearConverter(
                    f'gen_c{y}',
                    inputs=[cap_charge_input('k1_m0', 'drive', 'power', extra={'leaf_s': tp_price(10, 40)})],
                    outputs=[
                        fx.Flow(
                            'out',
                            bus='k0',
                            effects_per_flow_hour={'share_min': 1},
                            relative_maximum=availability(),
                            size=invest(
                                2,
                                20,
                                False,
                                per_size={
                                    'leaf_om': float(rng.uniform(2e4, 5e4)) * escalation,
                                    'zone_max': 1.0,
                                    **lump_effects(y),
                                },
                            ),
                        )
                    ],
                    conversion_factors=[{'drive': coeff_fleet, 'out': -1}],
                )
            )

    def storage_flow(sid: str, cap) -> fx.Flow:
        effects = {'leaf_p': tp_price(1, 5)}
        if sid == 'discharging':
            effects['leaf_w'] = 0.05
        return fx.Flow(sid, bus='k0', size=cap, relative_maximum=level_cap, effects_per_flow_hour=effects)

    for i, (cap, rate) in enumerate(((1000, 100), (3000, 150))):
        components.append(
            fx.Storage(
                f'store_fixed_{i}',
                eta_charge=float(rng.uniform(0.95, 0.99)),
                eta_discharge=float(rng.uniform(0.95, 0.99)),
                relative_loss_per_hour=float(rng.uniform(2e-4, 6e-4)),
                relative_maximum_charge_state=level_cap,
                capacity_in_flow_hours=invest(
                    0,
                    float(cap),
                    True,
                    per_size=lump_effects(years[min(i, n_periods - 1)]) if i else None,
                ),
                charging=storage_flow('charging', invest(0, float(rate), True)),
                discharging=storage_flow('discharging', invest(0, float(rate * 1.1), True)),
                initial_charge_state='equals_final',
                prevent_simultaneous_charge_and_discharge=False,
            )
        )
    for pi, y in enumerate(years):
        if pi < 4:
            continue
        components.append(
            fx.Storage(
                f'store_{y}',
                eta_charge=0.99,
                eta_discharge=0.99,
                relative_loss_per_hour=0.00025,
                relative_maximum_charge_state=level_cap,
                capacity_in_flow_hours=invest(
                    600,
                    40_000,
                    False,
                    per_size=lump_effects(y),
                    fixed={'leaf_om': float(rng.uniform(2e5, 5e5)) * escalation * window(y, y + 29)},
                ),
                charging=storage_flow('charging', invest(25, 800, False)),
                discharging=storage_flow('discharging', invest(25, 800, False)),
                initial_charge_state='equals_final',
                prevent_simultaneous_charge_and_discharge=False,
            )
        )

    fs = fx.FlowSystem(timesteps, periods=periods, weight_of_last_period=4.0)
    fs.add_elements(*buses, *effects, *components)
    return fs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--timesteps', type=int, default=240, help='timesteps per period')
    ap.add_argument('--periods', type=int, default=16)
    args = ap.parse_args()

    t0 = _time.perf_counter()
    fs = build_system(args.timesteps, args.periods)
    t1 = _time.perf_counter()
    print(f'system:       {t1 - t0:6.1f}s')

    fs.build_model()
    t2 = _time.perf_counter()
    m = fs.model
    print(f'build_model:  {t2 - t1:6.1f}s')
    print(f'variables:    {int(m.nvars):,} ({int(m.binaries.nvars):,} binary)')
    print(f'constraints:  {int(m.ncons):,}')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f'peak RSS:     {peak:6.2f} GB')


if __name__ == '__main__':
    main()
