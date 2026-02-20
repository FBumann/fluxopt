"""Quick performance smoketest for fluxopt model building.

Adapted from flixOpt PR #588 benchmark. Creates synthetic energy systems of
varying size and measures ModelData.build() + FlowSystem.build() + LP write.

Usage:
    uv run python scripts/benchmark_build.py
    uv run python scripts/benchmark_build.py --xl        # Include XL stress test
    uv run python scripts/benchmark_build.py --lp        # Also benchmark LP file write
    uv run python scripts/benchmark_build.py --xl -n 1   # XL with single iteration
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fluxopt import (
    Bus,
    Converter,
    Effect,
    Flow,
    FlowSystem,
    ModelData,
    Port,
    Sizing,
    Status,
    Storage,
)


@dataclass
class BenchmarkResult:
    name: str
    n_timesteps: int = 0
    n_flows: int = 0
    n_converters: int = 0
    n_storages: int = 0
    n_vars: int = 0
    n_cons: int = 0
    data_ms: float = 0.0
    build_ms: float = 0.0
    lp_ms: float = 0.0
    total_ms: float = 0.0
    lp_mb: float = 0.0


def create_system(
    *,
    n_timesteps: int = 720,
    n_converters: int = 20,
    n_storages: int = 5,
    n_effects: int = 2,
    with_status: bool = True,
    with_sizing: bool = True,
) -> dict:
    """Create element kwargs for ModelData.build()."""
    timesteps = pd.date_range('2024-01-01', periods=n_timesteps, freq='h')

    effects = [Effect('cost', is_objective=True)]
    effects.extend(Effect(f'eff_{i}') for i in range(1, n_effects))

    buses = [Bus('Elec'), Bus('Heat'), Bus('Gas')]

    rng = np.random.default_rng(42)
    base = 50 + 30 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
    heat_profile = np.clip((base + rng.normal(0, 5, n_timesteps)) / base.max(), 0.2, 1.0)
    elec_profile = np.clip((base * 0.5 + rng.normal(0, 3, n_timesteps)) / base.max(), 0.1, 1.0)
    gas_price = 30 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))
    elec_price = 50 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)

    def _effects_dict(base_cost: float | np.ndarray) -> dict[str, float | np.ndarray]:
        d: dict[str, float | np.ndarray] = {'cost': base_cost}
        for i in range(1, n_effects):
            d[f'eff_{i}'] = 0.2
        return d

    ports = [
        Port('GasGrid', imports=[Flow(bus='Gas', size=5000, effects_per_flow_hour=_effects_dict(gas_price))]),
        Port('ElecBuy', imports=[Flow(bus='Elec', size=2000, effects_per_flow_hour=_effects_dict(elec_price))]),
        Port('ElecSell', exports=[Flow(bus='Elec', size=1000, effects_per_flow_hour={'cost': -elec_price * 0.8})]),
        Port('HeatDemand', exports=[Flow(bus='Heat', size=1, fixed_relative_profile=heat_profile)]),
        Port('ElecDemand', exports=[Flow(bus='Elec', size=1, fixed_relative_profile=elec_profile)]),
    ]

    converters = []
    for i in range(n_converters):
        is_chp = i % 3 != 0

        size: float | Sizing = 150.0
        if with_sizing:
            size = Sizing(min_size=50, max_size=200, effects_per_size={'cost': 100})

        status: Status | None = None
        if with_status:
            status = Status(effects_per_startup={'cost': 500})

        if is_chp:
            fuel = Flow(bus='Gas', id='fuel')
            elec = Flow(bus='Elec', id='elec', size=100)
            heat = Flow(bus='Heat', id='heat', size=size, relative_minimum=0.2, status=status)
            converters.append(
                Converter(f'CHP_{i}', [fuel], [elec, heat], [{fuel: 0.35, elec: -1}, {fuel: 0.50, heat: -1}])
            )
        else:
            fuel = Flow(bus='Gas', id='fuel')
            heat = Flow(bus='Heat', id='heat', size=size, relative_minimum=0.2, status=status)
            converters.append(Converter(f'Boiler_{i}', [fuel], [heat], [{fuel: 0.90, heat: -1}]))

    storages = []
    for i in range(n_storages):
        cap: float | Sizing = 500.0
        if with_sizing:
            cap = Sizing(min_size=0, max_size=1000, mandatory=True, effects_per_size={'cost': 10})

        storages.append(
            Storage(
                f'Store_{i}',
                charging=Flow(bus='Heat', size=100),
                discharging=Flow(bus='Heat', size=100),
                capacity=cap,
                prior_level=0,
                eta_charge=0.95,
                eta_discharge=0.95,
                relative_loss_per_hour=0.001,
            )
        )

    return {
        'timesteps': timesteps,
        'buses': buses,
        'effects': effects,
        'ports': ports,
        'converters': converters or None,
        'storages': storages or None,
    }


def bench(name: str, *, iterations: int = 3, write_lp: bool = False, **system_kwargs) -> BenchmarkResult:
    """Benchmark data build + model build (+ optional LP write)."""
    result = BenchmarkResult(name=name)

    data_times = []
    build_times = []
    lp_times = []

    for _ in range(iterations):
        kwargs = create_system(**system_kwargs)

        t0 = time.perf_counter()
        data = ModelData.build(**kwargs)
        t1 = time.perf_counter()
        model = FlowSystem(data)
        model.build()
        t2 = time.perf_counter()

        data_times.append(t1 - t0)
        build_times.append(t2 - t1)

        if write_lp:
            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, 'model.mps')
                model.m.to_file(mps_path, progress=False)
                t3 = time.perf_counter()
                result.lp_mb = os.path.getsize(mps_path) / 1e6
            lp_times.append(t3 - t2)

        result.n_timesteps = len(data.time)
        result.n_flows = data.flows.bound_type.sizes.get('flow', 0)
        result.n_converters = len(data.converters.eq_mask.coords['converter']) if data.converters else 0
        result.n_storages = len(data.storages.capacity.coords['storage']) if data.storages else 0
        result.n_vars = model.m.nvars
        result.n_cons = model.m.ncons

    result.data_ms = float(np.mean(data_times)) * 1000
    result.build_ms = float(np.mean(build_times)) * 1000
    result.lp_ms = float(np.mean(lp_times)) * 1000 if lp_times else 0.0
    result.total_ms = result.data_ms + result.build_ms
    return result


def print_results(results: list[BenchmarkResult], *, write_lp: bool = False) -> None:
    rows = []
    for r in results:
        row = {
            'System': r.name,
            'Steps': r.n_timesteps,
            'Flows': r.n_flows,
            'Conv': r.n_converters,
            'Stor': r.n_storages,
            'Vars': r.n_vars,
            'Cons': r.n_cons,
            'Data (ms)': round(r.data_ms),
            'Build (ms)': round(r.build_ms),
            'Total (ms)': round(r.total_ms),
        }
        if write_lp:
            row['LP (ms)'] = round(r.lp_ms)
            row['LP (MB)'] = round(r.lp_mb, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def bench_write(system_kwargs: dict, iterations: int = 3) -> None:
    """Compare LP/MPS write backends on a single built model."""
    kwargs = create_system(**system_kwargs)
    data = ModelData.build(**kwargs)
    model = FlowSystem(data)
    model.build()
    n_vars = model.m.nvars
    n_cons = model.m.ncons
    print(f'  Model: {n_vars:,} vars, {n_cons:,} cons')
    print()

    backends = ['lp', 'lp-polars', 'mps']
    rows = []
    for backend in backends:
        ext = 'mps' if backend == 'mps' else 'lp'
        times = []
        size_mb = 0.0
        for _ in range(iterations):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, f'model.{ext}')
                t0 = time.perf_counter()
                model.m.to_file(path, io_api=backend, progress=False)
                t1 = time.perf_counter()
                size_mb = os.path.getsize(path) / 1e6
            times.append(t1 - t0)
        mean_ms = float(np.mean(times)) * 1000
        print(f'  {backend:12s}  {mean_ms:8.0f} ms  {size_mb:6.1f} MB')
        rows.append({'Backend': backend, 'Time (ms)': round(mean_ms), 'Size (MB)': round(size_mb, 1)})

    print()
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description='fluxopt build performance smoketest')
    parser.add_argument('--xl', action='store_true', help='Include XL stress test')
    parser.add_argument('--lp', action='store_true', help='Also benchmark LP file write')
    parser.add_argument('--write-bench', action='store_true', help='Compare LP write backends (lp vs lp-polars vs mps)')
    parser.add_argument('--iterations', '-n', type=int, default=3)
    args = parser.parse_args()

    if args.write_bench:
        print('linopy write backend comparison')
        print('=' * 50)
        configs = [
            ('Medium (720h, 20c)', {'n_timesteps': 720, 'n_converters': 20, 'n_storages': 5}),
            ('Large (720h, 50c)', {'n_timesteps': 720, 'n_converters': 50, 'n_storages': 10}),
        ]
        if args.xl:
            configs.append(
                ('XL (2000h, 300c, 50s)', {'n_timesteps': 2000, 'n_converters': 300, 'n_storages': 50}),
            )
        for name, sys_kwargs in configs:
            print(f'\n--- {name} ---')
            bench_write(sys_kwargs, iterations=args.iterations)
        return

    configs: list[tuple[str, dict]] = [
        (
            'Small (168h, 10c, basic)',
            {'n_timesteps': 168, 'n_converters': 10, 'n_storages': 2, 'with_status': False, 'with_sizing': False},
        ),
        ('Medium (720h, 20c, full)', {'n_timesteps': 720, 'n_converters': 20, 'n_storages': 5}),
        ('Large (720h, 50c)', {'n_timesteps': 720, 'n_converters': 50, 'n_storages': 10}),
        ('Many effects (720h, 5eff)', {'n_timesteps': 720, 'n_converters': 50, 'n_effects': 5}),
        ('Full year (8760h, 10c)', {'n_timesteps': 8760, 'n_converters': 10, 'n_storages': 3, 'with_status': False}),
    ]
    if args.xl:
        configs.append(
            ('XL (2000h, 300c, 50s)', {'n_timesteps': 2000, 'n_converters': 300, 'n_storages': 50}),
        )

    lp_note = ' + LP write' if args.lp else ''
    print(f'fluxopt build benchmark  (iterations={args.iterations}{lp_note})')
    print('=' * 80)

    results = []
    for name, kwargs in configs:
        print(f'  {name}...', end=' ', flush=True)
        r = bench(name, iterations=args.iterations, write_lp=args.lp, **kwargs)
        lp_part = f', lp={r.lp_ms:.0f}' if args.lp else ''
        print(f'{r.total_ms:.0f} ms  (data={r.data_ms:.0f}, build={r.build_ms:.0f}{lp_part})')
        results.append(r)

    print()
    print_results(results, write_lp=args.lp)


if __name__ == '__main__':
    main()
