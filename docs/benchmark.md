# Benchmark

fluxopt ships a user-runnable benchmark that builds a few realistic energy
systems and reports how fast the build pipeline
(`Elements → ModelData → linopy model`) runs on *your* hardware — and how much
memory it peaks at:

```console
$ python -m fluxopt.benchmark
fluxopt 0.9.0 — build-pipeline benchmark
Python 3.13.2 · Darwin arm64 · 8 CPUs
8760 hourly timesteps (1.0 years)

model                grid  comps  flows  effects  series  variables  binary  constraints  elements    data   build  peak rss
----------------------------------------------------------------------------------------------------------------------------
district_heating   8760x1      8     15        2       6       140k       0         298k      6 ms   64 ms  170 ms   200 MiB
industry_park      8760x1      9     16        2       5       289k     79k         534k      5 ms   70 ms  429 ms   232 MiB
green_city         8760x1     15     25        3      10       245k       0         499k      6 ms   67 ms  211 ms   229 MiB
energy_transition  8760x8     15     25        3      10      1.96M       0        3.85M     48 ms   46 ms  488 ms   947 MiB
stress             547x16     83    204       30     270      1.99M     55k        4.69M     9.2 s  534 ms   2.6 s   5.2 GiB
```

The label columns are each system's size: the temporal grid
(timesteps × periods), element stats (components, flows, effects, and the
number of time-varying series), and the measured solver model (variables,
binaries, constraints).

**peak rss** is the whole build subprocess's OS-level high-water mark — it
catches every allocation (numpy buffers, solver C libraries) but includes the
~140 MiB interpreter-and-imports footprint and allocator slack: the number
that has to fit in your RAM. For allocator-level numbers (net of the
interpreter, attributable to code), run the same systems under
[pytest-benchmem](https://github.com/fluxopt/pytest-benchmem) via the
repository's `benchmark/test_reference.py` — that is what the CodSpeed
dashboard and the PR benchmark hint report.

Each system is built in a fresh subprocess, so peak memory is attributed per
model, and all input data is deterministic — two runs of the same version on
the same machine measure the same workload.

## The reference systems

The models are realistic and readable — constant and time-varying data,
several effects, and cross-effect couplings (CO₂ priced into cost at
45 €/t via `Effect.contribution_from`). Their builders in
`fluxopt/benchmark.py` double as worked examples:

- **`district_heating`** — a municipal utility: gas boiler, ramp-limited CHP
  and an air-source heat pump with a weather-driven COP feed a 20 MW-peak
  heat network backed by a hot-water tank. Seasonal gas tariff, day-ahead
  electricity prices, hourly grid CO₂ intensity.
- **`industry_park`** — a factory site: two steam boilers with minimum load,
  minimum up/down times and startup costs (unit commitment), a gas-engine CHP
  with a piecewise-linear part-load curve, and investment decisions for an
  electrode boiler and a steam accumulator with annualized capital cost and
  embodied CO₂. Site emissions carry an annual CO₂ cap.
- **`green_city`** — a sector-coupled city: a wind PPA with a contracted
  annual energy cap, rooftop PV and a grid connection supply the city load, a
  battery sized by the optimizer (with a 10 % reserve level), and two
  district-heating networks. Tracks cost, CO₂ and primary energy.
- **`energy_transition`** — `green_city` planned over eight five-year
  investment periods (2025–2060), each represented by a full hourly year:
  demand grows with electrification, grid CO₂ intensity falls, the carbon
  price rises from 45 to 130 €/t, and the battery becomes a multi-period
  `Investment` — 15-year lifetime, overnight capex falling along a learning
  curve, recurring O&M.
- **`stress`** — the exception to realistic and readable: an abstract,
  structure-only stress workload with neutral ids and fixed-seed random
  parameters. ~190 flows over 16 irregularly spaced investment periods, a
  30-effect share graph, optional `Sizing` in bulk, multi-node balances and
  piecewise/status/storage features; its `--timesteps` budget is split
  across the periods.

## Options

```console
$ python -m fluxopt.benchmark district_heating   # a single system
$ python -m fluxopt.benchmark --timesteps 720    # one month instead of a year
$ python -m fluxopt.benchmark --solve            # also time the HiGHS solve
$ python -m fluxopt.benchmark --json             # machine-readable output
```

The solve is excluded by default: solver time depends on HiGHS, not on
fluxopt, and is much less deterministic than the build.

!!! note "Regression benchmarks"
    This command answers "how fast is fluxopt *here*". Tracking performance
    *between versions* is done by the CodSpeed suite in the repository's
    `benchmark/` directory.

## Comparison with flixopt

The `stress` system exists as a 1:1 port to
[flixopt](https://github.com/flixOpt/flixopt)
(`benchmark/flixopt_stress.py` in the repository), so the two frameworks can
be measured building the **identical system specification**: the same
fixed-seed rng draw order produces the same parameters, and the binary
variable counts match exactly at every scale — the models describe the same
MILP.

**Measured 2026-07-24** — fluxopt `main` @ `dfb5f31` vs flixopt `7.2.3`
(PyPI), both on linopy `0.7.0`, Python 3.13, Apple Silicon (8 cores,
24 GB RAM, macOS/arm64). Build pipeline only (no solve), each run in a fresh
process; time is the full element-construction-to-linopy-model wall time,
peak is whole-process RSS.

Solver-model size for the identical spec:

| grid (t×p) | binaries (both) | flixopt vars / cons | fluxopt vars / cons |
| :--- | ---: | ---: | ---: |
| 24×16 | 5,200 | 232k / 348k | 94k / 213k |
| 136×16 | 15,952 | 1.21M / 1.87M | 499k / 1.17M |
| 240×16 | 25,936 | 2.12M / 3.28M | 875k / 2.06M |
| 547×16 | 55,408 | 4.80M / 7.44M | 1.99M / 4.69M |
| 730×16 | 72,976 | 6.40M / 9.93M | 2.65M / 6.26M |

Build cost:

| grid (t×p) | flixopt time | fluxopt time | flixopt peak RSS | fluxopt peak RSS |
| :--- | ---: | ---: | ---: | ---: |
| 24×16 | 20.9 s | 1.5 s | 0.26 GB | 0.39 GB |
| 136×16 | 20.4 s | 3.6 s | 0.43 GB | 1.54 GB |
| 240×16 | 21.5 s | 5.6 s | 0.58 GB | 2.60 GB |
| 547×16 | 20.7 s | 10.2 s | 1.23 GB | 5.18 GB |
| 730×16 | 20.5 s | 13.3 s | 1.51 GB | 6.22 GB |

What the numbers say:

- **fluxopt emits a ~2.4× smaller solver model** for the same specification
  (2.65M vs 6.40M variables, 6.26M vs 9.93M constraints at 730×16, equal
  binaries). Every solver iteration afterwards works on the leaner model.
- **fluxopt builds faster at every tested scale** — 14× on small models,
  1.5× at 730×16. flixopt's build time is flat (~20.5 s) regardless of
  scale, dominated by per-element overhead; fluxopt's grows with the data
  but from a near-zero base, with the element-validation phase as its
  largest share.
- **flixopt's build-time memory footprint is lower at scale** (1.5 GB vs
  6.2 GB at 730×16). fluxopt's peak sits in the dense per-flow effect
  tensors of the data layer.

Replica fidelity: the port keeps flixopt idiomatic (`Bus` per carrier:node,
`share_from_temporal`/`share_from_periodic` for the effect graph,
`InvestParameters`, `StatusParameters`). Known deviations, all minor: flixopt
rejects a bus without flows, so the one unused node is dropped; the
piecewise unit's availability is applied as `relative_maximum` instead of
scaling the upper breakpoint; storage cycling uses
`initial_charge_state='equals_final'`.

Reproduce (per-period steps × 16 periods; fluxopt's `--timesteps` budget is
the product):

```console
$ python -m fluxopt.benchmark stress --timesteps 8752 --json    # 547×16
$ uv run --no-project --with 'flixopt==7.2.3' \
    python benchmark/flixopt_stress.py --timesteps 547 --periods 16
```
