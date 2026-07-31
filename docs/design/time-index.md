# Time index: flat `time` dim with per-period coordinates

Decision record for how fluxopt represents time in multi-period models.
Supersedes the orthogonal `time × period` grid for operational data.
Implemented in PR [#213](https://github.com/fluxopt/fluxopt/pull/213),
whose description holds the full alternatives analysis, implementation
notes and migration history; this file keeps only the binding rules.

## Decision

One temporal dimension, named `time`, spanning the whole horizon:

```
time          dim     all timesteps across all periods (real timestamps or ints)
time_period   coord   (time,) — investment period each timestep belongs to
dt            data    (time,) — timestep duration [h], may differ per period
weights       data    (time,) — operational weights (representative-period
                      occurrence counts; 1.0 by default)
period        dim     unchanged — investment-period axis (build/active
                      variables, periodic effect bounds)
period_weights data   (period,) — unchanged, period duration/discount weights
```

Why flat rather than the alternatives:

- **Orthogonal `time × period` grid** (previous design): cannot express
  ragged periods (hourly 2030 + 4-hourly 2050) or real calendar labels.
- **pandas MultiIndex snapshots** (PyPSA style): no NetCDF round-trip,
  second-class xarray semantics.
- **Masked dense grid**: ragged-capable but degrades `time` to a
  positional index — kills datetime features on model variables,
  which custom constraints need.

Key rules:

- The dim keeps the name `time`; `snapshot` exists only at the PyPSA
  interop boundary (`rename` + MultiIndex assembly reproduces PyPSA's
  snapshot index exactly).
- The coord is `time_period`, not `period` — coordinate names are global
  within a Dataset and `period` is reserved for the investment dim.
  Vectorized indexing (`sel(period=time_period)`) does not care.
- Single code path: single-period models are the flat dim with no
  `time_period` coord; boundary masks are no-ops with one episode.

## Episodes

An *episode* is a maximal run of timesteps consecutive in modeled time;
temporal-coupling constraints (SOC recursion, status windows, switch
transitions, ramps) never chain across episode boundaries. A *period* is
an accounting unit. Today they coincide 1:1 (boundaries where
`time_period` changes); under TSA one period will contain many episodes.
The concept is explained for users in the multi-period tutorial; the
machinery lives in the `Episodes` value class
(`fluxopt.constraints.episodes`), built via `Dims.episodes`.

The constraint helpers **require** an explicit `Episodes` — no
single-episode default, so a multi-period model cannot silently chain
across period boundaries by omission; `Episodes.single(...)` is the
explicit opt-out.

Constraint semantics:

- Temporal coupling resets at episode starts; `prior_level`,
  `prior_rates` and `Storage.cyclic` apply per episode.
- Per-period aggregates (`flow_hours`, `load_factor`, Effect
  `periodic_min/max`) are `groupby('time_period')` reductions
  (`Dims.sum_time` / `mean_time`); effect scopes map onto the naming
  grammar: `rate` = per timestep, `periodic` = per period, `total` =
  `period_weights`-weighted sum.
- Period-scoped parameters expand onto flat time via vectorized
  indexing (`Dims.map_to_time`).

## User API and input policy

- Single-period: `timesteps=<index>` — unchanged.
- Uniform multi-period: `timesteps=idx, periods=[2030, 2040]` — the
  index is replicated per period; datetime labels shift by each period's
  year gap **to the first period**, so the first period keeps the base
  labels (a 2024 base with periods `[2030, 2040]` yields 2024 and 2034
  labels), and integer labels offset by the span. The shift is a constant
  whole-day offset, so time-of-day and dt are preserved exactly; dates
  after Feb 28 drift one calendar day where leap parity differs. Real,
  unique labels are required for datetime features on a flat dim.
- Ragged multi-period: `timesteps={2030: hourly_idx, 2050: coarse_idx}`
  — per-period grids; real timestamps required.

**No silent resampling.** Once resolutions differ per period, resampling
an intensive quantity needs `mean`, an extensive one `sum` — guessing
produces quietly wrong costs. Time-varying inputs must align to the flat
index, a `{period: series}` mapping, a `(period,)` array, or (uniform
mode) the within-period grid; mismatches raise. Conversion lives in
`_TimeMapper`.

## Datetime rules on coarse grids

1. **Weight by `dt`, always** — `(rate * dt).groupby(...)` is exact on
   any resolution mix; bare counts silently diverge.
2. **Labels are interval starts** — selections finer than a period's
   resolution match nothing and constraints built from them become
   silently vacuous.
3. **Boundary smearing** — a timestep belongs to a window iff its start
   does.

The planned mitigation is a `window_weights` interval-overlap helper
(fractional overlap weights from `time` + `dt`, loud error on dead
windows). **It must ship before ragged grids are promoted in user
docs.** Uniform mean-coarsening flattens spikes and understates
flexibility value — representative periods are the sharper instrument
for the same goal; docs must present them in that order.

## TSA outlook (representative periods)

TSA extends the **coords, not the dims**: representative timesteps
concatenate on flat `time` with a `time_cluster(time)` coord
(period-local cluster ids); `weights(time)` carries occurrence counts.
Episode boundaries become "where `time_period` *or* `time_cluster`
changes" (`Episodes.from_changes`) — same machinery, no constraint
changes. Different cluster counts per period are just more raggedness.
Seasonal storage (Kotzur superposition) later adds one new dim — the
original chronology with a `chronology_cluster` mapping — carrying only
inter-period linking variables, never operational data.

Datetime caveat under TSA: within-day features stay valid; calendar-scale
groupbys (`dt.month`) are silently wrong on representative days and must
route through the occurrence mapping.

### Package boundary (tsam / tsam-xarray)

Aggregation tooling owns the clustering lifecycle; **fluxopt never
learns what a cluster is** beyond episode boundaries and weights. The
contract is plain data, no cross-imports:

- Inbound: `{period: timesteps}`, `{period: profiles}`,
  `{period: weights}`, and — once cluster episodes land —
  `{period: cluster_labels}`. Episode boundaries are explicit inputs;
  fluxopt never infers them from calendar-day changes (breaks under
  segmentation and weekly clusters).
- Outbound: solution arrays keep `time` labels (the representatives'
  real timestamps) and the `time_period` coord — enough for the
  aggregator's stored mapping to disaggregate results.
- The clustering mapping is a study artifact (e.g. `clustering.json`),
  not model data; a run reproduces from `model.nc + clustering.json`.

fluxopt-side prerequisites, in order: public `weights` input;
per-segment `dt` for non-contiguous rep timesteps; cluster-episode
boundary resets. Until all three land, TSA feeds are only correct for
models without intra-period temporal coupling.

## Terminology

The temporal vocabulary sorts into three registers; names are generated
by five rules, recorded here so they survive contributors:

1. **Bare singular noun = a dim** (an axis variables can live on):
   `time`, `period`, `build_period`. Reserved for future dims:
   `cluster`, `chronology`, `scenario`, `episode` (if per-episode
   parameters ever need an axis, e.g. episode-indexed prior state).
2. **Ride-along coord on a dim = `<dim>_<what>`**: `time_period`,
   later `time_cluster`; on the chronology dim, `chronology_cluster`
   (not `cluster_of`). The prefix keeps the bare noun free for the dim.
3. **Unprefixed temporal data vars live on `time`** (`dt`, `weights`);
   anything on another dim carries the dim as prefix
   (`period_weights`, a future `cluster_weights`).
4. **`period` is accounting, `episode` is coupling topology** — neither
   register borrows the other's word. Element fields may say
   period-things (`periodic_min`), never episode-things; constraint
   helpers say episode-things, never period-things. `Storage.cyclic`
   and `prior_level` are episode semantics and documented as such.
5. **Audience split for borrowed words**: prose says "representative
   period" ↔ code says `cluster`; prose "investment period" ↔ code
   `period`; `snapshot` exists only at the PyPSA boundary. "Horizon" /
   "chunk" is reserved for the rolling-horizon driver's solver
   decomposition (state deliberately carries across chunk boundaries)
   and must never be called an episode.

## Accepted trade-offs

- Episode masks are a permanent tax on time-coupled constraints —
  centralized in `Episodes`, and the required parameter makes omission
  impossible, but "impossible to get wrong" became "correct if applied".
- Per-period reductions are groupbys, heavier than dim sums (tracked by
  the `multi_period` benchmark scenario).
- Identical-across-periods profiles materialize at full horizon length
  (data memory only; variable counts are unchanged, smaller for ragged).

## Resolved questions

1. **Uniform-grid label replication**: replicated datetime labels are
   shifted apart by the period gaps (anchored at the base year, first
   period unshifted) — datetime features need real, unique labels, and
   identical labels would collide on a flat dim.
2. **Integer time labels**: allowed single-period and uniform
   multi-period (running index); ragged multi-period requires real
   timestamps.
3. **`weights` stays separate from `dt`**: occurrence count and duration
   are different physical quantities that compose (a 4-hour segment
   occurring 12 times needs both); folding them PyPSA-style makes
   disaggregation and per-quantity weighting painful.

## Open questions

1. **Enforce divisor-of-24h frequencies** for ragged per-period grids
   (validation error) or document-and-warn only? Leaning: warn — hard
   enforcement blocks legitimate irregular grids (e.g. rolling-horizon
   stubs), and `dt`-weighting keeps the math correct regardless.
