# fluxopt on lpspec

fluxopt's modeling layer becomes [lpspec](https://github.com/fluxopt/lpspec).
linopy stays reachable as an optional lane and stops being a dependency of the
core. This page is the argument and the decisions; the work is a stack of PRs,
each of which links back here.

Two things make the change smaller than it sounds. `relational/core.yaml`
already states most of fluxopt's math, so the target is not a new artifact but
an existing one finished. And lpspec's own linopy lane builds the *same file*
as a `linopy.Model`, so nothing about this choice burns the linopy path — it
moves it behind an extra.

## The decision

**fluxopt has one math and it is a file.** `relational/core.yaml` is the whole
model; `sources.py` binds data to it; lpspec validates, lowers and streams it.
`model.py` is deleted rather than ported.

Today the math is stated twice — once as 1693 lines of linopy calls in
`model.py`, once as 553 lines of YAML in `core.yaml` — and
`tests/test_relational_parity.py` exists to catch the two drifting apart. That
test is the tell. It is insurance against a divergence that only exists because
the math was written twice, and it stops being needed the moment it stops being
possible.

## What replaces `customize`

`customize(model: FlowSystemModel)` — a Python callback reaching into linopy
between build and solve — does not survive, and gets no replacement in Python.
It was untyped, unreviewable and unserializable, and it is precisely what
lpspec's rule 5 refuses.

The replacement is that **the math is data you can edit**:

| you want to | the call |
|---|---|
| read the equations fluxopt will solve | `system.math()` → `lpspec.Model` |
| review or diff them | `.to_yaml()` |
| put them in a paper | `lps.to_latex(...)` · `to_markdown` · `to_typst` |
| name a quantity and read it back after a solve | add to `expressions:`, then `result.expression(name)` |
| add a constraint of your own | add to `constraints:`, bind its data through `parameters=` |
| replace one fluxopt emitted | edit or drop it in the dict, pass the model back |

Every rung is the same artifact: a `Model` goes in and out as a dict, so an
edited model is validated, dim-checked and lowered exactly as a file is. Two
consequences fall out of that and are the point rather than a side effect:

- **`core.yaml`'s declaration names become public API.** A user replacing
  `storage_balance` needs it to still be called that next release.
- **`docs/math/*.md` stops being written by hand.** Seven prose files currently
  describe what `model.py` does, with nothing enforcing that they still do.
  They become typeset output of the file that is actually solved.

The one piece of new plumbing is a `parameters=` binding channel alongside the
existing `profiles=`, so a user-added constraint can carry its own data.

## Why linopy becomes an extra rather than a lane fluxopt maintains

lpspec's `linopy` extra pins an **unreleased linopy branch** — the v1 arithmetic
convention, since no release carries `options['semantics']`. That has a
consequence worth writing down, because it was found by walking into it:

> uv resolves one version per package across the whole lock, **extras
> included**. Declaring `linopy = ["lpspec[linopy]"]` therefore pulls the branch
> linopy into the *base* environment, where `model.py` builds against release
> semantics and emits ~2900 `LinopySemanticsWarning`s.

So the extra is declared only in the commit that deletes `model.py`, which is
the commit after which nothing in fluxopt core imports linopy at all. Until
then the base install stays on released linopy and the existing suite stays
green — which matters, because `model.py` is the oracle the migration is tested
against and must keep working right up until it is deleted.

**Converting `model.py` to v1 semantics first was considered and rejected.** It
is ~2900 warnings' worth of fixes in the one module the plan deletes, and the
v1 requirement belongs to lpspec's lane rather than to fluxopt's builder.

## The layers after

```
elements.py · components.py    pydantic element layer          unchanged
flow_system.py                 validation + assembly           unchanged
        |
sources.py                     elements -> tidy frames         absorbs model_data.py
relational/core.yaml           the math                        absorbs model.py
        |
lpspec                         check · lower · stream · solve · typeset
        |
results.py                     a label join over result frames shrinks
```

The element layer does not move. It is the DX, it is good, and nothing about
the backend reaches it.

### What is deleted, and why it is structural

| Module | Lines | Why it goes |
|---|---|---|
| `model.py` | 1693 | the math, stated a second time |
| `constraints/` | 465 | `sparse.py` is a sparse weighted-sum kernel that exists because linopy is dense; lpspec is sparse by construction, where a mask is an absent row |
| `contributions.py` | 254 | effect decomposition reconstructed post-solve, then checked against the solver. Becomes a view over `expressions:` — see [what actually happened](#what-this-page-got-wrong) |
| `effect_terms.py` | 216 | a single declaration kept so `model.py` and `contributions.py` could not drift. One consumer, no drift, no abstraction |
| most of `contract.py` | 117 | the NaN-as-absent convention and `BoundType` dispatch are dense-array workarounds |
| most of `model_data.py` | 1872 | see below |

Roughly 4600 lines out against ~900 into `core.yaml` and `sources.py`.

### `ModelData` stops being dense xarray

Three things in `ModelData` are shaped for linopy rather than for fluxopt: NaN
as "absent", the regime dims (`sizing_flow`, `status_flow`) that are renamed
back to their entity dim at constraint time, and the padded ragged
`governed_flows` string array. Each is a workaround for linopy wanting aligned
dense arrays, and under lpspec each is simply *which rows exist*.

`sources.py` currently pays to undo all three — that is what
`_tidy(..., drop_zero=True)` and the `has_*` boolean tables are for. So the
middle layer becomes tidy frames and `sources.py` stops being an adapter on top
of a layer and becomes the layer. Serialization follows it from NetCDF to
parquet.

This is the largest break and it is sequenced last, because the parity test
lets it be deferred without blocking anything ahead of it.

## Decisions taken, so they are not re-argued

**One monolithic program, not per-component templates.** lpspec's ceiling notes
sketch component libraries as parametrised templates merged at build, which
needs namespacing and a native schema merge — neither shipped. fluxopt's
component vocabulary is fixed (`Port`, `Converter`, `Storage`), so one program
with `has_*` masks says everything a merge would, and needs no unshipped
feature. Cardinality lives in data either way.

**Topology is data.** Every relation a flow participates in is a lookup column
on the `flow` index — `component_of`, `carrier_of`, `charge_storage` — never an
adjacency matrix. `sum(by=)` reads them.

**No Python escape hatch.** Not `customize`, not registered helpers, not
generated YAML text. Math fluxopt cannot state is a gap to close in `core.yaml`
or upstream in the language, and it is visible as one either way.

**lpspec is pinned to a tag.** Its language surface is pre-1.0 and still moves;
an unpinned git ref would let a `uv sync` change what `core.yaml` means. Bumping
the pin is a deliberate change with the suite behind it.

## The two gaps, and that they are closed

`relational/__init__.py` names piecewise conversion and component status as
blocked on an *indexed lookup* — "the adjoint of `group_sum`, joining a mapping
table without aggregating" — described as planned upstream and not yet
existing. It exists, spelled `at(x, by=lookup)`, and both were verified against
the pinned tag before this stack was written.

**Component status** is `at()` reading one per-component decision onto each of
the flows that belong to it:

```yaml
lookups:
  component_of: {over: flow, into: component}
constraints:
  rate_below_status:
    foreach: [flow, time]
    expression: "rate <= rate_max * at(on, by=component_of)"
```

**Piecewise conversion** is lpspec's `piecewise:` formulation, and the shape
worth noting is that one declaration covers *every* converter — the links carry
`[component, time]`, so the curve is indexed by `[component, bp]` and
vectorizes:

```yaml
piecewise:
  curve:
    over: bp
    links:
      - ['sum(rate * is_fuel, by=component_of)', fuel_bp]
      - ['sum(rate * is_heat, by=component_of)', heat_bp]
    method: adjacency
    active: on
```

That replaces a per-converter Python loop calling
`linopy.piecewise.add_piecewise_formulation`, an API fluxopt currently
suppresses an `EvolvingAPIWarning` for.

Two spellings cost time to find and are recorded so they cost it once:
`piecewise:` links do **not** resolve named `expressions:`, so the expression is
inlined; and a lookup map binds as a column on the dimension's index table named
for the **lookup**, not for the dimension it maps into.

### What does not map

`PiecewiseConversion.method='lp'` is linopy's tangent-line formulation, which is
lpspec [#695](https://github.com/fluxopt/lpspec/issues/695) and not in the
language. `method: convex` is the hull, exact under optimisation pressure for a
curve of matching curvature — which is the condition fluxopt's own `'auto'`
already tests before picking LP. Whether that is a substitution or a behaviour
change is settled in the PR that ports piecewise, against the parity test.

## Order of work

Each step is a PR stacked on the one above it.

1. **Pin lpspec, rename `farkas`.** The repo moved to `fluxopt/lpspec` and the
   package with it. Declares lpspec a core dependency and leaves the linopy
   extra commented, for the resolver reason above.
2. **Migrate `core.yaml` to the pinned surface.** `group_sum(x, over=d, by=l)`
   → `sum(x, by=l)`; `binary: true` → `domain: binary`; `roll(...)` →
   `shift(..., edge='wrap')`; a dimension's `coords:` → a top-level `lookups:`
   block; `shift(level, time=1)` → `shift(level, over=time, by=1)`;
   `equations:` → `expression:`. `lps.check()` is the oracle and binds no data.
3. **Close piecewise and component status**, and drop
   `UnsupportedFeatureError`. Parity green across the full feature matrix is the
   gate: it is what "the YAML is the whole math" means, and nothing after it may
   start before it passes.
4. **`Result` off the lpspec result**, so both lanes answer with one object.
   Effect contributions become named expressions the ledger sums, and the
   `Result` carries every named quantity — `effect_terms.py` goes.
5. **Delete `model.py` and `constraints/`**; declare the linopy extra.
6. **Re-cut `ModelData` as tidy frames**, parquet for serialization.

Steps 2 and 3 carry the risk. Step 5 is mostly `git rm`.

## What this page got wrong

Written before the work, and two of its claims did not survive it. Left in
place above with this section as the correction, because a design note that
quietly edits itself is worth less than one that says what it learned.

**The parity test did not retire.** The plan assumed it goes with `model.py`,
both sides of it being hand-written models. Instead it became a *differential
between engines* — the same `program.yaml` built relationally and by lpspec's
eager lane — which is a sharper instrument than what it replaced: a
disagreement is now an engine bug rather than possibly a bug in either model.
It is skipped pending [lpspec#1108](https://github.com/fluxopt/lpspec/issues/1108),
and what stands in its place is `tests/math` + `tests/math_port` running on
the shipping lane. That is the stronger gate anyway — it was the math suite,
not parity, that caught flow aggregates going missing.

**`contributions.py` shrank rather than went, by a better route than
described.** The plan said each contribution becomes a named expression read
back with `result.expression(name)`. Costed that way it is a bad trade: the
program's coefficients are pre-scaled and Leontief-folded *for the solver*, so
reading contributions off them means either dividing the scaling back out or
carrying raw coefficients beside the folded ones — undoing the fold that took
the effects build from 18.5 s to 4.2 s.

What works instead is to **name the contribution and have the ledger sum that
very expression**:

```yaml
expressions:
  contribution_flow_hour:
    expression: rate * effects_per_flow_hour       # keeps the entity
  effect_temporal:
    expression: sum(sum(contribution_flow_hour * time_weight, over=time), over=flow) + ...
```

One declaration, two readings — no second implementation and no second set of
coefficients. The scaling question dissolves because the aggregation weight
moves into the sum; the attribution question dissolves because the entity is
still there when the reader looks. `effect_terms.py` and the `Contribution`
vocabulary go; `contributions.py` becomes the assembly onto a contributor axis
and nothing else.

The general lesson is the one the ceiling notes already state and this page
restated badly: a named expression costs nothing where it is *referenced*, so
naming a quantity the model already computes is free. Paying to compute it
twice is what is expensive.

**A consequence neither anticipated**: a named expression is evaluated against
a solution, so it can only be had at solve time. A `Result` therefore carries
every quantity the model names — a caller's own included — and the effect
breakdown is a view over those rather than a second thing stored beside them.

## What this costs

**Solver breadth.** lpspec streams to HiGHS directly, has Gurobi behind an
extra, and writes LP files; linopy's ten backends arrive only through the
optional lane, which is also the lane that gives up the streaming build. This is
the real loss and it is not mitigated, only located.

**polars** joins the dependency set, via lpspec.

**Duals and diagnostics** move the other way: `Result` exposes no duals today
and lpspec has `.dual(name)`, and `bound.diagnostics()` gives columns, rows,
nonzeros and timings that `stats.py` computes by hand.

**Data preparation does not move.** The Leontief inverse stays folded into
coefficients on the data side, where it already is — it is data prep, and the
language refuses it deliberately rather than by omission.
