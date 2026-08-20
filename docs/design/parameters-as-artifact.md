# The parameters are an artifact too

**Status:** Draft / decision record, written before the work.
**Companion:** [`lpspec-direction.md`](lpspec-direction.md), whose step 6 this
finishes and sharpens, and [`validation-layers.md`](validation-layers.md),
whose layer 3 it half closes.

fluxopt's math is a file. Its numbers are not — they are computed at every
solve and thrown away. This page argues they should have the same *status* as
the math, says what that changes, and says plainly where the symmetry stops.

## Where we are

The tidy-frames re-cut already happened at the data layer. `contract.py` went
from 117 lines to 54; NaN-as-absent is gone in favour of row absence;
`BoundType` is gone; the `ModelData` containers are polars frames; and
`ModelData.save` / `load` is a directory of parquet, one file per frame.

What did not happen is the half this page is about. The artifact we store is
**fluxopt's own schema** — `flows.sizes`, `flows.envelope`, `sizing.bounds`,
`effects` — while the thing lpspec binds is **the program's 93 parameter
tables**. `math/sources.py` stands between them, 1089 lines, running at every
solve and producing something no caller can see or keep.

So the two halves of a solve have unequal standing:

| | math | parameters |
|---|---|---|
| artifact | `math/program.yaml`, shipped package data | none |
| readable | `system.math()` → `lpspec.Model` | nothing |
| persisted | in the repo | only indirectly, as the `ModelData` parquet tree |
| handed back | `optimize(math=...)` | `optimize(parameters=...)` — but you cannot read what fluxopt bound |

That last row is the tell. A caller can read, edit, diff and typeset the
equations, and then supply data for a constraint of their own through a
channel whose existing contents are invisible to them. `solve` checks exactly
one thing about the supply: that it does not collide with a name the program
already binds.

## The decision

**`system.parameters(profiles=...)` returns the bound parameter set, it
round-trips to a parquet directory, and `optimize` takes it back.**
`(program.yaml, parameters/)` is then the whole problem: hashable, diffable,
solvable without fluxopt.

The larger consequence, and the reason this is worth doing rather than merely
nice, is that it forces a question the current shape lets us avoid: **do we
keep two schemas, or is the stored middle layer the program's parameter set?**

```
today
  elements ──▶ ModelData ──▶ build_sources ──▶ {93 param tables}  ──▶ lpspec
               parquet dir     1089 lines        ephemeral
               fluxopt schema  at solve time

proposed
  elements ──▶ build ──▶ parameters/ ──▶ lpspec
                         parquet dir, program schema, stored
                              │
                         labels/  ──▶ results · stats · plot
```

The middle layer stops being a schema of our own that gets translated, and
becomes the binding itself, persisted.

## What it changes

### One schema instead of two

`contract.py`'s vocabulary, the container classes and `program.yaml`'s 93
declarations state overlapping facts, and `sources.py` is the translation
between them. The renaming half of that collapses.

**Only the renaming half.** A large part of `sources.py` is computation that
has to happen somewhere whatever the container is: the Leontief fold,
`_size_upper`'s big-M, composing an absolute envelope from `size × relative`,
mapping time labels to the ordinals the program indexes by, and cross-joining
`PERIOD_PARAMS` onto every period. Under this shape that code *is* the build
rather than a second pass after one — which is the saving. It does not
disappear, and a plan that counts it as deleted is counting wrong.

### Validation moves to the layer that can answer

[`validation-layers.md`](validation-layers.md) keeps layer 3 partly alive for
reload: a parquet tree that was hand-edited, or written by an older version,
never passed through layers 1 and 2, so each container re-checks what its
element already refused. That page already anticipates the reload half closing
if a reloaded file were rebuilt through `ModelData.build`.

Storing the parameters gives a stronger version of the same result. A reloaded
parameter set is checked by **lpspec against the program it will be bound
to** — layer 4, which that page names as the best checker and the one fluxopt
must not reimplement. Unknown labels, a `float` column where the program says
`bool`, a constant side the parameters do not cover: all of it, with the
program in hand.

The `ProfileRef` half of layer 3 does not move and should not. A rule about
values that arrive later has to be checked later.

### Results need a labels table, and the split is clean

The program already carries topology, as `lookups:` — `carrier_of`,
`converter_of`, `charge_storage`, `discharge_storage`, `status_of`,
`pw_status_of`. That is what `Result.topology()` wants.

What it does not carry, and never will, is `carrier.unit`, `carrier.color`
and `carrier.description`. The math has no use for them, so they are not
parameters, and no amount of tidying will make them into some. So the division
is honest rather than a fudge:

> the program carries what the math needs; a labels table carries what
> presentation needs.

`results.py`, `stats.py`, `contributions.py` and `math/results.py` read the
labels table plus a handful of quantities — `dt`, `flow_hour_weight`, the
period weights — that are already bound parameters and can be read as such.

### The parameter set is lossy, in the one direction that matters

The fold and the pre-scaling dissolve *declared* coefficients into *solver*
coefficients. A stored parameter set therefore cannot answer "what did the
user write for this effect?", only "what is the solver adding up".

This is the same wall the effects work hit, and it has the same answer: name
the contribution as an expression and let the ledger sum that very expression,
rather than storing raw coefficients beside the folded ones. Attribution is a
question for the math, not for the parameter file — and a plan that tries to
answer it by keeping a second, unfolded copy of the coefficients is undoing
the fold that took the effects build from 18.5 s to 4.2 s.

## It is not a YAML file

"Store it like the math" means the same **status**, not the same **format**.
Three reasons, and each has already cost us something:

1. **Size.** 93 parameters over a horizon; the `stress` benchmark is ~2M
   variables. YAML has no random access and no column types.
2. **Types.** `_INT_DIMS`, `_BOOL_PARAMS` and `_stamp_empty_dtypes` exist
   because an empty column has no dtype and pandas guesses `float64`, which
   the engine reads as a numeric label space and refuses. Parquet carries
   dtypes. YAML would reintroduce that whole class of bug at the file
   boundary.
3. **Diff.** YAML earns its keep by line-diffing. A million-row table does not
   line-diff usefully; its meaningful diff is a digest.

The status that *is* shared with the math is the part worth having: a named
artifact, validated at load against a declaration, round-trippable, and
passable back to `optimize()`.

And the YAML config file already exists — `FlowSystem.to_yaml()`: structure,
scalars and `ProfileRef`s, with bulk numbers referenced rather than inlined,
per that decision in
[`config-and-pydantic-direction.md`](config-and-pydantic-direction.md). Four
artifacts, four jobs:

| artifact | format | authored by | checked by |
|---|---|---|---|
| `FlowSystem` — structure, scalars, refs | YAML | the user | layers 1–2 |
| profiles — bulk series | parquet / netCDF, referenced | the user's data pipeline | layer 3, on resolve |
| `program.yaml` — the math | YAML | fluxopt, editable via `math=` | lpspec |
| `parameters/` — the bound numbers | parquet directory | **derived, never authored** | lpspec, against the program |

## Decisions taken, so they are not re-argued

**Derived, never authored.** The parameter set is persistable, not a writing
surface. Editing `effects_per_flow_hour` means editing numbers whose relation
to the declared inputs the fold has already dissolved — an edit nobody can
review, which is the objection that retired `customize`, moved to the data
side. The authoring surfaces stay the element layer for data and
`program.yaml` for math.

**Parquet, not YAML.** Above.

**A labels table is a real thing, not a leftover.** Presentation metadata has
no parameter and needs none. Naming it up front stops it being smuggled into
the program as parameters nothing reads.

**Attribution is answered by named expressions.** Not by a second copy of the
coefficients.

## What this costs

**`Result.data` is public and its layout ships.** Changing what the stored
artifact *is* breaks saved results. Step 2 below is where that break lands and
it should land once.

**Two representations during the migration.** Between steps 1 and 3 the
parameter set is derived twice — once for the stored artifact, once inside
`sources.py`. That is the cost of not doing it as one commit, and it is
cheaper than the alternative.

**Nothing is saved on the first step.** Step 1 is pure addition. The line
count only moves at step 3.

## Order of work

1. **`system.parameters(profiles=...)`**, plus `write_parquet` / `read_parquet`
   over what `build_sources` already returns. No layer change. It makes the
   `math=` / `parameters=` pair symmetric, which is the smallest statement of
   the whole idea.
2. **`Result.save` persists `parameters/` and a digest** instead of the
   fluxopt-schema `model/` tree. Provenance lands here: a result that says
   which numbers produced it.
3. **Collapse the schemas.** The build emits program-keyed tables directly;
   `ModelData` shrinks to the labels table; layer 3's reload half closes.
   Breaking, and gated by `tests/math` and `tests/math_port` on the shipping
   lane.

Step 1 is small and useful on its own, so it is worth doing even if step 3 is
never reached.

## A side effect worth recording

This closes the seam that
[bindspec](https://github.com/fluxopt/bindspec) wants and that fluxopt cannot
currently offer it. bindspec binds *lpspec parameter names against an lpspec
model*, while a `ProfileRef` names an element field — different key spaces, so
it has nothing to attach to today. Once the parameter set is an artifact keyed
by parameter name, its `expect` contracts and its deterministic manifest apply
to ours directly, as an optional check on the way out rather than a layer on
the way in.

That is a reason to prefer this direction over adopting a binding layer at the
ingest end, not a commitment to take the dependency.
