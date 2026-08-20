# Where a lookup may be plural

**Status:** Draft / language note, aimed upstream at
[lpspec](https://github.com/fluxopt/lpspec).
**Companion:** [`lpspec-direction.md`](lpspec-direction.md), which settles that
topology is data and every relation is a lookup rather than an adjacency
matrix.

A lookup maps one dimension's members onto another's. This page asks, position
by position, which parts of that may be plural — and finds that the answer
differs at every one, which is itself the argument for keeping them separate
keys.

Math throughout uses `$…$` / `$$…$$` so it renders both in the docs site and on
GitHub.

## What a lookup is today

`lookups: {over: <dim>, into: <dim>}` — a named single-valued map out of a
dimension, checked for containment when data is bound. Our program declares
six. `sum(x, by=L)` lands terms on the target dimension; `at(x, by=L)` reads a
target-indexed value back onto the source rows; `shift(x, by=L)` partitions the
axis it walks. The verbs are the directions of one edge.

`lpspec.to_latex` renders the declaration as an arrow between index sets:

$$
\mathrm{carrier\_of} : \mathcal{L} \to \mathcal{C}
$$

and the use as a restriction on the summation index. This is the real carrier
balance, verbatim from `to_latex(PROGRAM)`:

$$
\sum_{l \in \mathcal{L} \,:\, \mathrm{carrier\_of}(l) = c}
    \mathit{rate}_{l,t,p} \cdot \mathit{carrier}^{\mathrm{sign}}_{l} = 0
\qquad \forall\, c \in \mathcal{C},\ t \in \mathcal{T},\ p \in \mathcal{P}
$$

The typesetting confirms why a lookup is directed: it prints as a *function*,
and `over` is the side the function is total on. The direction is not an
orientation a caller could supply — it is which side is single-valued, which is
what makes $\mathrm{at}$ well defined and what lets the map live as a column on
the source dimension's index.

## The four positions

| position | plural? | why |
|---|---|---|
| `over` — the source being contracted | **no** | contraction *captures* the index; a pair index escapes the $\forall$, and it means nothing for the verbs that contract nothing |
| `per` — the source being conditioned on | **yes, freely** | join keys, and joins compose; each adds one free variable to the restriction |
| `into` — the target | **no** | one arrow is one function; two targets are two lookups |
| `by=` — the *call* | **yes — shipped** | grouping by several maps at once is a fact about one aggregation, not about any map |

The rest of this page is that table, argued.

## `over` may not be plural

Bidding zones are reconfigured between investment periods: the balance is still
one equation per zone, timestep and period, but *which* generators sit in a
zone depends on the period. The map wanted is

$$
\mathrm{zone\_of} : \mathcal{L} \times \mathcal{P} \to \mathcal{Z}
$$

and the question is what an operator does with the extra source axis. Spelled
as a composite source, `sum(by=)` contracts everything the lookup is `over`, so
both axes go:

$$
\sum_{(l,p) \,\in\, \mathcal{L} \times \mathcal{P} \,:\, \mathrm{zone\_of}(l,p) = z}
    \mathit{rate}_{l,t,p} \cdot \mathit{sign}_{l} = 0
\qquad \forall\, z \in \mathcal{Z},\ t \in \mathcal{T}
$$

The summation index is now a *pair*, so $p$ is bound by the $\sum$ and cannot
also appear in the quantifier. That pools every period's membership into one
sum indexed $(z, t)$ — not the constraint, and refused rather than silently
wrong, since `foreach: [zone, time, period]` no longer matches the expression's
dims.

**It does not type-check for the other verbs.** `at(x, by=L)` contracts
nothing; it reads a value through the map onto the source rows. Under a
composite source there is no $\sum$ to bind the pair, so the lookup application
carries a free variable in a subscript. The same objection holds for
`shift(…, by=L)`. A composite `over:` would be a form usable by one of the
three operators that accept `by=`.

**And it is recoverable anyway** — see below.

## `per` may be plural, freely

The same map, declared as a *family* of functions indexed by period. `over` is
contracted; `per` is joined on and kept:

```yaml
lookups:
  zone_of: {over: flow, into: zone, per: [period]}
```

$$
\sum_{l \in \mathcal{L} \,:\, \mathrm{zone\_of}(l, p) = z}
    \mathit{rate}_{l,t,p} \cdot \mathit{sign}_{l} = 0
\qquad \forall\, z \in \mathcal{Z},\ t \in \mathcal{T},\ p \in \mathcal{P}
$$

$p$ is free inside the restriction and bound by the $\forall$. Ordinary
set-builder notation with nothing invented, and it is the constraint that was
wanted.

### The test

> **Is $p$ bound by the $\sum$ or by the $\forall$?**

That single question is the whole difference, and the notation makes it visible
where the YAML does not. It also names the two kinds of source axis a keyed map
has — the key is `over ∪ per`, and composite `over:` is the corner where the
whole key is contracted.

### Arity fails in opposite directions

Widening `per` changes nothing structural:

$$
\sum_{l \in \mathcal{L} \,:\, \mathrm{zone\_of}(l, p, s) = z}
    \mathit{rate}_{l,t,p,s} \cdot \mathit{sign}_{l} = 0
\qquad \forall\, z \in \mathcal{Z},\ t \in \mathcal{T},\ p \in \mathcal{P},\ s \in \mathcal{S}
$$

A set-builder restriction does not care how many free variables it carries, and
`at` composes the same way: $\mathit{zone\_price}_{\,\mathrm{zone\_of}(l,p,s),\,t}$.

**That is the cleanest argument for the split.** Arity breaks `over` because
contraction captures indices; arity is free for `per` because conditioning only
adds a column to the join. Two source-axis kinds that fail in opposite
directions under widening are two different things wearing one name.

### Why `per` is the general one

Composite behaviour is a follow-up call away from `per` —
`sum(sum(x, by=zone_of), over=period)` — while `per` behaviour is unreachable
from composite, because $p$ is destroyed inside the operator with no point at
which it exists to be kept. One direction is a second call; the other is
information gone.

`per` also composes with all three verbs unchanged, because it is a property of
the map's **key** rather than a new role for the map — and the roles (group
target, read-through, axis partition) are orthogonal to how the map is
addressed.

**The distinction already exists in the operator vocabulary.**
`sum_back(x, over=<dim>, within=<n|parameter>)` walks one axis while something
else partitions it, described upstream as "the same lookup in a different
position: it says which rows are neighbours, not which group a term lands in."
`per:` moves that same contracted-versus-conditioning split onto the
declaration, which is where it belongs when it is a property of the map rather
than of one call.

### Two rules, at any arity

- **`per` ⊆ dims(expression).** The operand must carry every conditioning axis,
  or the join has no key column. The same check at arity one or five.
- **The key size is the product.** With one `per` dim the map is still
  comfortably an index column; with three it is the size of an ordinary
  parameter over those dims, and the "a lookup lives as a column on the source
  dimension's index" story has dissolved. Which is evidence for the discriminator
  being `into:` — its values are labels of a dimension, so it can be traversed —
  rather than its storage or its arity.

The genuinely open question is not arity. It is what should happen when a `per`
dim is absent from the operand but present in the constraint's `foreach`:
refuse, or broadcast the result into it. Refusing fits the language's
strictness; broadcasting is what a caller would expect. Widening `per` makes
that surface more often, but it exists at arity one too.

## `into` may not be plural — `by=` already is

`into: [zone, region]` buys nothing. `sum(by=)` lands terms on one dimension, so
a two-target lookup would need the call to pick which anyway, and the storage is
two index columns either way. Two lookups sharing a key is the honest spelling,
and the type says so:

```python
over: str
into: str | None = None
```

One arrow, one function.

The real case behind the question is not solved by two declarations: aggregating
by **(zone, technology)**, where both are attributes of the generator. After
`sum(rate, by=zone_of)` the source dim is gone, so `tech_of` is unreachable and
nesting cannot help. What is needed is grouping by both at once — a conjunction
of restrictions:

$$
\sum_{l \in \mathcal{L} \,:\, \mathrm{zone\_of}(l) = z \,\wedge\, \mathrm{tech\_of}(l) = k}
    \mathit{rate}_{l,t,p}
\qquad \forall\, z \in \mathcal{Z},\ k \in \mathcal{K},\ t \in \mathcal{T},\ p \in \mathcal{P}
$$

**This shipped** — lpspec [#704](https://github.com/fluxopt/lpspec/issues/704),
closed. `sum(array, by=[lookup, …])` lands onto every dim the lookups map into,
and `at` takes the same list, the pullback binding each source row at its own
`(zone, tech)` value. Both lanes agree.

Note it needs no `per:` at all — both lookups are `over: generator` with no
conditioning axis — which is why this half shipped first and separately.

### The two rules on a list

- **Shared `over`.** Every lookup in the list must be over the same dimension —
  only the rows actually being collapsed can be grouped.
- **Distinct `into`.** Two lookups in one list may not target the same
  dimension, or the result dims collide: the result is
  $(\mathrm{dims}(x) \setminus \{over\}) \cup \{\mathrm{into}(l), \mathrm{into}(m)\}$,
  and a union cannot hold one dim twice.

The first is derivable from what grouping means; the second is a consequence of
the result-dim algebra and is easy to miss. Both are checked at load.

### A misreadable docstring

`Builtin.lookup_kwargs` is documented as "`lookup_kwargs` name a lookup",
singular — which describes the kwarg's *kind*, not its arity. The grammar
resolves a name **list** in that position. It reads like a restriction and is
not one.

### Not at our pin

The list form is on lpspec `main`; `pyproject.toml` pins
`v0.0.1-alpha.213`, where it does not parse:

```text
by=zone_of                ok
by=[zone_of, tech_of]     SchemaError: Failed to parse expression … Expected end of text, found '('
```

So using it in `program.yaml` waits on a pin bump — which, per the pin's own
comment, is a deliberate change with the suite behind it.

## The rule that falls out

> The **declaration** stays a single-valued function between two dimensions.
> All composition happens in the **operators**.

The language's own stated taste is evidence for this partition rather than
against it. The sibling kwargs were deliberately collapsed into one `by=`:

> a lookup carries its own dimensions, so the sibling kwargs that used to
> restate them (`sum`'s `over=` beside `group_by=`, `at`'s `onto=`) are gone —
> what the two-keyword spelling once said, the name's *kind* now says, checked
> at load.

That principle is *facts about the map live on the map*, and `per:` obeys it —
the lookup still carries all its own dimensions, now including the conditioning
ones, which is why `by=`'s grammar needs no change to accept one. Which maps you
group by simultaneously is a fact about one aggregation rather than about any
map, so it varies per call and belongs on the call.

The two moves are not in tension: `over=`/`onto=` went **onto** the declaration
and `by=` gained a **list**, in the same design pass. They are the two halves of
one rule.

## Expressible without either

Neither is an expressiveness gap — which is why the ask is about legibility
rather than reach. A weighted many-to-many relation is an ordinary sparse
parameter, and a 0/1 membership table is a map:

```yaml
parameters:
  in_zone: {dims: [flow, zone, period]}       # 1.0 where the flow is in that zone
  is_tech: {dims: [flow, technology]}

constraints:
  zone_balance:
    foreach: [zone, time, period]
    expression: "sum(rate * carrier_sign * in_zone, over=flow) == 0"
```

$$
\sum_{l \in \mathcal{L}} \mathit{rate}_{l,t,p} \cdot \mathit{sign}_{l}
    \cdot \mathit{in\_zone}_{l,z,p} = 0
\qquad \forall\, z \in \mathcal{Z},\ t \in \mathcal{T},\ p \in \mathcal{P}
$$

and two-way grouping is the same trick twice —
`sum(rate * in_zone * is_tech, over=flow)` over
`foreach: [zone, technology, time, period]`. That is what to write until the
lpspec pin carries `by=[…]`.

Sparsity carries the edges, so `in_zone` holds exactly one row per
$(\text{flow}, \text{period})$ — **the same row count a conditioned lookup
would store**. And `sum(by=)` lowers to a join and a group-by anyway, which is
what a product against a sparse 0/1 parameter already is.

Declare them `float` with `1.0` values rather than `bool`: the program uses
boolean parameters only in `where:` clauses, never in arithmetic, so coercion in
a product is unverified.

So what a declared map buys over a membership parameter is the containment
check, the function arrow in the typeset output, and a declaration that says the
map is a map. Not expressiveness — legibility, and one check that would
otherwise go unwritten.

## The ask, which is one item

**`per:` on the declaration** — lpspec
[#1163](https://github.com/fluxopt/lpspec/issues/1163), filed with the
bidding-zone probe. Generalises to any arity for free.

The issue already assumes the list form exists: a `per:` lookup inside one
should require every element to agree on `per:`, rhyming with the shared-`over:`
rule #704 enforces. So the two halves compose without either needing to know
about the other.

## What this means for fluxopt

The same limitation exists one layer up, and is ours rather than lpspec's.
Multi-node folds the node into the carrier label (`carrier:node`), and
`carrier_of` is `over: flow` with one value — so a flow that moves between zones
across periods would need *different carrier labels per period*, which the
element layer cannot state.

The membership-parameter form above is how it would be written, and it would sit
beside `carrier_balance` rather than replacing it. That is worth an issue
independently of whether the language ever grows `per:`.
