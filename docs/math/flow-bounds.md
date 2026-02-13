# Flow Bounds & Profiles

## Sized Flows

When a flow has a nominal capacity \(\bar{P}_f\) (`Flow.size`), the flow rate is
bounded by relative minimum and maximum profiles:

\[
\bar{P}_f \cdot \underline{p}_{f,t} \leq P_{f,t} \leq \bar{P}_f \cdot \bar{p}_{f,t} \quad \forall \, f, t
\]

By default, \(\underline{p}_{f,t} = 0\) and \(\bar{p}_{f,t} = 1\), so the bounds
simplify to \(0 \leq P_{f,t} \leq \bar{P}_f\).

## Unsized Flows

When no capacity is specified (\(\bar{P}_f = \infty\)), the flow is unbounded above:

\[
0 \leq P_{f,t} \quad \forall \, f, t
\]

## Fixed Profile

When `Flow.fixed_relative_profile` (\(\pi_{f,t}\)) is set, the flow rate is fixed to a
profile scaled by the capacity:

\[
P_{f,t} = \bar{P}_f \cdot \pi_{f,t} \quad \forall \, f, t
\]

This is implemented by setting both lower and upper bounds equal to the profile value.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\bar{P}_f\) | Nominal capacity | `Flow.size` |
| \(\underline{p}_{f,t}\) | Relative lower bound | `Flow.relative_minimum` |
| \(\bar{p}_{f,t}\) | Relative upper bound | `Flow.relative_maximum` |
| \(\pi_{f,t}\) | Fixed relative profile | `Flow.fixed_relative_profile` |

See [Notation](notation.md) for the full symbol table.

## Code Mapping

- **Bounds**: `model.py:32–44` — `_create_flow_variables()` creates the flow rate
  variable and applies lower/upper bound constraints from precomputed `bounds` DataFrame.
- **Fixed profile**: `model.py:47–49` — equality constraint `flow_rate == fixed_param`
  for flows where `fixed_relative_profile` is set.

## Example

A boiler with capacity \(\bar{P} = 10\) MW, minimum load
\(\underline{p} = 0.3\), maximum load \(\bar{p} = 1.0\):

\[
10 \times 0.3 \leq P_t \leq 10 \times 1.0 \quad \Rightarrow \quad 3 \leq P_t \leq 10 \; \text{MW}
\]
