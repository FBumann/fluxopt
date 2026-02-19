# Flows

A flow represents energy transfer on a bus. This page covers the full
mathematical model: sizing, bounds, fixed profiles, and effect contributions.

## Flow Rate Variable

Each flow \(f\) has a non-negative rate variable \(P_{f,t}\) at each timestep:

\[
P_{f,t} \geq 0 \quad \forall \, f \in \mathcal{F}, \; t \in \mathcal{T}
\]

## Sizing

The nominal capacity \(\bar{P}_f\) (`Flow.size`) sets the scale for all
relative parameters. When no capacity is specified (\(\bar{P}_f = \infty\)),
the flow is unbounded above.

## Bounds

### Sized Flows

When a flow has a nominal capacity \(\bar{P}_f\), the flow rate is
bounded by relative minimum and maximum profiles:

\[
\bar{P}_f \cdot \underline{p}_{f,t} \leq P_{f,t} \leq \bar{P}_f \cdot \bar{p}_{f,t} \quad \forall \, f, t
\]

By default, \(\underline{p}_{f,t} = 0\) and \(\bar{p}_{f,t} = 1\), so the bounds
simplify to \(0 \leq P_{f,t} \leq \bar{P}_f\).

### Unsized Flows

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

## Sizing (Investment)

When `Flow.size` is a `Sizing` object, the fixed capacity is replaced by a
decision variable. See [Sizing](sizing.md) for the full formulation.

## Status (On/Off)

When `Flow.status` is set, binary on/off behavior is added. The flow becomes
semi-continuous: \(\{0\} \cup [\underline{P}, \bar{P}]\). See [Status](status.md)
for the full formulation.

## Effect Contributions

Each flow can contribute to tracked effects (cost, emissions, ...). The
per-timestep contribution of flow \(f\) to effect \(k\) is:

\[
c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t
\]

where \(c_{f,k,t}\) is the effect coefficient per flow-hour
(`Flow.effects_per_flow_hour`). These contributions feed into the
[effect tracking](effects.md) equations.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\bar{P}_f\) | Nominal capacity | `Flow.size` |
| \(\underline{p}_{f,t}\) | Relative lower bound | `Flow.relative_minimum` |
| \(\bar{p}_{f,t}\) | Relative upper bound | `Flow.relative_maximum` |
| \(\pi_{f,t}\) | Fixed relative profile | `Flow.fixed_relative_profile` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(\Delta t_t\) | Timestep duration | dt |

See [Notation](notation.md) for the full symbol table.

## Examples

### Sized Flow with Minimum Load

A boiler with capacity \(\bar{P} = 10\) MW, minimum load
\(\underline{p} = 0.3\), maximum load \(\bar{p} = 1.0\):

\[
10 \times 0.3 \leq P_t \leq 10 \times 1.0 \quad \Rightarrow \quad 3 \leq P_t \leq 10 \; \text{MW}
\]

### Fixed Demand Profile

A demand of 100 MW capacity with profile \(\pi = [0.4, 0.7, 0.5, 0.6]\):

\[
P_t = 100 \cdot \pi_t \quad \Rightarrow \quad P = [40, 70, 50, 60] \; \text{MW}
\]

### Effect Contribution

A gas flow with cost coefficient \(c = 0.04\) €/MWh, rate \(P = 5\) MW,
duration \(\Delta t = 1\) h:

\[
0.04 \times 5 \times 1 = 0.2 \; \text{€}
\]
