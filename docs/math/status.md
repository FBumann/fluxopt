# Status (On/Off Constraints)

## Overview

Status adds binary on/off behavior to flows. The core variables are a binary
status indicator and transition indicators (startup/shutdown). Duration tracking
variables enforce minimum and maximum consecutive up- and downtime.

## Variables

| Symbol | Code | Domain | Description |
|---|---|---|---|
| \(\sigma_{f,t}\) | `flow--on[flow, time]` | \(\{0, 1\}\) | On/off indicator |
| \(\tau^+_{f,t}\) | `flow--startup[flow, time]` | \(\{0, 1\}\) | Startup event indicator |
| \(\tau^-_{f,t}\) | `flow--shutdown[flow, time]` | \(\{0, 1\}\) | Shutdown event indicator |
| \(\mathrm{D}^{\text{up}}_{f,t}\) | `uptime[flow, time]` | \(\geq 0\) | Consecutive uptime [h] |
| \(\mathrm{D}^{\text{down}}_{f,t}\) | `downtime[flow, time]` | \(\geq 0\) | Consecutive downtime [h] |

## Semi-Continuous Flow Rates

With fixed size, the on/off indicator gates the flow rate bounds:

\[
\bar{\mathrm{P}}_f \cdot \underline{\mathrm{p}}_{f,t} \cdot \sigma_{f,t} \leq P_{f,t} \leq \bar{\mathrm{P}}_f \cdot \bar{\mathrm{p}}_{f,t} \cdot \sigma_{f,t}
\]

When \(\sigma_{f,t} = 0\): \(P_{f,t} = 0\). When \(\sigma_{f,t} = 1\):
\(P_{f,t} \in [\bar{\mathrm{P}}_f \underline{\mathrm{p}}_{f,t}, \bar{\mathrm{P}}_f \bar{\mathrm{p}}_{f,t}]\).

This gives the semi-continuous behavior \(\{0\} \cup [\underline{\mathrm{P}}_{f,t}, \bar{\mathrm{P}}_{f,t}]\).

## Switch Transitions

Startup and shutdown indicators are linked to status changes:

\[
\tau^+_{f,t} - \tau^-_{f,t} = \sigma_{f,t} - \sigma_{f,t-1} \quad \forall \, t > 0
\]

At the first timestep with known previous state \(\sigma_{f,0}^{\text{prev}}\):

\[
\tau^+_{f,0} - \tau^-_{f,0} = \sigma_{f,0} - \sigma_{f,0}^{\text{prev}}
\]

The previous state is derived from `Flow.prior`: on if the last prior value > 0,
off otherwise. Without `prior`, the initial transition is unconstrained.

All chains on this page (transitions and duration tracking) reset at
*episode* boundaries: in multi-period models each period is an independent
episode, no constraint links its first timestep to the previous period's
last, and \(t = 0\) conditions apply at every episode start. The Big-M
\(M\) is the longest episode duration, not the horizon length.

## Duration Tracking

Duration tracking uses a Big-M formulation to count consecutive hours in a state.

### Uptime

The uptime variable \(\mathrm{D}^{\text{up}}_{f,t}\) tracks consecutive on-hours:

**Reset when off:**

\[
\mathrm{D}^{\text{up}}_{f,t} \leq \sigma_{f,t} \cdot M \quad \forall \, t
\]

**Forward accumulation:**

\[
\mathrm{D}^{\text{up}}_{f,t+1} \leq \mathrm{D}^{\text{up}}_{f,t} + \Delta t_{t+1} \quad \forall \, t
\]

**Backward tightening (force accumulation when on):**

\[
\mathrm{D}^{\text{up}}_{f,t+1} \geq \mathrm{D}^{\text{up}}_{f,t} + \Delta t_{t+1} + (\sigma_{f,t+1} - 1) \cdot M \quad \forall \, t
\]

where \(M\) is the longest episode duration (Big-M constant). The duration
follows the end-of-timestep convention — \(\mathrm{D}^{\text{up}}_{f,t}\)
includes timestep \(t\)'s own \(\Delta t_t\) — which makes \(M\) exactly
tight on any grid, uniform or not.

### Downtime

Downtime tracking uses the same formulation applied to the inverted state
\((1 - \sigma_{f,t})\).

### Minimum Duration

Minimum uptime is enforced at shutdown transitions — the accumulated duration
must meet the minimum before turning off:

\[
\mathrm{D}^{\text{up}}_{f,t} \geq \mathrm{D}^{\text{up,min}} \cdot (\sigma_{f,t} - \sigma_{f,t+1}) \quad \forall \, t < |\mathcal{T}|
\]

The term \((\sigma_{f,t} - \sigma_{f,t+1})\) equals 1 only at shutdown
(on → off), enforcing \(\mathrm{D}^{\text{up}}_{f,t} \geq \mathrm{D}^{\text{up,min}}\).

Minimum downtime follows the same pattern on \((1 - \sigma)\).

### Maximum Duration

Maximum duration is enforced as an upper bound on the duration variable itself.

Duration values are in **hours**. With sub-hourly timesteps (e.g., `dt=0.5`),
a `uptime_min=2` means the unit must stay on for 4 consecutive timesteps.

### Previous Duration Carryover

`Flow.prior_rates` provides the flow rates from timesteps **before** the
optimization horizon. This lets the solver know the initial on/off state
and how long the unit has been running or idle:

The prior rates determine:

1. **Initial on/off state**: last value > 0 means on, = 0 means off
2. **Previous duration**: consecutive hours in the current state at the end
   of the prior, used for duration constraint carryover

When prior provides historical state, the previous duration is computed
by counting consecutive matching timesteps at the end of the prior. At \(t=0\):

\[
\mathrm{D}^{\text{up}}_{f,0} = \sigma_{f,0} \cdot (\mathrm{D}^{\text{up,prev}} + \Delta t_0)
\]

If the previous uptime hasn't yet met the minimum, the unit is forced to stay on:

\[
\sigma_{f,0} \geq 1 \quad \text{if } 0 < \mathrm{D}^{\text{up,prev}} < \mathrm{D}^{\text{up,min}}
\]

## Effect Contributions

### Running Costs

A per-hour cost while the unit is on, independent of the flow rate:

\[
\Phi_{k,t}^{\text{running}} = \sum_{f} \mathrm{r}_{f,k,t} \cdot \sigma_{f,t} \cdot \Delta t_t
\]

where \(\mathrm{r}_{f,k,t}\) is `Status.effects_per_running_hour[k]`.

### Startup Costs

A one-time cost charged each time the unit switches from off to on:

\[
\Phi_{k,t}^{\text{startup}} = \sum_{f} \mathrm{u}_{f,k,t} \cdot \tau^+_{f,t}
\]

where \(\mathrm{u}_{f,k,t}\) is `Status.effects_per_startup[k]`.

Both feed into the [per-timestep effect equation](effects.md).

## Interaction with Sizing

When a flow has both `Status` and `Sizing`, the on/off indicator and the size
variable are decoupled via Big-M constraints:

\[
P_{f,t} \leq \sigma_{f,t} \cdot M^+ \qquad \text{(on-indicator gates flow)}
\]

\[
P_{f,t} \leq S_f \cdot \bar{\mathrm{p}}_{f,t} \qquad \text{(rate limited by size)}
\]

\[
P_{f,t} \geq (\sigma_{f,t} - 1) \cdot M^- + S_f \cdot \underline{\mathrm{p}}_{f,t} \qquad \text{(minimum when on)}
\]

where \(M^+ = \mathrm{S}^+ \cdot \bar{\mathrm{p}}_{f,t}\) and \(M^- = \mathrm{S}^+ \cdot \underline{\mathrm{p}}_{f,t}\),
using the maximum possible size \(\mathrm{S}^+\) as Big-M.

An additional constraint prevents the unit from being "on" with zero size:

\[
\sigma_{f,t} \leq S_f \quad \forall \, t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\sigma_{f,t}\) | On/off binary | `flow--on[flow, time]` |
| \(\tau^+_{f,t}\) | Startup indicator | `flow--startup[flow, time]` |
| \(\tau^-_{f,t}\) | Shutdown indicator | `flow--shutdown[flow, time]` |
| \(\mathrm{D}^{\text{up}}_{f,t}\) | Consecutive uptime | `uptime[flow, time]` |
| \(\mathrm{D}^{\text{down}}_{f,t}\) | Consecutive downtime | `downtime[flow, time]` |
| \(\mathrm{D}^{\text{up,min}}\) | Minimum uptime | [`Status.uptime_min`](../api/fluxopt/elements.md#fluxopt.elements.Status.uptime_min) |
| \(\mathrm{D}^{\text{up,max}}\) | Maximum uptime | [`Status.uptime_max`](../api/fluxopt/elements.md#fluxopt.elements.Status.uptime_max) |
| \(\mathrm{D}^{\text{down,min}}\) | Minimum downtime | [`Status.downtime_min`](../api/fluxopt/elements.md#fluxopt.elements.Status.downtime_min) |
| \(\mathrm{D}^{\text{down,max}}\) | Maximum downtime | [`Status.downtime_max`](../api/fluxopt/elements.md#fluxopt.elements.Status.downtime_max) |
| \(\mathrm{r}_{f,k,t}\) | Running cost coefficient | [`Status.effects_per_running_hour`](../api/fluxopt/elements.md#fluxopt.elements.Status.effects_per_running_hour) |
| \(\mathrm{u}_{f,k,t}\) | Startup cost coefficient | [`Status.effects_per_startup`](../api/fluxopt/elements.md#fluxopt.elements.Status.effects_per_startup) |
| \(M\) | Big-M (longest episode duration) | computed |

See [Notation](notation.md) for the full symbol table.
