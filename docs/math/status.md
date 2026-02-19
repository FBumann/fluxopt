# Status (On/Off Constraints)

## Overview

Status adds binary on/off behavior to flows. The core variables are a binary
status indicator and transition indicators (startup/shutdown). Duration tracking
variables enforce minimum and maximum consecutive up- and downtime.

## Variables

| Symbol | Code | Domain | Description |
|---|---|---|---|
| \(\sigma_{f,t}\) | `flow_on[flow, time]` | \(\{0, 1\}\) | On/off indicator |
| \(\tau^+_{f,t}\) | `flow_startup[flow, time]` | \(\{0, 1\}\) | Startup event indicator |
| \(\tau^-_{f,t}\) | `flow_shutdown[flow, time]` | \(\{0, 1\}\) | Shutdown event indicator |
| \(D^{\text{up}}_{f,t}\) | `uptime[flow, time]` | \(\geq 0\) | Consecutive uptime [h] |
| \(D^{\text{down}}_{f,t}\) | `downtime[flow, time]` | \(\geq 0\) | Consecutive downtime [h] |

## Semi-Continuous Flow Rates

With fixed size, the on/off indicator gates the flow rate bounds:

\[
\bar{P}_f \cdot \underline{p}_{f,t} \cdot \sigma_{f,t} \leq P_{f,t} \leq \bar{P}_f \cdot \bar{p}_{f,t} \cdot \sigma_{f,t}
\]

When \(\sigma_{f,t} = 0\): \(P_{f,t} = 0\). When \(\sigma_{f,t} = 1\):
\(P_{f,t} \in [\bar{P}_f \underline{p}, \bar{P}_f \bar{p}]\).

This gives the semi-continuous behavior \(\{0\} \cup [\underline{P}, \bar{P}]\).

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

## Duration Tracking

Duration tracking uses a Big-M formulation to count consecutive hours in a state.

### Uptime

The uptime variable \(D^{\text{up}}_{f,t}\) tracks consecutive on-hours:

**Reset when off:**

\[
D^{\text{up}}_{f,t} \leq \sigma_{f,t} \cdot M \quad \forall \, t
\]

**Forward accumulation:**

\[
D^{\text{up}}_{f,t+1} \leq D^{\text{up}}_{f,t} + \Delta t_t \quad \forall \, t
\]

**Backward tightening (force accumulation when on):**

\[
D^{\text{up}}_{f,t+1} \geq D^{\text{up}}_{f,t} + \Delta t_t + (\sigma_{f,t+1} - 1) \cdot M \quad \forall \, t
\]

where \(M\) is the total horizon length (Big-M constant).

### Downtime

Downtime tracking uses the same formulation applied to the inverted state
\((1 - \sigma_{f,t})\).

### Minimum Duration

Minimum uptime is enforced at shutdown transitions — the accumulated duration
must meet the minimum before turning off:

\[
D^{\text{up}}_{f,t} \geq D^{\text{up,min}} \cdot (\sigma_{f,t} - \sigma_{f,t+1}) \quad \forall \, t < |\mathcal{T}|
\]

The term \((\sigma_{f,t} - \sigma_{f,t+1})\) equals 1 only at shutdown
(on → off), enforcing \(D^{\text{up}}_{f,t} \geq D^{\text{up,min}}\).

Minimum downtime follows the same pattern on \((1 - \sigma)\).

### Maximum Duration

Maximum duration is enforced as an upper bound on the duration variable itself.

### Previous Duration Carryover

When `Flow.prior` provides historical state, the previous duration is computed
by counting consecutive matching timesteps at the end of the prior. At \(t=0\):

\[
D^{\text{up}}_{f,0} = \sigma_{f,0} \cdot (D^{\text{up,prev}} + \Delta t_0)
\]

If the previous uptime hasn't yet met the minimum, the unit is forced to stay on:

\[
\sigma_{f,0} \geq 1 \quad \text{if } 0 < D^{\text{up,prev}} < D^{\text{up,min}}
\]

## Effect Contributions

### Running Costs

Running costs are charged per hour of operation, multiplied by the on-indicator:

\[
\Phi_{k,t}^{\text{running}} = \sum_{f} r_{f,k,t} \cdot \sigma_{f,t} \cdot \Delta t_t
\]

where \(r_{f,k,t}\) is `Status.effects_per_running_hour[k]`.

### Startup Costs

Startup costs are charged per startup event (no duration scaling):

\[
\Phi_{k,t}^{\text{startup}} = \sum_{f} u_{f,k,t} \cdot \tau^+_{f,t}
\]

where \(u_{f,k,t}\) is `Status.effects_per_startup[k]`.

Both feed into the [per-timestep effect equation](effects.md).

## Interaction with Sizing

When a flow has both `Status` and `Sizing`, the on/off indicator and the size
variable are decoupled via Big-M constraints:

\[
P_{f,t} \leq \sigma_{f,t} \cdot M^+ \qquad \text{(on-indicator gates flow)}
\]

\[
P_{f,t} \leq S_f \cdot \bar{p}_{f,t} \qquad \text{(rate limited by size)}
\]

\[
P_{f,t} \geq (\sigma_{f,t} - 1) \cdot M^- + S_f \cdot \underline{p}_{f,t} \qquad \text{(minimum when on)}
\]

where \(M^+ = S^+ \cdot \bar{p}_{f,t}\) and \(M^- = S^+ \cdot \underline{p}_{f,t}\),
using the maximum possible size \(S^+\) as Big-M.

An additional constraint prevents the unit from being "on" with zero size:

\[
\sigma_{f,t} \leq S_f \quad \forall \, t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\sigma_{f,t}\) | On/off binary | `flow_on[flow, time]` |
| \(\tau^+_{f,t}\) | Startup indicator | `flow_startup[flow, time]` |
| \(\tau^-_{f,t}\) | Shutdown indicator | `flow_shutdown[flow, time]` |
| \(D^{\text{up}}_{f,t}\) | Consecutive uptime | `uptime[flow, time]` |
| \(D^{\text{down}}_{f,t}\) | Consecutive downtime | `downtime[flow, time]` |
| \(D^{\text{up,min}}\) | Minimum uptime | `Status.min_uptime` |
| \(D^{\text{up,max}}\) | Maximum uptime | `Status.max_uptime` |
| \(D^{\text{down,min}}\) | Minimum downtime | `Status.min_downtime` |
| \(D^{\text{down,max}}\) | Maximum downtime | `Status.max_downtime` |
| \(r_{f,k,t}\) | Running cost coefficient | `Status.effects_per_running_hour` |
| \(u_{f,k,t}\) | Startup cost coefficient | `Status.effects_per_startup` |
| \(M\) | Big-M (horizon length) | computed |

See [Notation](notation.md) for the full symbol table.
