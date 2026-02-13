# Effect Tracking & Bounding

## Overview

Effects represent quantities that are tracked across the optimization horizon (e.g.,
cost, CO₂ emissions, primary energy). One effect is designated as the objective to
minimize.

## Per-Timestep Tracking

Each effect accumulates contributions from all flows at each timestep:

\[
\Phi_{k,t} = \sum_{f \in \mathcal{F}} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t \quad \forall \, k \in \mathcal{K}, \; t \in \mathcal{T}
\]

The coefficient \(c_{f,k,t}\) specifies how much of effect \(k\) is produced per
flow-hour of flow \(f\) (e.g., €/MWh for cost, kg/MWh for emissions).

## Total Aggregation

The total effect sums per-timestep values with optional weights:

\[
\Phi_k = \sum_{t \in \mathcal{T}} \Phi_{k,t} \cdot w_t \quad \forall \, k \in \mathcal{K}
\]

Weights \(w_t\) allow scaling timesteps (e.g., a representative week scaled to a year).

## Total Bounds

Upper and lower bounds on the total effect over the entire horizon:

\[
\underline{\Phi}_k \leq \Phi_k \leq \bar{\Phi}_k
\]

This is useful for emission caps or budget constraints.

## Per-Timestep Bounds

Bounds on the effect value at each timestep:

\[
\underline{\Phi}_{k,t} \leq \Phi_{k,t} \leq \bar{\Phi}_{k,t} \quad \forall \, t \in \mathcal{T}
\]

This enforces per-hour limits (e.g., maximum hourly emissions).

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\Phi_{k,t}\) | Per-timestep effect variable | `effect_per_timestep[effect, time]` |
| \(\Phi_k\) | Total effect variable | `effect_total[effect]` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\bar{\Phi}_k\) | Maximum total | `Effect.maximum_total` |
| \(\underline{\Phi}_k\) | Minimum total | `Effect.minimum_total` |
| \(\bar{\Phi}_{k,t}\) | Maximum per hour | `Effect.maximum_per_hour` |
| \(\underline{\Phi}_{k,t}\) | Minimum per hour | `Effect.minimum_per_hour` |

See [Notation](notation.md) for the full symbol table.

## Code Mapping

- **Effect variables**: `model.py:82–85` — `effect_per_timestep` and `effect_total`
- **Per-timestep tracking**: `model.py:88–93` — `effect_per_timestep == sum_flow(coeff * flow_rate * dt)`
- **Total aggregation**: `model.py:96` — `effect_total == sum_time(effect_per_timestep * weight)`
- **Total bounds**: `model.py:98–108` — filters non-null bounds and applies `>=`/`<=`
- **Per-timestep bounds**: `model.py:111–114` — applies lower/upper bounds from
  precomputed DataFrames

## Example

A system with two effects — cost (objective) and CO₂ (capped at 1000 kg):

```python
effects = [
    Effect("cost", unit="€", is_objective=True),
    Effect("CO2", unit="kg", maximum_total=1000),
]
```

A gas flow with both effect coefficients:

```python
gas_flow = Flow("gas", bus="gas_bus", effects_per_flow_hour={"cost": 30, "CO2": 0.2})
```

At timestep \(t\) with \(P_{\text{gas},t} = 5\) MW and \(\Delta t = 1\) h:

- \(\Phi_{\text{cost},t} = 30 \times 5 \times 1 = 150\) €
- \(\Phi_{\text{CO₂},t} = 0.2 \times 5 \times 1 = 1.0\) kg
