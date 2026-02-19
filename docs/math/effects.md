# Effect Tracking & Bounding

## Overview

Effects represent quantities that are tracked across the optimization horizon (e.g.,
cost, CO₂ emissions, primary energy). One effect is designated as the objective to
minimize.

Effects are split into two **domains**:

- **Temporal** (with time dimension): per-timestep flow contributions, status costs
- **Periodic** (without time dimension): sizing costs, fixed yearly costs

Both domains support cross-effect chains via `contribution_from`.

## Temporal Domain

Each effect accumulates contributions from all flows at each timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f \in \mathcal{F}} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{direct flow contributions}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect contributions}} \quad \forall \, k, t
\]

The coefficient \(c_{f,k,t}\) specifies how much of effect \(k\) is produced per
flow-hour of flow \(f\) (e.g., €/MWh for cost, kg/MWh for emissions).

The cross-effect factor \(\alpha_{k,j,t}\) can be time-varying
(`contribution_from_per_hour`) or constant (`contribution_from`).
Because \(\Phi_{k,t}^{\text{temporal}}\) is a **variable**, the solver resolves
multi-level chains (e.g., PE → CO₂ → cost) automatically.

## Periodic Domain

Sizing costs and fixed costs (not time-varying) are accumulated per effect:

\[
\Phi_k^{\text{periodic}} = \underbrace{\Phi_k^{\text{invest,direct}}}_{\text{direct sizing costs}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j} \cdot \Phi_j^{\text{periodic}}}_{\text{cross-effect contributions}} \quad \forall \, k
\]

where the direct investment term is:

\[
\Phi_k^{\text{invest,direct}} = \sum_{f} \gamma_{f,k} \cdot S_f + \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \gamma_{s,k} \cdot S_s + \sum_{s} \phi_{s,k} \cdot y_s
\]

Because \(\Phi_k^{\text{periodic}}\) is a **variable** (not an expression), the
solver resolves multi-level chains correctly: if PE has sizing costs
and CO₂ depends on PE and cost depends on CO₂, the chain propagates through the
periodic domain just as it does through the temporal domain.

## Cross-Effect Contributions

An effect can include a weighted fraction of another effect's value via
`contribution_from`. This enables patterns like carbon pricing (CO₂ → cost)
or transitive chains (PE → CO₂ → cost).

The scalar factor \(\alpha_{k,j}\) from `contribution_from` applies to **both**
domains. The time-varying factor from `contribution_from_per_hour` overrides the
temporal factor only.

### Validation

Self-references (\(\alpha_{k,k}\)) and circular dependencies
(\(k \to j \to \cdots \to k\)) are rejected at build time to prevent singular systems.

## Total Aggregation

The total effect combines both domains:

\[
\Phi_k = \sum_{t \in \mathcal{T}} \Phi_{k,t}^{\text{temporal}} \cdot w_t + \Phi_k^{\text{periodic}} \quad \forall \, k \in \mathcal{K}
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
| \(\Phi_{k,t}^{\text{temporal}}\) | Per-timestep effect variable | `effect_temporal[effect, time]` |
| \(\Phi_k^{\text{periodic}}\) | Periodic effect variable (sizing, fixed costs) | `effect_periodic[effect]` |
| \(\Phi_k\) | Total effect variable | `effect_total[effect]` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(\alpha_{k,j,t}\) | Cross-effect contribution factor (per hour) | `Effect.contribution_from_per_hour` |
| \(\alpha_{k,j}\) | Cross-effect contribution factor (scalar) | `Effect.contribution_from` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\bar{\Phi}_k\) | Maximum total | `Effect.maximum_total` |
| \(\underline{\Phi}_k\) | Minimum total | `Effect.minimum_total` |
| \(\bar{\Phi}_{k,t}\) | Maximum per hour | `Effect.maximum_per_hour` |
| \(\underline{\Phi}_{k,t}\) | Minimum per hour | `Effect.minimum_per_hour` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Direct effects

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

### Carbon pricing via `contribution_from`

CO₂ priced at 50 €/t into the cost effect:

```python
effects = [
    Effect("cost", is_objective=True, contribution_from={"co2": 50}),
    Effect("co2", unit="kg"),
]
```

With \(\alpha_{\text{cost,co2}} = 50\), the per-timestep cost becomes:

\[
\Phi_{\text{cost},t} = c_{\text{cost}} \cdot P_t \cdot \Delta t + 50 \cdot \Phi_{\text{co2},t}
\]

The CO₂ total itself is **not** affected — `contribution_from` is one-directional.
