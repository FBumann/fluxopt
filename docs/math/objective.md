# Objective Function

## Formulation

The model minimizes the total value of the designated objective effect \(k^*\)
(the one with `is_objective=True`):

\[
\min \; \Phi_{k^*}
\]

The total effect combines the temporal and periodic domains:

\[
\Phi_k = \sum_{t \in \mathcal{T}} \Phi_{k,t}^{\text{temporal}} \cdot w_t + \Phi_k^{\text{periodic}}
\]

The **temporal** domain accumulates flow contributions, status costs, and
cross-effect contributions per timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{flow}} + \underbrace{\sum_{f} r_{f,k,t} \cdot \sigma_{f,t} \cdot \Delta t_t}_{\text{running}} + \underbrace{\sum_{f} u_{f,k,t} \cdot \tau^+_{f,t}}_{\text{startup}} + \underbrace{\sum_{j} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect}}
\]

The **periodic** domain accumulates investment costs and cross-effect contributions:

\[
\Phi_k^{\text{periodic}} = \underbrace{\sum_{f} \gamma_{f,k} \cdot S_f + \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \gamma_{s,k} \cdot S_s + \sum_{s} \phi_{s,k} \cdot y_s}_{\text{direct sizing costs}} + \underbrace{\sum_{j} \alpha_{k,j} \cdot \Phi_j^{\text{periodic}}}_{\text{cross-effect}}
\]

See [Sizing](sizing.md), [Status](status.md), and [Effects](effects.md) for
full formulations of each term.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(k^*\) | Objective effect | `Effect.is_objective = True` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\Phi_{k,t}^{\text{temporal}}\) | Temporal (per-timestep) effect variable | `effect_temporal[effect, time]` |
| \(\Phi_k^{\text{periodic}}\) | Periodic (investment) effect variable | `effect_periodic[effect]` |
| \(\Phi_k\) | Total effect variable | `effect_total[effect]` |

See [Notation](notation.md) for the full symbol table.

## Example

Consider a gas boiler over 3 timesteps (\(\Delta t = 1\,\text{h}\), \(w = 1\)):

| \(t\) | \(P_{\text{gas},t}\) (MW) | \(c_{\text{gas,cost}}\) (€/MWh) | \(\Phi_{\text{cost},t}\) (€) |
|---|---|---|---|
| 1 | 2.0 | 30 | \(30 \times 2.0 \times 1 = 60\) |
| 2 | 3.0 | 30 | \(30 \times 3.0 \times 1 = 90\) |
| 3 | 1.5 | 30 | \(30 \times 1.5 \times 1 = 45\) |

Total cost: \(\Phi_{\text{cost}} = 60 + 90 + 45 = 195\,\text{€}\)

The optimizer finds the \(P_{f,t}\) values that minimize \(\Phi_{k^*}\) subject to all
constraints (bus balance, flow bounds, conversion, storage dynamics).
