# Objective Function

## Formulation

The model minimizes the total value of the designated objective effect \(k^*\)
(the one with `is_objective=True`):

\[
\min \; \Phi_{k^*}
\]

The total effect aggregates per-timestep contributions weighted by timestep weights:

\[
\Phi_k = \sum_{t \in \mathcal{T}} \Phi_{k,t} \cdot w_t
\]

Each per-timestep effect is the sum of flow contributions scaled by duration:

\[
\Phi_{k,t} = \sum_{f \in \mathcal{F}} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(k^*\) | Objective effect | `Effect.is_objective = True` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\Phi_{k,t}\) | Per-timestep effect variable | `effect_per_timestep[effect, time]` |
| \(\Phi_k\) | Total effect variable | `effect_total[effect]` |

See [Notation](notation.md) for the full symbol table.

## Code Mapping

- **Objective**: `model.py:201–206` — `_set_objective()` minimizes `effect_total` filtered to the objective effect.
- **Effect tracking**: `model.py:87–90` — `effect_per_timestep = sum_flow(coeff * flow_rate * dt)`
- **Total aggregation**: `model.py:96` — `effect_total = sum_time(effect_per_timestep * weight)`

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
