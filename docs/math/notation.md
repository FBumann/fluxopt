# Notation

This page defines the canonical symbols used throughout the mathematical formulation.
Each symbol maps to a specific field or variable in the code.

## Sets & Indices

| Symbol | Description | Code |
|---|---|---|
| \(t \in \mathcal{T}\) | Timesteps | `time` dimension |
| \(f \in \mathcal{F}\) | Flows | `flow` dimension |
| \(b \in \mathcal{B}\) | Buses | `bus` dimension |
| \(s \in \mathcal{S}\) | Storages | `storage` dimension |
| \(k \in \mathcal{K}\) | Effects (cost, CO₂, …) | `effect` dimension |

## Variables

| Symbol | Code | Domain | Unit | Description |
|---|---|---|---|---|
| \(P_{f,t}\) | `flow_rate[flow, time]` | \(\geq 0\) | MW | Flow rate |
| \(E_{s,t}\) | `charge_state[storage, time]` | \(\geq 0\) | MWh | Stored energy |
| \(\Phi_{k,t}\) | `effect_per_timestep[effect, time]` | \(\mathbb{R}\) | varies | Per-timestep effect |
| \(\Phi_k\) | `effect_total[effect]` | \(\mathbb{R}\) | varies | Total effect over horizon |

## Parameters

| Symbol | Code | Domain | Unit | Description |
|---|---|---|---|---|
| \(\bar{P}_f\) | `Flow.size` | \(\geq 0\) or \(\infty\) | MW | Nominal capacity |
| \(\underline{p}_{f,t}\) | `Flow.relative_minimum` | \([0, 1]\) | — | Relative lower bound |
| \(\bar{p}_{f,t}\) | `Flow.relative_maximum` | \([0, 1]\) | — | Relative upper bound |
| \(\pi_{f,t}\) | `Flow.fixed_relative_profile` | \([0, 1]\) | — | Fixed profile |
| \(c_{f,k,t}\) | `Flow.effects_per_flow_hour` | \(\mathbb{R}\) | varies | Effect coefficient per flow-hour |
| \(\bar{E}_s\) | `Storage.capacity` | \(\geq 0\) | MWh | Storage capacity |
| \(\eta^{\text{c}}_s\) | `Storage.eta_charge` | \((0, 1]\) | — | Charging efficiency |
| \(\eta^{\text{d}}_s\) | `Storage.eta_discharge` | \((0, 1]\) | — | Discharging efficiency |
| \(\delta_s\) | `Storage.relative_loss_per_hour` | \(\geq 0\) | 1/h | Self-discharge rate |
| \(\underline{e}_s\) | `Storage.relative_minimum_charge_state` | \([0, 1]\) | — | Relative min SOC |
| \(\bar{e}_s\) | `Storage.relative_maximum_charge_state` | \([0, 1]\) | — | Relative max SOC |
| \(a_{f}\) | `Converter.conversion_factors` | \(\mathbb{R}\) | — | Conversion coefficient |
| \(w_t\) | weights | \(> 0\) | — | Timestep weight |
| \(\Delta t_t\) | dt | \(> 0\) | h | Timestep duration |

## Naming Conventions

| Convention | Meaning | Example |
|---|---|---|
| Uppercase Latin | Decision variables | \(P\) (power/flow rate), \(E\) (stored energy) |
| Lowercase Latin | Relative/dimensionless parameters | \(\underline{p}\) (rel. min), \(\bar{p}\) (rel. max) |
| Greek | Physical properties | \(\eta\) (efficiency), \(\delta\) (loss rate) |
| Overbar / underbar | Bounds | \(\bar{P}\) (capacity), \(\underline{P}\) (lower bound) |
| Subscripts | Indexing | \(f\) (flow), \(t\) (time), \(s\) (storage), \(b\) (bus), \(k\) (effect) |
| Superscripts | Qualification | \(\eta^{\text{c}}\) (charge), \(\eta^{\text{d}}\) (discharge) |
