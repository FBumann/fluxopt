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
| \(E_{s,t}\) | `storage--level[storage, time]` | \(\geq 0\) | MWh | Stored energy |
| \(\Phi_{k,t}\) | `effect_per_timestep[effect, time]` | \(\mathbb{R}\) | varies | Per-timestep effect |
| \(\Phi_k\) | `effect_total[effect]` | \(\mathbb{R}\) | varies | Total effect over horizon |
| \(S_f\) | `flow_size[flow]` | \(\geq 0\) | MW | Invested flow capacity |
| \(y_f\) | `flow_size_indicator[flow]` | \(\{0, 1\}\) | — | Binary invest indicator (flow) |
| \(S_s\) | `storage_capacity[storage]` | \(\geq 0\) | MWh | Invested storage capacity |
| \(y_s\) | `storage_size_indicator[storage]` | \(\{0, 1\}\) | — | Binary invest indicator (storage) |
| \(\sigma_{f,t}\) | `flow_on[flow, time]` | \(\{0, 1\}\) | — | On/off indicator |
| \(\tau^+_{f,t}\) | `flow_startup[flow, time]` | \(\{0, 1\}\) | — | Startup event indicator |
| \(\tau^-_{f,t}\) | `flow_shutdown[flow, time]` | \(\{0, 1\}\) | — | Shutdown event indicator |
| \(D^{\text{up}}_{f,t}\) | `uptime[flow, time]` | \(\geq 0\) | h | Consecutive uptime |
| \(D^{\text{down}}_{f,t}\) | `downtime[flow, time]` | \(\geq 0\) | h | Consecutive downtime |

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
| \(\underline{e}_s\) | `Storage.relative_minimum_level` | \([0, 1]\) | — | Relative min SOC |
| \(\bar{e}_s\) | `Storage.relative_maximum_level` | \([0, 1]\) | — | Relative max SOC |
| \(a_{f}\) | `Converter.conversion_factors` | \(\mathbb{R}\) | — | Conversion coefficient |
| \(\alpha_{k,j}\) | `Effect.contribution_from` | \(\mathbb{R}\) | varies | Cross-effect factor (scalar) |
| \(\alpha_{k,j,t}\) | `Effect.contribution_from_per_hour` | \(\mathbb{R}\) | varies | Cross-effect factor (time-varying) |
| \(S^-\) | `Sizing.min_size` | \(\geq 0\) | MW/MWh | Minimum invested size |
| \(S^+\) | `Sizing.max_size` | \(\geq 0\) | MW/MWh | Maximum invested size |
| \(\gamma_{f,k}\) | `Sizing.effects_per_size` | \(\mathbb{R}\) | varies | Per-size investment cost |
| \(\phi_{f,k}\) | `Sizing.effects_fixed` | \(\mathbb{R}\) | varies | Fixed investment cost |
| \(D^{\text{up,min}}\) | `Status.min_uptime` | \(\geq 0\) | h | Minimum consecutive uptime |
| \(D^{\text{up,max}}\) | `Status.max_uptime` | \(\geq 0\) | h | Maximum consecutive uptime |
| \(D^{\text{down,min}}\) | `Status.min_downtime` | \(\geq 0\) | h | Minimum consecutive downtime |
| \(D^{\text{down,max}}\) | `Status.max_downtime` | \(\geq 0\) | h | Maximum consecutive downtime |
| \(r_{f,k,t}\) | `Status.effects_per_running_hour` | \(\mathbb{R}\) | varies | Running cost coefficient |
| \(u_{f,k,t}\) | `Status.effects_per_startup` | \(\mathbb{R}\) | varies | Startup cost coefficient |
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
