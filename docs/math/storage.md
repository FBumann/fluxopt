# Storage Dynamics

## Charge Balance

The stored energy evolves over time according to charging, discharging, and
self-discharge losses:

\[
E_{s,t+1} = E_{s,t} \left(1 - \delta_s \, \Delta t_t \right) + P^{\text{c}}_{s,t} \, \eta^{\text{c}}_s \, \Delta t_t - \frac{P^{\text{d}}_{s,t}}{\eta^{\text{d}}_s} \, \Delta t_t
\]

where:

- \(E_{s,t}\) — stored energy at the start of timestep \(t\)
- \(P^{\text{c}}_{s,t}\) — charging flow rate (energy entering the storage)
- \(P^{\text{d}}_{s,t}\) — discharging flow rate (energy leaving the storage)
- \(\eta^{\text{c}}_s\) — charging efficiency (losses during charging)
- \(\eta^{\text{d}}_s\) — discharging efficiency (losses during discharging)
- \(\delta_s\) — self-discharge rate per hour
- \(\Delta t_t\) — timestep duration in hours

The charge state has \(|\mathcal{T}| + 1\) values (one before each timestep plus one
after the last timestep).

## Charge State Bounds

The charge state is bounded by relative SOC limits scaled by the storage capacity:

\[
\bar{E}_s \cdot \underline{e}_s \leq E_{s,t} \leq \bar{E}_s \cdot \bar{e}_s \quad \forall \, s, t
\]

## Initial & Cyclic Conditions

**Fixed initial state:**

\[
E_{s,t_0} = E_0
\]

where \(E_0\) is `Storage.prior_level` (absolute MWh). If `prior_level` is `None`,
the initial level is unconstrained (the optimizer chooses).

**Cyclic condition** (when `Storage.cyclic = True`):

\[
E_{s,t_{\text{end}}} = E_{s,t_0}
\]

This ensures the storage ends at the same level it started.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(E_{s,t}\) | Stored energy variable | `storage--level[storage, time]` |
| \(P^{\text{c}}_{s,t}\) | Charging flow rate | `flow_rate[charge_flow, time]` |
| \(P^{\text{d}}_{s,t}\) | Discharging flow rate | `flow_rate[discharge_flow, time]` |
| \(\bar{E}_s\) | Storage capacity | `Storage.capacity` |
| \(\eta^{\text{c}}_s\) | Charging efficiency | `Storage.eta_charge` |
| \(\eta^{\text{d}}_s\) | Discharging efficiency | `Storage.eta_discharge` |
| \(\delta_s\) | Self-discharge rate | `Storage.relative_loss_per_hour` |
| \(\underline{e}_s\) | Relative min SOC | `Storage.relative_minimum_level` |
| \(\bar{e}_s\) | Relative max SOC | `Storage.relative_maximum_level` |
| \(\Delta t_t\) | Timestep duration | dt |

See [Notation](notation.md) for the full symbol table.

## Example

A battery with \(\bar{E} = 10\) MWh, \(\eta^{\text{c}} = 0.95\),
\(\eta^{\text{d}} = 0.95\), \(\delta = 0.001\)/h, \(\Delta t = 1\) h:

Starting at \(E_0 = 5\) MWh, charging at \(P^{\text{c}} = 2\) MW:

\[
E_1 = 5 \times (1 - 0.001 \times 1) + 2 \times 0.95 \times 1 = 4.995 + 1.9 = 6.895 \; \text{MWh}
\]
