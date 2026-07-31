# Storage Dynamics

## Charge Balance

The stored energy evolves over time according to charging, discharging, and
self-discharge losses:

\[
E_{s,t+1} = E_{s,t} \left(1 - \delta_s\right)^{\Delta t_t} + P^{\text{c}}_{s,t} \, \eta^{\text{c}}_s \, \Delta t_t - \frac{P^{\text{d}}_{s,t}}{\eta^{\text{d}}_s} \, \Delta t_t
\]

where:

- \(E_{s,t}\) — stored energy at the start of timestep \(t\)
- \(P^{\text{c}}_{s,t}\) — charging flow rate (energy entering the storage)
- \(P^{\text{d}}_{s,t}\) — discharging flow rate (energy leaving the storage)
- \(\eta^{\text{c}}_s\) — charging efficiency (losses during charging)
- \(\eta^{\text{d}}_s\) — discharging efficiency (losses during discharging)
- \(\delta_s \in [0, 1]\) — self-discharge rate per hour
- \(\Delta t_t\) — timestep duration in hours

The charge state has \(|\mathcal{T}| + 1\) values (one before each timestep plus one
after the last timestep).

## Charge State Bounds

The charge state is bounded by relative SOC limits scaled by the storage capacity:

\[
\bar{\mathrm{E}}_s \cdot \underline{\mathrm{e}}_s \leq E_{s,t} \leq \bar{\mathrm{E}}_s \cdot \bar{\mathrm{e}}_s \quad \forall \, s, t
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

**Final level bounds** (`final_level_min` / `final_level_max`) — absolute
bounds [MWh] on the level at the last timestep, per period:

\[
\underline{\mathrm{E}}^{\text{end}}_s \leq E_{s,t_{\text{end}}} \leq \bar{\mathrm{E}}^{\text{end}}_s
\]

They compose with `cyclic` (the prior level is then bounded too, since
\(E_{s,t_0} = E_{s,t_{\text{end}}}\)).

## Episode Boundaries

The charge balance is a chain — it links each timestep to its predecessor,
which is only physical while timesteps are genuinely consecutive. In
multi-period models the time axis jumps between periods, so each period is
an independent *episode*: the recursion never links a period's first
timestep to the previous period's last. The initial condition
(`prior_level`) and the cyclic condition apply per episode — \(t_0\) and
\(t_{\text{end}}\) above are each episode's first and last timestep. With a
single period there is exactly one episode and the classic reading applies
unchanged.

## Simultaneous Charge & Discharge

With \(\eta^{\text{c}} \cdot \eta^{\text{d}} < 1\), charging and discharging
at once destroys energy — occasionally optimal (e.g. under must-run surplus)
but physically impossible for most devices. `prevent_simultaneous = True`
excludes it with a binary \(b_{s,t}\) per timestep:

\[
P^{\text{c}}_{s,t} \leq \mathrm{M}^{\text{c}}_s \cdot b_{s,t}
\qquad
P^{\text{d}}_{s,t} \leq \mathrm{M}^{\text{d}}_s \cdot (1 - b_{s,t})
\]

where \(\mathrm{M}\) is the static size bound of the respective flow (fixed
size, or the sizing/investment maximum). Both flows must therefore be sized.

!!! tip "Consider a soft penalty first"
    The binary turns the model into a MILP. In most models simultaneous
    cycling only pays off under negative prices or must-run surplus — a
    small penalty on the charge/discharge flows
    (`effects_per_flow_hour={'penalty': 1e-3}`) usually discourages it
    while keeping the problem linear and the real cost accounting clean
    (this is PyPSA's recommended approach; cf.
    [Parzen et al. 2023](https://doi.org/10.1016/j.isci.2022.105729) on
    unintended storage cycling). Reach for `prevent_simultaneous` when
    the exclusion must be exact.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(E_{s,t}\) | Stored energy variable | `storage--level[storage, time]` |
| \(P^{\text{c}}_{s,t}\) | Charging flow rate | `flow--rate[charge_flow, time]` |
| \(P^{\text{d}}_{s,t}\) | Discharging flow rate | `flow--rate[discharge_flow, time]` |
| \(\bar{\mathrm{E}}_s\) | Storage capacity | [`Storage.capacity`](../api/fluxopt/elements.md#fluxopt.elements.Storage.capacity) |
| \(\eta^{\text{c}}_s\) | Charging efficiency | [`Storage.eta_charge`](../api/fluxopt/elements.md#fluxopt.elements.Storage.eta_charge) |
| \(\eta^{\text{d}}_s\) | Discharging efficiency | [`Storage.eta_discharge`](../api/fluxopt/elements.md#fluxopt.elements.Storage.eta_discharge) |
| \(\delta_s\) | Self-discharge rate | [`Storage.relative_loss_per_hour`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_loss_per_hour) |
| \(\underline{\mathrm{e}}_s\) | Relative min SOC | [`Storage.relative_level_min`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_level_min) |
| \(\bar{\mathrm{e}}_s\) | Relative max SOC | [`Storage.relative_level_max`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_level_max) |
| \(\underline{\mathrm{E}}^{\text{end}}_s\) | Min final level [MWh] | [`Storage.final_level_min`](../api/fluxopt/elements.md#fluxopt.elements.Storage.final_level_min) |
| \(\bar{\mathrm{E}}^{\text{end}}_s\) | Max final level [MWh] | [`Storage.final_level_max`](../api/fluxopt/elements.md#fluxopt.elements.Storage.final_level_max) |
| \(b_{s,t}\) | Charging indicator binary | [`Storage.prevent_simultaneous`](../api/fluxopt/elements.md#fluxopt.elements.Storage.prevent_simultaneous) |
| \(\Delta t_t\) | Timestep duration | dt |

See [Notation](notation.md) for the full symbol table.

## Example

A battery with \(\bar{\mathrm{E}} = 10\) MWh, \(\eta^{\text{c}} = 0.95\),
\(\eta^{\text{d}} = 0.95\), \(\delta = 0.001\)/h, \(\Delta t = 1\) h:

Starting at \(E_0 = 5\) MWh, charging at \(P^{\text{c}} = 2\) MW:

\[
E_1 = 5 \times (1 - 0.001)^{1} + 2 \times 0.95 \times 1 = 4.995 + 1.9 = 6.895 \; \text{MWh}
\]
