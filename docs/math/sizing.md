# Sizing (Investment Optimization)

## Overview

Sizing introduces a capacity decision variable \(S\) that replaces the fixed
nominal capacity \(\bar{P}_f\). The solver optimizes both the capacity and the
dispatch simultaneously.

## Variables

| Symbol | Code | Domain | Description |
|---|---|---|---|
| \(S_f\) | `flow_size[flow]` | \(\geq 0\) | Invested flow capacity |
| \(y_f\) | `flow_size_indicator[flow]` | \(\{0, 1\}\) | Binary: invest yes/no (optional only) |
| \(S_s\) | `storage_capacity[storage]` | \(\geq 0\) | Invested storage capacity |
| \(y_s\) | `storage_size_indicator[storage]` | \(\{0, 1\}\) | Binary: invest yes/no (optional only) |

## Mandatory Sizing

When `mandatory=True`, the component must be built. The capacity is continuous:

\[
S^- \leq S_f \leq S^+ \quad \text{(mandatory)}
\]

where \(S^-\) = `min_size` and \(S^+\) = `max_size`.

## Optional Sizing

When `mandatory=False`, a binary indicator \(y_f\) gates the capacity:

\[
S^- \cdot y_f \leq S_f \leq S^+ \cdot y_f
\]

When \(y_f = 0\): \(S_f = 0\) (not built). When \(y_f = 1\): \(S_f \in [S^-, S^+]\).

### Binary Invest

When \(S^- = S^+\), the sizing reduces to a binary yes/no decision at exactly
that capacity.

## Flow Rate Bounds with Sizing

With sizing, the fixed capacity \(\bar{P}_f\) is replaced by the variable \(S_f\).
The relative bounds scale by the invested size:

\[
S_f \cdot \underline{p}_{f,t} \leq P_{f,t} \leq S_f \cdot \bar{p}_{f,t} \quad \forall \, t
\]

Similarly for fixed profiles:

\[
P_{f,t} = S_f \cdot \pi_{f,t} \quad \forall \, t
\]

## Storage Sizing

The same pattern applies to storage capacity. The charge state bounds become:

\[
S_s \cdot \underline{e}_s \leq E_{s,t} \leq S_s \cdot \bar{e}_s \quad \forall \, t
\]

## Investment Effects

Investment costs contribute to effect totals. For each effect \(k\):

### Per-Size

\[
\Phi_k^{\text{invest,per\_size}} = \sum_{f} \gamma_{f,k} \cdot S_f + \sum_{s} \gamma_{s,k} \cdot S_s
\]

where \(\gamma_{f,k}\) is `Sizing.effects_per_size[k]`.

### Fixed

\[
\Phi_k^{\text{invest,fixed}} = \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \phi_{s,k} \cdot y_s
\]

where \(\phi_{f,k}\) is `Sizing.effects_fixed[k]`. Only applies when
`mandatory=False` (binary indicator exists).

### Total

The direct investment contribution to effect \(k\) is:

\[
\Phi_k^{\text{invest}} = \Phi_k^{\text{invest,per\_size}} + \Phi_k^{\text{invest,fixed}}
\]

This feeds into the [effect total](effects.md) equation and can be further
weighted by [cross-effect contributions](effects.md#cross-effect-contributions).

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(S_f\) | Flow capacity variable | `flow_size[flow]` |
| \(S_s\) | Storage capacity variable | `storage_capacity[storage]` |
| \(y_f\), \(y_s\) | Binary invest indicator | `flow_size_indicator`, `storage_size_indicator` |
| \(S^-\) | Minimum size | `Sizing.min_size` |
| \(S^+\) | Maximum size | `Sizing.max_size` |
| \(\gamma_{f,k}\) | Per-size investment cost | `Sizing.effects_per_size` |
| \(\phi_{f,k}\) | Fixed investment cost | `Sizing.effects_fixed` |

See [Notation](notation.md) for the full symbol table.
