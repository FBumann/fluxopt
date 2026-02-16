# Bus Balance

## Formulation

Every bus \(b\) must be balanced at every timestep — total outflow equals total inflow:

\[
\sum_{f \in \mathcal{F}_b^{\text{out}}} P_{f,t} - \sum_{f \in \mathcal{F}_b^{\text{in}}} P_{f,t} = 0 \quad \forall \, b \in \mathcal{B}, \; t \in \mathcal{T}
\]

where:

- \(\mathcal{F}_b^{\text{out}}\) — flows that produce into bus \(b\) (imports of ports and outputs of converters)
- \(\mathcal{F}_b^{\text{in}}\) — flows that consume from bus \(b\) (exports of ports and inputs of converters)

The sign convention uses coefficients: \(+1\) for flows producing into the bus and
\(-1\) for flows consuming from the bus. The constraint is then:

\[
\sum_{f \in \mathcal{F}_b} \text{coeff}_{b,f} \cdot P_{f,t} = 0 \quad \forall \, b, t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\mathcal{F}_b^{\text{out}}\) | Flows producing into bus \(b\) | `bus_coeff[f.id] = +1` (port imports, converter outputs, storage discharging) |
| \(\mathcal{F}_b^{\text{in}}\) | Flows consuming from bus \(b\) | `bus_coeff[f.id] = -1` (port exports, converter inputs, storage charging) |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |

See [Notation](notation.md) for the full symbol table.

## Example

A thermal bus with a boiler output (3 MW) and a demand input (3 MW):

\[
\underbrace{P_{\text{boiler\_th},t}}_{+1 \times 3} + \underbrace{(-1) \cdot P_{\text{demand},t}}_{-1 \times 3} = 0 \quad \checkmark
\]
