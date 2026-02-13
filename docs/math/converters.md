# Linear Conversion

## Formulation

A `LinearConverter` enforces linear coupling between its input and output flows.
Each conversion equation requires:

\[
\sum_{f} a_{f} \cdot P_{f,t} = 0 \quad \forall \, \text{converter}, \; \text{eq\_idx}, \; t \in \mathcal{T}
\]

where \(a_f\) is the conversion coefficient for flow \(f\). A converter can have
multiple equations (one per row in `conversion_factors`), allowing multi-output
devices like CHP plants.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(a_f\) | Conversion coefficient | `LinearConverter.conversion_factors` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |

See [Notation](notation.md) for the full symbol table.

## Code Mapping

- **Conversion constraint**: `model.py:61–69` — `_create_converter_constraints()`
  applies `(Param(flow_coefficients) * flow_rate).sum('flow') == 0`.

## Examples

### Boiler

A gas boiler with thermal efficiency \(\eta_{\text{th}} = 0.9\):

\[
\eta_{\text{th}} \cdot P_{\text{gas},t} - P_{\text{th},t} = 0
\]

\[
0.9 \cdot P_{\text{gas},t} = P_{\text{th},t}
\]

So 10 MW gas input produces 9 MW thermal output.

```python
LinearConverter.boiler("boiler", thermal_efficiency=0.9, fuel_flow=gas, thermal_flow=th)
# conversion_factors = [{gas.label: 0.9, th.label: -1}]
```

### Heat Pump

A heat pump with COP = 3.5:

\[
\text{COP} \cdot P_{\text{el},t} - P_{\text{th},t} = 0
\]

\[
3.5 \cdot P_{\text{el},t} = P_{\text{th},t}
\]

So 1 MW electrical input produces 3.5 MW thermal output.

```python
LinearConverter.heat_pump("hp", cop=3.5, electrical_flow=el, thermal_flow=th)
# conversion_factors = [{el.label: 3.5, th.label: -1}]
```

### CHP (Combined Heat and Power)

A CHP with \(\eta_{\text{el}} = 0.4\) and \(\eta_{\text{th}} = 0.5\) has **two**
conversion equations:

\[
\eta_{\text{el}} \cdot P_{\text{fuel},t} - P_{\text{el},t} = 0
\]

\[
\eta_{\text{th}} \cdot P_{\text{fuel},t} - P_{\text{th},t} = 0
\]

So 10 MW fuel input produces 4 MW electrical + 5 MW thermal.

```python
LinearConverter.chp("chp", eta_el=0.4, eta_th=0.5,
                     fuel_flow=fuel, electrical_flow=el, thermal_flow=th)
# conversion_factors = [
#     {fuel.label: 0.4, el.label: -1},
#     {fuel.label: 0.5, th.label: -1},
# ]
```

### Time-Varying Coefficients

Conversion coefficients can be time-varying (e.g., a heat pump with hourly COP from
weather data). Pass a list or array instead of a scalar:

```python
cop_profile = [3.2, 3.5, 3.8, 3.1]  # one value per timestep
LinearConverter.heat_pump("hp", cop=cop_profile, electrical_flow=el, thermal_flow=th)
```
