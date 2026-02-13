# Converters

A `Converter` enforces linear coupling between input and output flows. It
models devices like boilers, heat pumps, and CHP plants.

See [Converters (Math)](../math/converters.md) for the formulation.

## Factory Methods

### Boiler

Single input (fuel), single output (heat), with thermal efficiency:

```python
from fluxopt import Converter, Flow

fuel = Flow(bus='gas', size=300)
heat = Flow(bus='heat', size=200)

boiler = Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat)
```

This creates the conversion equation: `0.9 * P_gas - P_heat = 0`,
so 10 MW gas input produces 9 MW heat.

### Heat Pump

Single input (electricity), single output (heat), with COP:

```python
el = Flow(bus='elec', size=50)
th = Flow(bus='heat', size=200)

hp = Converter.heat_pump('hp', cop=3.5, electrical_flow=el, thermal_flow=th)
```

Conversion equation: `3.5 * P_el - P_heat = 0`.

### CHP (Combined Heat and Power)

Single input (fuel), two outputs (electricity + heat). Two conversion
equations, one per output:

```python
fuel = Flow(bus='gas', size=100)
el = Flow(bus='elec', size=50)
th = Flow(bus='heat', size=60)

chp = Converter.chp('chp', eta_el=0.4, eta_th=0.5,
                     fuel_flow=fuel, electrical_flow=el, thermal_flow=th)
```

This produces two equations:

- `0.4 * P_fuel - P_el = 0`
- `0.5 * P_fuel - P_heat = 0`

So 10 MW fuel input produces 4 MW electrical + 5 MW thermal.

## Custom Conversion Factors

For devices not covered by factory methods, pass `conversion_factors` directly.
Each dict in the list is one conversion equation, mapping flows to their
coefficients:

```python
in1 = Flow(bus='a', size=100)
in2 = Flow(bus='b', size=100)
out = Flow(bus='c', size=100)

conv = Converter(
    id='custom',
    inputs=[in1, in2],
    outputs=[out],
    conversion_factors=[{in1: 0.5, in2: 0.3, out: -1}],
)
```

This enforces: `0.5 * P_a + 0.3 * P_b - P_c = 0`.

## Time-Varying Coefficients

Coefficients can vary per timestep (e.g., a heat pump with weather-dependent
COP):

```python
cop_profile = [3.2, 3.5, 3.8, 3.1]  # one value per timestep
hp = Converter.heat_pump('hp', cop=cop_profile, electrical_flow=el, thermal_flow=th)
```

## Full Example

Gas boiler serving a heat demand:

```python
from datetime import datetime
from fluxopt import Bus, Converter, Effect, Flow, Port, solve

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
demand = [40.0, 70.0, 50.0, 60.0]

gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})
fuel = Flow(bus='gas', size=300)
heat = Flow(bus='heat', size=200)
demand_flow = Flow(bus='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])

result = solve(
    timesteps=timesteps,
    buses=[Bus('gas'), Bus('heat')],
    effects=[Effect('cost', is_objective=True)],
    ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand_flow])],
    converters=[Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat)],
)

# Gas consumed = heat / efficiency
print(result.flow_rate('boiler(gas)'))
```
