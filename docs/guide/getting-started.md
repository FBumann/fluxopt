# Getting Started

This walkthrough builds a simple heat system end to end: define components,
solve, and inspect results.

## The System

A gas boiler supplies heat to meet a demand profile. We minimize fuel cost.

```
gas bus ──▶ [boiler η=0.9] ──▶ heat bus ──▶ demand
   ▲
   │
 grid (gas source, 0.04 €/kWh)
```

## Step by Step

### 1. Imports and Timesteps

```python
from datetime import datetime
from fluxopt import Bus, Converter, Effect, Flow, Port, solve

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
```

Timesteps can be `datetime` objects or plain integers. The duration `dt` is
inferred from consecutive timestamps (here 1 h each).

### 2. Define Buses

Buses are energy carriers — nodes where flows must balance.

```python
buses = [Bus('gas'), Bus('heat')]
```

### 3. Define Effects

Effects track quantities across the horizon. Mark one as the objective.

```python
effects = [Effect('cost', is_objective=True)]
```

### 4. Define Flows

Flows carry energy on a bus. Each flow has a `size` (nominal capacity) and
optional parameters like `fixed_relative_profile` or `effects_per_flow_hour`.

```python
# Gas source: up to 500 MW, costs 0.04 €/kWh
gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})

# Boiler fuel input and heat output
fuel = Flow(bus='gas', size=300)
heat = Flow(bus='heat', size=200)

# Heat demand: 100 MW capacity, profile sets actual demand per timestep
demand = Flow(bus='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
```

### 5. Define Ports and Converters

**Ports** connect flows to the outside world (sources and sinks).
**Converters** couple input and output flows with conversion equations.

```python
ports = [
    Port('grid', imports=[gas_source]),
    Port('demand', exports=[demand]),
]

converters = [
    Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat),
]
```

### 6. Solve

```python
result = solve(
    timesteps=timesteps,
    buses=buses,
    effects=effects,
    ports=ports,
    converters=converters,
)
```

### 7. Inspect Results

```python
# Objective value (total cost)
print(result.objective_value)

# Flow rates for a specific flow
print(result.flow_rate('boiler(gas)'))

# All flow rates
print(result.flow_rates)

# Effect totals
print(result.effects)

# Per-timestep effects
print(result.effects_per_timestep)

# Per-source contributions (which flows contributed to which effects)
print(result.contributions)
```

Flow ids are qualified as `{component}({bus_or_id})` — e.g., `boiler(gas)`,
`grid(gas)`, `demand(heat)`.

## Next Steps

- [Flows](flows.md) — sizing, bounds, profiles, effect coefficients
- [Converters](converters.md) — boiler, heat pump, CHP, custom conversion
- [Storage](storage.md) — batteries, thermal storage
- [Effects](effects.md) — multi-effect tracking, bounds, contributions
