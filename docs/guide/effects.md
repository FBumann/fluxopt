# Effects

An `Effect` tracks a quantity (cost, CO2, primary energy, ...) across the
optimization horizon. One effect is the objective to minimize; others can be
bounded.

See [Effects (Math)](../math/effects.md) for the formulation.

## Defining Effects

```python
from fluxopt import Effect

# Objective effect (minimized)
cost = Effect('cost', is_objective=True)

# Tracked effect with a unit
co2 = Effect('co2', unit='kg')
```

Exactly one effect must have `is_objective=True`.

## Linking Flows to Effects

Flows contribute to effects via `effects_per_flow_hour`. The value is in
effect-units per flow-hour (e.g., €/MWh):

```python
from fluxopt import Flow

# Single effect
gas = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})

# Multiple effects
gas = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04, 'co2': 0.2})
```

At each timestep, the contribution is `coefficient * flow_rate * dt`.

## Bounding Effects

### Total Bounds

Limit the total effect over the entire horizon:

```python
# CO2 budget: max 1000 kg total
co2 = Effect('co2', unit='kg', maximum_total=1000)

# Cost floor (e.g., minimum revenue)
revenue = Effect('revenue', minimum_total=500)
```

### Per-Hour Bounds

Limit the effect value at each timestep:

```python
# Max 50 kg CO2 per hour
co2 = Effect('co2', unit='kg', maximum_per_hour=50)

# Time-varying per-hour bound
co2 = Effect('co2', unit='kg', maximum_per_hour=[50, 40, 60, 50])
```

## Accessing Results

After solving, the `SolvedModel` provides several views into effect values:

```python
result = solve(...)

# Total effect values (one row per effect)
print(result.effects)

# Per-timestep effect values
print(result.effects_per_timestep)

# Objective value (shortcut for the objective effect's total)
print(result.objective_value)

# Per-source contributions: which flows contributed what to each effect
print(result.contributions)
```

### Filtering Contributions

The `contributions` DataFrame has columns: `source`, `contributor`, `effect`,
`time`, `solution`.

```python
# All flow contributions
flow_contribs = result.contributions.filter(result.contributions['source'] == 'flow')

# CO2 contributions only
co2_contribs = result.contributions.filter(result.contributions['effect'] == 'co2')
```

## Full Example

Two sources with different cost/CO2 tradeoffs, subject to an emission cap:

```python
from datetime import datetime
from fluxopt import Bus, Effect, Flow, Port, solve

timesteps = [datetime(2024, 1, 1, h) for h in range(3)]

demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
cheap_dirty = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
expensive_clean = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

result = solve(
    timesteps=timesteps,
    buses=[Bus('elec')],
    effects=[
        Effect('cost', is_objective=True),
        Effect('co2', maximum_total=100),
    ],
    ports=[
        Port('cheap', imports=[cheap_dirty]),
        Port('clean', imports=[expensive_clean]),
        Port('demand', exports=[demand]),
    ],
)

print(f"Total cost: {result.objective_value:.2f}")
print(result.effects)
print(result.contributions)
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Effect identifier |
| `unit` | `str` | `''` | Unit label |
| `is_objective` | `bool` | `False` | Whether this effect is minimized |
| `maximum_total` | `float \| None` | `None` | Upper bound on total |
| `minimum_total` | `float \| None` | `None` | Lower bound on total |
| `maximum_per_hour` | `TimeSeries \| None` | `None` | Upper bound per timestep |
| `minimum_per_hour` | `TimeSeries \| None` | `None` | Lower bound per timestep |
