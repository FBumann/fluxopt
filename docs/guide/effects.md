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

## Cross-Effect Contributions

An effect can include a weighted contribution from another effect using
`contribution_from`. This is useful for carbon pricing, primary energy factors,
or any chain where one tracked quantity feeds into another.

### Scalar (investment + per-hour)

A scalar factor applies to both investment and per-timestep values:

```python
effects = [
    Effect('cost', is_objective=True, contribution_from={'co2': 50}),
    Effect('co2', unit='kg'),
]
```

Here, every kg of CO₂ adds 50 € to cost — both for operational emissions and
investment-related emissions (e.g., from `Sizing.effects_per_size`).

### Time-Varying (per-hour only)

Use `contribution_from_per_hour` for time-varying factors that override the
scalar for the per-timestep constraint:

```python
effects = [
    Effect(
        'cost',
        is_objective=True,
        contribution_from={'co2': 50},  # scalar for investment
        contribution_from_per_hour={'co2': [40, 50, 60]},  # time-varying for operations
    ),
    Effect('co2', unit='kg'),
]
```

### Transitive Chains

Contributions chain transitively. A PE → CO₂ → cost chain is modeled as:

```python
effects = [
    Effect('cost', is_objective=True, contribution_from={'co2': 50}),
    Effect('co2', unit='kg', contribution_from={'pe': 0.3}),
    Effect('pe', unit='kWh'),
]
```

### Restrictions

- **No self-references**: an effect cannot reference itself
- **No cycles**: `cost → co2 → cost` is rejected at build time

See [Effects (Math)](../math/effects.md#cross-effect-contributions) for the
formulation.

## Accessing Results

After solving, the `Result` provides several views into effect values:

```python
result = optimize(...)

# Objective value (shortcut for the objective effect's total)
print(result.objective)

# Total effect values as (effect,) DataArray
print(result.effect_totals)

# Per-timestep effect values as (effect, time) DataArray
print(result.effects_per_timestep)
```

## Full Example

Two sources with different cost/CO2 tradeoffs, subject to an emission cap:

```python
from datetime import datetime
from fluxopt import Bus, Effect, Flow, Port, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(3)]

demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
cheap_dirty = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
expensive_clean = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

result = optimize(
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

print(f"Total cost: {result.objective:.2f}")
print(result.effect_totals)
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
| `contribution_from` | `dict[str, float]` | `{}` | Scalar cross-effect factors (invest + per-hour) |
| `contribution_from_per_hour` | `dict[str, TimeSeries]` | `{}` | Time-varying cross-effect factors (per-hour only) |
