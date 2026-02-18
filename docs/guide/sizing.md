# Sizing (Investment Optimization)

`Sizing` lets the solver decide the optimal capacity of a flow or storage,
instead of fixing it upfront. This models investment decisions: how large
should a boiler, PV array, or battery be?

See [Sizing (Math)](../math/sizing.md) for the formulation.

## Basic Usage

Pass a `Sizing` object instead of a numeric `size`:

```python
from fluxopt import Flow, Sizing

# Solver picks optimal size between 50 and 200 MW
source = Flow(bus='elec', size=Sizing(min_size=50, max_size=200, mandatory=True))
```

## Mandatory vs Optional

### Mandatory (`mandatory=True`)

The component **must** be built. The solver picks a continuous size in
`[min_size, max_size]`:

```python
# Always built, size in [50, 200] MW
Flow(bus='elec', size=Sizing(min_size=50, max_size=200, mandatory=True))
```

### Optional (`mandatory=False`, default)

The solver decides **whether** to build the component. A binary indicator
variable gates the size: either 0 (not built) or in `[min_size, max_size]`:

```python
# Built at [50, 200] MW or not built at all
Flow(bus='elec', size=Sizing(min_size=50, max_size=200))
```

### Binary Invest (fixed-size yes/no)

When `min_size == max_size`, it becomes a binary invest decision — build at
exactly that size or not at all:

```python
# Either build a 100 MW unit or nothing
Flow(bus='elec', size=Sizing(min_size=100, max_size=100))
```

## Investment Effects

Sizing can contribute to tracked effects (cost, CO₂, etc.) via two mechanisms:

### Per-Size Effects

Cost proportional to the invested size (e.g., €/MW or kg_CO₂/MWh):

```python
# 500 €/MW investment cost
Flow(
    bus='elec',
    size=Sizing(min_size=50, max_size=200, mandatory=True, effects_per_size={'cost': 500}),
)
```

### Fixed Effects

One-time cost charged when the component is built. Only applies to optional
sizing (`mandatory=False`) since it's gated by the binary indicator:

```python
# 10,000 € fixed cost if built, plus 500 €/MW
Flow(
    bus='elec',
    size=Sizing(
        min_size=50, max_size=200,
        effects_per_size={'cost': 500},
        effects_fixed={'cost': 10_000},
    ),
)
```

## Storage Sizing

`Storage.capacity` also accepts `Sizing` for optimizing battery or tank size:

```python
from fluxopt import Flow, Sizing, Storage

charge = Flow(bus='elec', size=50)
discharge = Flow(bus='elec', size=50)

battery = Storage(
    'battery',
    charging=charge,
    discharging=discharge,
    capacity=Sizing(min_size=100, max_size=1000, mandatory=True, effects_per_size={'cost': 200}),
)
```

## Interaction with Other Features

### With Bounds

Relative bounds (`relative_minimum`, `relative_maximum`) are fractions of the
**optimized** size variable, not a fixed number. If the solver picks 80 MW and
`relative_minimum=0.3`, the minimum flow rate is 24 MW.

### With Status

When a flow has both `Sizing` and `Status`, a big-M formulation decouples the
binary on/off from the continuous size. See [Status](status.md) for details.

## Full Example

Two competing sources — the solver decides whether to invest in a cheaper source:

```python
from datetime import datetime
from fluxopt import Bus, Effect, Flow, Port, Sizing, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]

demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

# Always-available expensive source
grid = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': 0.10})

# Optional cheap source with investment cost
solar = Flow(
    bus='elec',
    size=Sizing(min_size=50, max_size=200, effects_per_size={'cost': 20}),
    effects_per_flow_hour={'cost': 0.01},
    fixed_relative_profile=[0.0, 0.8, 0.8, 0.0],  # only available midday
)

result = optimize(
    timesteps=timesteps,
    buses=[Bus('elec')],
    effects=[Effect('cost', is_objective=True)],
    ports=[
        Port('grid', imports=[grid]),
        Port('solar', imports=[solar]),
        Port('demand', exports=[demand]),
    ],
)

print(f"Objective: {result.objective:.2f}")
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_size` | `float` | required | Minimum size if invested |
| `max_size` | `float` | required | Maximum size |
| `mandatory` | `bool` | `False` | If True, must be built (no binary indicator) |
| `effects_per_size` | `dict[str, float]` | `{}` | Effect cost per unit size (e.g., €/MW) |
| `effects_fixed` | `dict[str, float]` | `{}` | Fixed effect cost if built (optional only) |
