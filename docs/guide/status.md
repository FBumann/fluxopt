# Status (On/Off Behavior)

`Status` adds binary on/off behavior to a flow. This enables:

- **Semi-continuous dispatch**: flow is either off (0) or within its bounds
- **Startup costs**: one-time cost each time a unit turns on
- **Running costs**: cost per hour while the unit is on
- **Duration constraints**: minimum/maximum consecutive up- and downtime

See [Status (Math)](../math/status.md) for the formulation.

## Basic Usage

```python
from fluxopt import Flow, Status

# Semi-continuous: flow is either 0 or in [30%, 100%] of size
boiler_heat = Flow(
    bus='heat',
    size=100,
    relative_minimum=0.3,
    status=Status(),
)
```

Without `Status`, relative bounds give a continuous range `[30, 100]` MW.
With `Status`, the flow becomes semi-continuous: `{0} U [30, 100]` MW.

## Startup & Running Costs

### Startup Costs

A one-time cost charged each time the unit switches from off to on:

```python
# 50 € per startup event
Flow(
    bus='heat',
    size=100,
    status=Status(effects_per_startup={'cost': 50}),
)
```

Startup costs discourage frequent cycling. The solver may keep a unit running
through a low-demand period rather than paying repeated startup costs.

### Running Costs

A per-hour cost while the unit is on, independent of the flow rate:

```python
# 5 €/h while running (regardless of load)
Flow(
    bus='heat',
    size=100,
    status=Status(effects_per_running_hour={'cost': 5}),
)
```

Both can be time-varying:

```python
Status(
    effects_per_startup={'cost': [50, 60, 50]},
    effects_per_running_hour={'cost': [5, 8, 5]},
)
```

## Duration Constraints

### Minimum Uptime

Once turned on, the unit must stay on for at least this many hours:

```python
# Must run for at least 3 hours once started
Flow(bus='heat', size=100, status=Status(min_uptime=3))
```

### Maximum Uptime

The unit cannot run for more than this many consecutive hours:

```python
# Must shut down after 8 hours of continuous operation
Flow(bus='heat', size=100, status=Status(max_uptime=8))
```

### Minimum Downtime

Once turned off, the unit must stay off for at least this many hours:

```python
# Must stay off for at least 2 hours after shutdown
Flow(bus='heat', size=100, status=Status(min_downtime=2))
```

### Maximum Downtime

The unit cannot stay off for more than this many consecutive hours:

```python
# Must restart within 4 hours of being off
Flow(bus='heat', size=100, status=Status(max_downtime=4))
```

### Combining Durations

All duration constraints can be combined:

```python
Status(min_uptime=3, max_uptime=8, min_downtime=2)
```

Duration values are in **hours**. With sub-hourly timesteps (e.g., `dt=0.5`),
a `min_uptime=2` means the unit must stay on for 4 consecutive timesteps.

## Prior (Historical State)

`Flow.prior` provides the flow rates from timesteps **before** the optimization
horizon. This lets the solver know the initial on/off state and how long the
unit has been running or idle:

```python
# Unit was running at 80 MW in the previous timestep
Flow(bus='heat', size=100, status=Status(min_uptime=3), prior=[80])

# Unit was off in the previous 2 timesteps
Flow(bus='heat', size=100, status=Status(min_downtime=2), prior=[0, 0])

# Unit was running for the last 4 timesteps
Flow(bus='heat', size=100, status=Status(min_uptime=3), prior=[50, 60, 70, 80])
```

Without `prior`, the initial state is free (solver decides).

The prior is used to:

1. **Set initial on/off state**: last value > 0 means on, = 0 means off
2. **Compute previous uptime/downtime**: consecutive hours in the current state
   at the end of the prior, used for duration constraint carryover

## Interaction with Sizing

When a flow has both `Status` and `Sizing`, a big-M formulation decouples the
binary on/off from the continuous size variable:

```python
from fluxopt import Sizing, Status

Flow(
    bus='heat',
    size=Sizing(min_size=50, max_size=200, mandatory=True),
    relative_minimum=0.3,
    status=Status(effects_per_startup={'cost': 100}),
)
```

An additional constraint `on <= size` prevents the unit from being "on" when
its invested size is zero (relevant for optional sizing).

See [Status (Math)](../math/status.md#interaction-with-sizing) for the big-M
formulation.

## Full Example

A gas boiler with startup costs and minimum run time:

```python
from datetime import datetime
from fluxopt import Bus, Converter, Effect, Flow, Port, Status, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(6)]

demand = Flow(bus='heat', size=100, fixed_relative_profile=[0.3, 0.8, 0.2, 0.1, 0.7, 0.4])

fuel = Flow(bus='gas', size=200)
heat_out = Flow(
    bus='heat',
    size=100,
    relative_minimum=0.3,
    status=Status(
        min_uptime=2,
        min_downtime=1,
        effects_per_startup={'cost': 50},
        effects_per_running_hour={'cost': 5},
    ),
)
gas_source = Flow(bus='gas', size=500, effects_per_flow_hour={'cost': 0.04})

result = optimize(
    timesteps=timesteps,
    buses=[Bus('gas'), Bus('heat')],
    effects=[Effect('cost', is_objective=True)],
    ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand])],
    converters=[Converter.boiler('boiler', 0.9, fuel, heat_out)],
)
```

## Parameters Summary

### Status

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_uptime` | `float \| None` | `None` | Minimum consecutive on-hours |
| `max_uptime` | `float \| None` | `None` | Maximum consecutive on-hours |
| `min_downtime` | `float \| None` | `None` | Minimum consecutive off-hours |
| `max_downtime` | `float \| None` | `None` | Maximum consecutive off-hours |
| `effects_per_running_hour` | `dict[str, TimeSeries]` | `{}` | Effect cost per running hour |
| `effects_per_startup` | `dict[str, TimeSeries]` | `{}` | Effect cost per startup event |

### Flow.prior

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prior` | `list[float] \| None` | `None` | Flow rates [MW] before the horizon |
