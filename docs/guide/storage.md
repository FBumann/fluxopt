# Storage

A `Storage` models energy storage with charge/discharge flows, capacity,
efficiency, and self-discharge.

See [Storage (Math)](../math/storage.md) for the formulation.

## Basic Construction

A storage needs two flows (charging and discharging) on the same bus:

```python
from fluxopt import Flow, Storage

charge = Flow(bus='elec', size=50)     # max charge rate 50 MW
discharge = Flow(bus='elec', size=50)  # max discharge rate 50 MW

battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)
```

Flow ids are auto-qualified: `battery(charge)` and `battery(discharge)`.

## Parameters

### Capacity

`capacity` sets the maximum stored energy \(\bar{E}_s\) in MWh:

```python
battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)
```

### Efficiency

`eta_charge` and `eta_discharge` set round-trip efficiency. Losses are applied
during charging and discharging respectively:

```python
battery = Storage(
    'battery', charging=charge, discharging=discharge,
    capacity=100.0,
    eta_charge=0.95,
    eta_discharge=0.95,
)
```

With these values, a full charge/discharge cycle retains 90.25% of the energy.

### Self-Discharge

`relative_loss_per_hour` sets the fraction of stored energy lost per hour:

```python
battery = Storage(
    'battery', charging=charge, discharging=discharge,
    capacity=100.0,
    relative_loss_per_hour=0.001,  # 0.1%/h
)
```

### Prior Level and Cyclic Constraint

`prior_level` sets the energy level at the start of the horizon as an absolute
value in MWh. `cyclic` enforces that the storage ends at the same level it
started:

```python
# Fixed initial level (absolute MWh), no cyclic constraint
battery = Storage(..., prior_level=50.0, cyclic=False)

# Unconstrained initial level (optimizer chooses), cyclic (default)
battery = Storage(..., prior_level=None, cyclic=True)
```

The default is `prior_level=None` (unconstrained) and `cyclic=True`.

### Level Bounds

`relative_minimum_level` and `relative_maximum_level` limit the SOC as
fractions of capacity:

```python
battery = Storage(
    'battery', charging=charge, discharging=discharge,
    capacity=100.0,
    relative_minimum_level=0.2,  # never below 20%
    relative_maximum_level=0.9,  # never above 90%
)
```

## Full Example

Battery arbitrage — charge in cheap hours, discharge in expensive hours:

```python
from datetime import datetime
from fluxopt import Bus, Effect, Flow, Port, Storage, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
prices = [0.02, 0.08, 0.02, 0.08]

source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': prices})
demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

charge = Flow(bus='elec', size=50)
discharge = Flow(bus='elec', size=50)
battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)

result = optimize(
    timesteps=timesteps,
    buses=[Bus('elec')],
    effects=[Effect('cost', is_objective=True)],
    ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    storages=[battery],
)

print(result.flow_rate('battery(charge)'))
print(result.flow_rate('battery(discharge)'))
print(result.storage_level('battery'))
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Storage identifier |
| `charging` | `Flow` | required | Charging flow |
| `discharging` | `Flow` | required | Discharging flow |
| `capacity` | `float \| Sizing \| None` | `None` | Maximum stored energy [MWh] or [investment](sizing.md) |
| `eta_charge` | `TimeSeries` | `1.0` | Charging efficiency |
| `eta_discharge` | `TimeSeries` | `1.0` | Discharging efficiency |
| `relative_loss_per_hour` | `TimeSeries` | `0.0` | Self-discharge rate [1/h] |
| `prior_level` | `float \| None` | `None` | Initial energy level [MWh], None = unconstrained |
| `cyclic` | `bool` | `True` | End level must equal start level |
| `relative_minimum_level` | `TimeSeries` | `0.0` | Min SOC as fraction of capacity |
| `relative_maximum_level` | `TimeSeries` | `1.0` | Max SOC as fraction of capacity |
