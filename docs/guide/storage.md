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

### Initial Charge State

`initial_charge_state` sets the energy level at the start of the horizon:

```python
# Fixed initial state (absolute MWh)
battery = Storage(..., initial_charge_state=50.0)

# Fraction of capacity (values <= 1 are interpreted as fractions)
battery = Storage(..., initial_charge_state=0.5)  # 50% of capacity

# Cyclic: end state must equal start state
battery = Storage(..., initial_charge_state='cyclic')
```

The default is `0.0` (empty).

### Charge State Bounds

`relative_minimum_charge_state` and `relative_maximum_charge_state` limit the
SOC as fractions of capacity:

```python
battery = Storage(
    'battery', charging=charge, discharging=discharge,
    capacity=100.0,
    relative_minimum_charge_state=0.2,  # never below 20%
    relative_maximum_charge_state=0.9,  # never above 90%
)
```

## Full Example

Battery arbitrage — charge in cheap hours, discharge in expensive hours:

```python
from datetime import datetime
from fluxopt import Bus, Effect, Flow, Port, Storage, solve

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
prices = [0.02, 0.08, 0.02, 0.08]

source = Flow(bus='elec', size=200, effects_per_flow_hour={'cost': prices})
demand = Flow(bus='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])

charge = Flow(bus='elec', size=50)
discharge = Flow(bus='elec', size=50)
battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)

result = solve(
    timesteps=timesteps,
    buses=[Bus('elec')],
    effects=[Effect('cost', is_objective=True)],
    ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    storages=[battery],
)

print(result.flow_rate('battery(charge)'))
print(result.flow_rate('battery(discharge)'))
print(result.charge_state('battery'))
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Storage identifier |
| `charging` | `Flow` | required | Charging flow |
| `discharging` | `Flow` | required | Discharging flow |
| `capacity` | `float \| None` | `None` | Maximum stored energy [MWh] |
| `eta_charge` | `TimeSeries` | `1.0` | Charging efficiency |
| `eta_discharge` | `TimeSeries` | `1.0` | Discharging efficiency |
| `relative_loss_per_hour` | `TimeSeries` | `0.0` | Self-discharge rate [1/h] |
| `initial_charge_state` | `float \| str \| None` | `0.0` | Initial energy or `'cyclic'` |
| `relative_minimum_charge_state` | `TimeSeries` | `0.0` | Min SOC as fraction of capacity |
| `relative_maximum_charge_state` | `TimeSeries` | `1.0` | Max SOC as fraction of capacity |
