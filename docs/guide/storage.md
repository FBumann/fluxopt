# Storage

A `Storage` models energy storage with charge/discharge flows, capacity,
efficiency, and self-discharge.

See [Storage (Math)](../math/storage.md) for the formulation.

## Basic Construction

A storage needs two flows (charging and discharging) on the same bus:

=== "Python"

    ```python
    from fluxopt import Flow, Storage

    charge = Flow(bus='elec', size=50)     # max charge rate 50 MW
    discharge = Flow(bus='elec', size=50)  # max discharge rate 50 MW

    battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging:
          bus: elec
          size: 50
        discharging:
          bus: elec
          size: 50
        capacity: 100
    ```

Flow ids are auto-qualified: `battery(charge)` and `battery(discharge)`.

## Parameters

### Capacity

`capacity` sets the maximum stored energy $\bar{E}_s$ in MWh:

=== "Python"

    ```python
    battery = Storage('battery', charging=charge, discharging=discharge, capacity=100.0)
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging: { bus: elec, size: 50 }
        discharging: { bus: elec, size: 50 }
        capacity: 100
    ```

### Efficiency

`eta_charge` and `eta_discharge` set round-trip efficiency. Losses are applied
during charging and discharging respectively:

=== "Python"

    ```python
    battery = Storage(
        'battery', charging=charge, discharging=discharge,
        capacity=100.0,
        eta_charge=0.95,
        eta_discharge=0.95,
    )
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging: { bus: elec, size: 50 }
        discharging: { bus: elec, size: 50 }
        capacity: 100
        eta_charge: 0.95
        eta_discharge: 0.95
    ```

With these values, a full charge/discharge cycle retains 90.25% of the energy.

### Self-Discharge

`relative_loss_per_hour` sets the fraction of stored energy lost per hour:

=== "Python"

    ```python
    battery = Storage(
        'battery', charging=charge, discharging=discharge,
        capacity=100.0,
        relative_loss_per_hour=0.001,  # 0.1%/h
    )
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging: { bus: elec, size: 50 }
        discharging: { bus: elec, size: 50 }
        capacity: 100
        relative_loss_per_hour: 0.001
    ```

### Prior Level and Cyclic Constraint

`prior_level` sets the energy level at the start of the horizon as an absolute
value in MWh. `cyclic` enforces that the storage ends at the same level it
started:

=== "Python"

    ```python
    # Fixed initial level (absolute MWh), no cyclic constraint
    battery = Storage(..., prior_level=50.0, cyclic=False)

    # Unconstrained initial level (optimizer chooses), cyclic (default)
    battery = Storage(..., prior_level=None, cyclic=True)
    ```

=== "YAML"

    ```yaml
    # Fixed initial level
    prior_level: 50.0
    cyclic: false

    # Unconstrained initial level, cyclic (default)
    cyclic: true
    ```

The default is `prior_level=None` (unconstrained) and `cyclic=True`.

### Level Bounds

`relative_minimum_level` and `relative_maximum_level` limit the SOC as
fractions of capacity:

=== "Python"

    ```python
    battery = Storage(
        'battery', charging=charge, discharging=discharge,
        capacity=100.0,
        relative_minimum_level=0.2,  # never below 20%
        relative_maximum_level=0.9,  # never above 90%
    )
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging: { bus: elec, size: 50 }
        discharging: { bus: elec, size: 50 }
        capacity: 100
        relative_minimum_level: 0.2
        relative_maximum_level: 0.9
    ```

## Full Example

Battery arbitrage — charge in cheap hours, discharge in expensive hours:

=== "Python"

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

=== "YAML"

    ```yaml
    # model.yaml
    timesteps:
      - "2024-01-01 00:00"
      - "2024-01-01 01:00"
      - "2024-01-01 02:00"
      - "2024-01-01 03:00"

    buses:
      - id: elec

    effects:
      - id: cost
        is_objective: true

    ports:
      - id: grid
        imports:
          - bus: elec
            size: 200
            effects_per_flow_hour:
              cost: [0.02, 0.08, 0.02, 0.08]
      - id: demand
        exports:
          - bus: elec
            size: 100
            fixed_relative_profile: [0.5, 0.5, 0.5, 0.5]

    storages:
      - id: battery
        charging:
          bus: elec
          size: 50
        discharging:
          bus: elec
          size: 50
        capacity: 100
    ```

    ```python
    from fluxopt import solve_yaml

    result = solve_yaml('model.yaml')
    print(result.flow_rate('battery(charge)'))
    print(result.flow_rate('battery(discharge)'))
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
