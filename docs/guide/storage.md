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

### Initial Charge State

`initial_charge_state` sets the energy level at the start of the horizon:

=== "Python"

    ```python
    # Fixed initial state (absolute MWh)
    battery = Storage(..., initial_charge_state=50.0)

    # Fraction of capacity (values <= 1 are interpreted as fractions)
    battery = Storage(..., initial_charge_state=0.5)  # 50% of capacity

    # Cyclic: end state must equal start state
    battery = Storage(..., initial_charge_state='cyclic')
    ```

=== "YAML"

    ```yaml
    # Fixed initial state
    initial_charge_state: 50.0

    # Fraction of capacity
    initial_charge_state: 0.5

    # Cyclic
    initial_charge_state: cyclic
    ```

The default is `0.0` (empty).

### Charge State Bounds

`relative_minimum_charge_state` and `relative_maximum_charge_state` limit the
SOC as fractions of capacity:

=== "Python"

    ```python
    battery = Storage(
        'battery', charging=charge, discharging=discharge,
        capacity=100.0,
        relative_minimum_charge_state=0.2,  # never below 20%
        relative_maximum_charge_state=0.9,  # never above 90%
    )
    ```

=== "YAML"

    ```yaml
    storages:
      - id: battery
        charging: { bus: elec, size: 50 }
        discharging: { bus: elec, size: 50 }
        capacity: 100
        relative_minimum_charge_state: 0.2
        relative_maximum_charge_state: 0.9
    ```

## Full Example

Battery arbitrage — charge in cheap hours, discharge in expensive hours:

=== "Python"

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
| `capacity` | `float \| None` | `None` | Maximum stored energy [MWh] |
| `eta_charge` | `TimeSeries` | `1.0` | Charging efficiency |
| `eta_discharge` | `TimeSeries` | `1.0` | Discharging efficiency |
| `relative_loss_per_hour` | `TimeSeries` | `0.0` | Self-discharge rate [1/h] |
| `initial_charge_state` | `float \| str \| None` | `0.0` | Initial energy or `'cyclic'` |
| `relative_minimum_charge_state` | `TimeSeries` | `0.0` | Min SOC as fraction of capacity |
| `relative_maximum_charge_state` | `TimeSeries` | `1.0` | Max SOC as fraction of capacity |
