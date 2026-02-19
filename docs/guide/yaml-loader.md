# YAML Loader

Define fluxopt models as YAML files with optional CSV time series.
Useful for parameter studies, config-driven workflows, and non-programmer
users.

`load_yaml` returns the same kwargs you'd pass to `optimize()`, so you can
always drop into Python for anything the YAML format doesn't cover.

## Quick Example

```yaml
# model.yaml
timeseries: data.csv

buses:
  - id: gas
  - id: heat

effects:
  - id: cost
    is_objective: true

ports:
  - id: grid
    imports:
      - bus: gas
        size: 500
        effects_per_flow_hour:
          cost: "gas_price * 1.19"
  - id: demand
    exports:
      - bus: heat
        size: 100
        fixed_relative_profile: "demand_profile"

converters:
  - id: boiler
    type: boiler
    thermal_efficiency: 0.9
    fuel:
      bus: gas
      size: 300
    thermal:
      bus: heat
      size: 200
```

```python
from fluxopt import solve_yaml

result = solve_yaml('model.yaml')
print(result.objective)
```

## Timesteps

Timesteps can come from a CSV index or be listed explicitly:

```yaml
# Option 1: derived from CSV index (preferred)
timeseries: data.csv

# Option 2: explicit list (no CSV needed)
timesteps:
  - "2024-01-01 00:00"
  - "2024-01-01 01:00"
  - "2024-01-01 02:00"
  - "2024-01-01 03:00"
```

When using a CSV, the first column is treated as the index and parsed as
datetime. Explicit timesteps are only used when no CSV is provided.

## Time Series from CSV

Point `timeseries` at a CSV file (path relative to the YAML file). Column
names become variables you can reference in expressions:

```csv
time,gas_price,demand_profile
2024-01-01 00:00,0.03,0.4
2024-01-01 01:00,0.05,0.7
2024-01-01 02:00,0.04,0.5
2024-01-01 03:00,0.06,0.6
```

```yaml
effects_per_flow_hour:
  cost: "gas_price"                 # direct column reference
  cost: "gas_price * 1.19"          # arithmetic on columns
fixed_relative_profile: [0.4, 0.7]  # inline list still works
relative_minimum: 0.3               # scalar still works
```

### Multiple CSV Files

For larger projects, split time series across files:

```yaml
timeseries:
  prices: market_prices.csv
  weather: weather_data.csv
```

All columns from all files are merged into one flat namespace. Rules:

- All CSVs must have the **same number of rows**
- Column names must be **unique across files**
- Timesteps are derived from the **first** CSV's index

### Column Names

Column names used in expressions must be valid identifiers (`gas_price`,
`solar_cf`) or wrapped in single quotes for special characters:

```yaml
effects_per_flow_hour:
  cost: "'gas-price' * 1.19"
  revenue: "'price [EUR/MWh]' / 1000"
```

## Expressions

Any string value in a time series field is evaluated as an arithmetic
expression. Supported operations:

| Syntax | Meaning |
|---|---|
| `+`, `-`, `*`, `/` | Arithmetic (standard precedence) |
| `(`, `)` | Grouping |
| `-x` | Unary minus |
| `1.5e-2` | Scientific notation |
| `column_name` | CSV column reference |
| `'special-name'` | Quoted column reference |

Expressions work element-wise on arrays. If `gas_price` is a 4-element
column, `gas_price * 1.19` produces a 4-element result.

## Buses and Effects

```yaml
buses:
  - id: gas
    carrier: natural_gas   # optional
  - id: heat

effects:
  - id: cost
    is_objective: true
  - id: co2
    unit: "t"
    maximum_total: 1000
```

## Ports

```yaml
ports:
  - id: grid
    imports:
      - bus: gas
        size: 500
        effects_per_flow_hour:
          cost: 0.04
          co2: 0.2
    exports: []             # optional, defaults to []
  - id: demand
    exports:
      - bus: heat
        size: 100
        fixed_relative_profile: "demand_profile"
```

Each import/export is a flow. All `Flow` fields are supported:

| YAML key | Flow field |
|---|---|
| `bus` | `bus` (required) |
| `id` | `id` |
| `size` | `size` (scalar or sizing dict) |
| `relative_minimum` | `relative_minimum` |
| `relative_maximum` | `relative_maximum` |
| `fixed_relative_profile` | `fixed_relative_profile` |
| `effects_per_flow_hour` | `effects_per_flow_hour` |
| `status` | `status` (`true` or dict) |
| `prior` | `prior` |

### Sizing

When `size` is a dict, it creates a `Sizing` object for capacity optimization:

```yaml
size:
  min: 10
  max: 100
  mandatory: false          # optional, default false
  effects_per_size:         # optional
    cost: 1000
```

### Status

`status: true` creates a `Status` with defaults. Use a dict for specific
parameters:

```yaml
status:
  min_uptime: 2
  min_downtime: 1
  effects_per_startup:
    cost: 50
```

## Converters

### Shorthand Form

Use `type` for built-in converter factories:

=== "Boiler"

    ```yaml
    converters:
      - id: boiler
        type: boiler
        thermal_efficiency: 0.9
        fuel:
          bus: gas
          size: 300
        thermal:
          bus: heat
          size: 200
    ```

=== "Heat Pump"

    ```yaml
    converters:
      - id: hp
        type: heat_pump
        cop: 3.5
        electrical:
          bus: elec
          size: 50
        thermal:
          bus: heat
          size: 175
    ```

=== "CHP"

    ```yaml
    converters:
      - id: chp
        type: chp
        eta_el: 0.4
        eta_th: 0.5
        fuel:
          bus: gas
          size: 500
        electrical:
          bus: elec
          size: 200
        thermal:
          bus: heat
          size: 250
    ```

### Explicit Form

For custom conversion equations, use `inputs`/`outputs` with
`conversion_factors`:

```yaml
converters:
  - id: chp
    inputs:
      - bus: gas
        size: 300
    outputs:
      - bus: elec
        size: 100
      - bus: heat
        size: 200
    conversion_factors:
      - gas: 0.4
        elec: -1
      - gas: 0.5
        heat: -1
```

Keys in `conversion_factors` reference flows by their short id (the bus
name, or explicit `id` if set).

## Storages

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
    eta_charge: 0.95
    eta_discharge: 0.95
    prior_level: 0.0
    relative_loss_per_hour: 0.001
```

`capacity` accepts a scalar or a sizing dict (same as flow `size`).

## Hybrid Workflow

`load_yaml` returns plain `optimize()` kwargs. Use this to load topology from
YAML and tweak in Python:

```python
from fluxopt import load_yaml, optimize

kwargs = load_yaml('system.yaml')

# Override a parameter
kwargs['storages'][0].cyclic = True

# Add custom dt
kwargs['dt'] = 0.25

result = optimize(**kwargs)
```

This is the recommended approach for anything beyond what YAML supports —
custom constraints, programmatic parameter sweeps, post-processing, etc.

## API Reference

::: fluxopt.yaml_loader.load_yaml

::: fluxopt.yaml_loader.solve_yaml

::: fluxopt.yaml_loader.YamlLoadError
