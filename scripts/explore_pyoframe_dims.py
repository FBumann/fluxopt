"""Explore how pyoframe handles expressions with differing dimensions.

Goal: Can we accumulate effect contributions from different sources where each
source has different sparsity (not every source contributes to every effect)?

Findings:
- Pyoframe requires explicit dimension alignment for addition
- .keep_extras() on the LARGER expression preserves its unmatched rows
- The accumulator pattern: acc.keep_extras() + sparse_contrib at each step
- Alternative: build full (effect, time, contributor) Param and .sum('contributor')
"""

import polars as pl
import pyoframe as pf

effects = pl.DataFrame({'effect': ['costs', 'co2']})
times = pl.DataFrame({'time': [0, 1, 2]})
flows = pl.DataFrame({'flow': ['boiler(gas)', 'grid(elec)']})
comps = pl.DataFrame({'comp': ['boiler', 'grid']})

m = pf.Model('highs')
m.flow_rate = pf.Variable(flows, times, lb=0)
m.on_off = pf.Variable(comps, times, lb=0, ub=1)

# ---------------------------------------------------------------------------
# 1. Contributors with different sparsity
# ---------------------------------------------------------------------------
print('=== 1. Build contributions with different effect coverage ===\n')

# Flow costs: covers BOTH effects (costs, co2)
flow_coeffs = pl.DataFrame(
    {
        'effect': ['costs', 'costs', 'co2', 'co2'],
        'flow': ['boiler(gas)', 'grid(elec)', 'boiler(gas)', 'grid(elec)'],
        'value': [50.0, 80.0, 0.5, 1.2],
    }
)
flow_contrib = (pf.Param(flow_coeffs) * m.flow_rate).sum('flow')
print(
    f'flow_contrib covers: costs, co2  (shape: effect={flow_contrib.data["effect"].n_unique()}, time={flow_contrib.data["time"].n_unique()})'
)

# Status costs: covers ONLY 'costs'
status_coeffs = pl.DataFrame(
    {
        'effect': ['costs', 'costs'],
        'comp': ['boiler', 'grid'],
        'value': [5.0, 2.0],
    }
)
status_contrib = (pf.Param(status_coeffs) * m.on_off).sum('comp')
print(
    f'status_contrib covers: costs only (shape: effect={status_contrib.data["effect"].n_unique()}, time={status_contrib.data["time"].n_unique()})'
)

# ---------------------------------------------------------------------------
# 2. Direct addition fails when sparsity differs
# ---------------------------------------------------------------------------
print('\n=== 2. Direct addition fails ===\n')

try:
    flow_contrib + status_contrib
except Exception:
    print('flow_contrib + status_contrib → PyoframeError')
    print('  "expression 1 has extra labels" (the co2 rows)\n')

# ---------------------------------------------------------------------------
# 3. keep_extras() on the LARGER expression preserves unmatched rows
# ---------------------------------------------------------------------------
print('=== 3. keep_extras() on the larger expression ===\n')

total = flow_contrib.keep_extras() + status_contrib
print(f'flow_contrib.keep_extras() + status_contrib:\n{total}\n')
print('→ co2 rows kept intact (only flow_rate terms)')
print('→ costs rows have both flow_rate AND on_off terms\n')

# ---------------------------------------------------------------------------
# 4. Accumulator pattern with keep_extras
# ---------------------------------------------------------------------------
print('=== 4. Accumulator pattern ===\n')

# Startup costs: also sparse (only 'costs')
m.startup = pf.Variable(comps, times, lb=0, ub=1)
startup_coeffs = pl.DataFrame(
    {
        'effect': ['costs'],
        'comp': ['boiler'],
        'value': [100.0],
    }
)
startup_contrib = (pf.Param(startup_coeffs) * m.startup).sum('comp')

# Chain: keep_extras on the growing accumulator at each step
acc = flow_contrib  # (costs, co2) × time
acc = acc.keep_extras() + status_contrib  # status only on costs
acc = acc.keep_extras() + startup_contrib  # startup only on costs
print(f'Chained accumulator:\n{acc}\n')

# ---------------------------------------------------------------------------
# 5. Two-bucket total: sum_time(temporal) + periodic
# ---------------------------------------------------------------------------
print('=== 5. Two-bucket total: temporal.sum(time) + periodic ===\n')

periodic = pf.Param(
    pl.DataFrame(
        {
            'effect': ['costs', 'co2'],
            'value': [1000.0, 500.0],
        }
    )
)

m.effect_total = pf.Variable(effects)
m.total_eq = m.effect_total == acc.sum('time') + periodic
print(f'Total constraint:\n{m.total_eq}\n')

# ---------------------------------------------------------------------------
# 6. Per-timestep bounds directly on expression (no intermediate variable)
# ---------------------------------------------------------------------------
print('=== 6. Per-timestep bounds on expression ===\n')

time_ub = pf.Param(
    pl.DataFrame(
        {
            'effect': ['costs', 'costs', 'costs'],
            'time': [0, 1, 2],
            'value': [500.0, 500.0, 500.0],
        }
    )
)
# acc is (effect, time) — can constrain directly
m.temporal_ub = acc.drop_extras() <= time_ub
print(f'Bound on expression (no variable needed):\n{m.temporal_ub}\n')

# ---------------------------------------------------------------------------
# 7. With intermediate variable (for solution extraction)
# ---------------------------------------------------------------------------
print('=== 7. With intermediate variable ===\n')

m.effect_per_ts = pf.Variable(effects, times)
m.temporal_eq = m.effect_per_ts == acc
print(f'Intermediate variable constraint:\n{m.temporal_eq}\n')

# ---------------------------------------------------------------------------
# 8. Alternative: explicit contributor Param with .sum('contributor')
# ---------------------------------------------------------------------------
print('=== 8. Alternative: explicit contributor column ===\n')
print('If all coefficients are known at table-build time (no variables),')
print('you can build a single (effect, time, contributor, value) DataFrame')
print('and .sum("contributor"). But since our contributions involve different')
print('variables (flow_rate, on_off, startup), we MUST use the accumulator pattern.')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n=== SUMMARY ===\n')
print("""
Recommended pattern for accumulating sparse effect contributions:

    # Each source: (effect, <source_dim>, time) → .sum(<source_dim>) → (effect, time)
    # Note: each source may cover different subsets of effects!

    acc = flow_cost_contrib                     # full (effect, time) — or start with zero base
    acc = acc.keep_extras() + status_contrib    # sparse: only some effects
    acc = acc.keep_extras() + startup_contrib   # sparse: only some effects

    # Constrain:
    m.effect_temporal == acc                # (effect, time) = (effect, time)
    m.effect_total == acc.sum('time') + periodic_costs   # (effect,) = (effect,)

Key rules:
  1. .keep_extras() goes on the LARGER expression (the one with extra index labels)
  2. In the accumulator pattern, the accumulator grows → always acc.keep_extras()
  3. After .sum('time'), dimensions align to (effect,) for periodic addition
  4. .drop_extras() needed when constraining against a param that covers fewer labels
  5. No intermediate variable needed for bounds — can constrain expressions directly
""")
