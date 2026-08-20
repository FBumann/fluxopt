import numpy as np
from conftest import ts

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Storage, optimize

# Fixed three-step demand: 50, 80, 60 MW -> 190 MWh at dt=1.
_DEMAND_PROFILE = [0.5, 0.8, 0.6]
_DEMAND_ENERGY = 190.0


def _solve(source, *, dt=None, storages=()):
    """One-bus electricity model: `source` imports, a fixed demand exports."""
    demand = Flow(carrier='elec', size=100, fixed_relative_profile=_DEMAND_PROFILE)
    return optimize(
        timesteps=ts(3),
        dt=dt,
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        storages=list(storages),
    )


def test_stats_summary_quickstart():
    """`summary` is a KPI namespace: objective, effect totals, per-flow utilization."""
    summary = _solve(Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})).stats.summary

    # KPIs present; no storages -> no storage KPIs.
    assert set(summary.data_vars) == {
        'objective',
        'effect_totals',
        'total_duration',
        'size',
        'total_flow_hours',
        'capacity_factor',
    }

    # Meaningful content, not just keys.
    assert np.isfinite(summary['objective'].item())
    assert np.isclose(summary['effect_totals'].sel(effect='cost').item(), _DEMAND_ENERGY * 0.04)
    assert summary['total_duration'].item() == 3.0

    # It's a namespace, not a flat table: per-flow KPIs share `flow`, effect
    # totals live on `effect`.
    assert 'effect' in summary['effect_totals'].dims
    for var in ('size', 'total_flow_hours', 'capacity_factor'):
        assert summary[var].dims == ('flow',)

    flow = 'grid(elec)'
    size = summary['size'].sel(flow=flow).item()
    tfh = summary['total_flow_hours'].sel(flow=flow).item()
    cf = summary['capacity_factor'].sel(flow=flow).item()

    assert size == 200
    assert np.isclose(tfh, _DEMAND_ENERGY)
    # Capacity factor is the dimensionless quotient, in [0, 1].
    assert 0 <= cf <= 1
    assert np.isclose(cf, _DEMAND_ENERGY / (200 * 3))


def test_unsized_flow_has_nan_size_but_real_throughput():
    """An unsized flow reports NaN size/CF, yet its throughput stays visible."""
    stats = _solve(Flow(carrier='elec', effects_per_flow_hour={'cost': 0.04})).stats  # size=None
    flow = 'grid(elec)'

    assert np.isnan(stats.resolved_sizes.sel(flow=flow).item())
    assert np.isnan(stats.capacity_factor.sel(flow=flow).item())
    # The design point: throughput is real even when size is unknown.
    assert np.isclose(stats.total_flow_hours.sel(flow=flow).item(), _DEMAND_ENERGY)


def test_resolved_sizes_fills_in_invested_size():
    """For an invested flow, resolved_sizes uses the optimized size, and CF follows."""
    # A per-size cost makes the solver pick the smallest feasible size (the peak).
    sizing = Sizing(size_min=0, size_max=500, effects_per_size={'cost': 1.0})
    result = _solve(Flow(carrier='elec', size=sizing, effects_per_flow_hour={'cost': 0.04}))
    stats = result.stats
    flow = 'grid(elec)'

    invested = result.sizes.sel(flow=flow).item()
    resolved = stats.resolved_sizes.sel(flow=flow).item()
    # An invested flow has no row in `sizes`; resolved_sizes fills it from the solution.
    assert flow not in result.data.flows.sized_ids
    assert np.isfinite(resolved)
    assert np.isclose(resolved, invested)
    assert np.isclose(resolved, 80)  # peak demand

    cf = stats.capacity_factor.sel(flow=flow).item()
    assert np.isclose(cf, _DEMAND_ENERGY / (resolved * 3))


def test_capacity_factor_is_horizon_independent():
    """Scaling every timestep duration leaves CF unchanged while throughput scales."""

    def source():
        return Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})

    flow = 'grid(elec)'

    short = _solve(source(), dt=[1.0, 1.0, 1.0]).stats
    long = _solve(source(), dt=[2.0, 2.0, 2.0]).stats

    # Throughput and horizon both double...
    assert np.isclose(long.total_duration.item(), 2 * short.total_duration.item())
    assert np.isclose(
        long.total_flow_hours.sel(flow=flow).item(),
        2 * short.total_flow_hours.sel(flow=flow).item(),
    )
    # ...but the dimensionless capacity factor does not change.
    assert np.isclose(
        long.capacity_factor.sel(flow=flow).item(),
        short.capacity_factor.sel(flow=flow).item(),
    )


def test_stats_summary_with_storage():
    """With storages, `summary` adds capacity and relative mean level on `storage`."""
    source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': [0.1, 0.9, 0.1]})
    demand = Flow(carrier='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5])
    storage = Storage(
        id='batt', charging=Flow(carrier='elec', size=80), discharging=Flow(carrier='elec', size=80), capacity=80
    )

    result = optimize(
        timesteps=ts(3),
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='load', exports=[demand])],
        storages=[storage],
    )
    stats = result.stats
    summary = stats.summary

    assert 'capacity' in summary
    assert 'relative_mean_level' in summary
    assert summary['capacity'].dims == ('storage',)
    assert summary['relative_mean_level'].dims == ('storage',)

    assert summary['capacity'].sel(storage='batt').item() == 80

    # relative_mean_level is the dt-weighted mean level over capacity, in [0, 1].
    rml = stats.relative_mean_level.sel(storage='batt').item()
    assert 0 <= rml <= 1

    dims = result.data.dims
    expected = float(
        (result.storage_levels * dims.dt * dims.weights).sum('time').sel(storage='batt') / stats.total_duration / 80
    )
    assert np.isclose(rml, expected)
