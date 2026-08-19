"""User-runnable benchmark: build a few realistic energy systems, report speed and memory.

Run it against your installation to see how fast fluxopt's build pipeline
(Elements → ModelData → linopy model) is on your hardware::

    python -m fluxopt.benchmark                        # all systems, one hourly year
    python -m fluxopt.benchmark district_heating       # a single system
    python -m fluxopt.benchmark --timesteps 720        # one month instead of a year
    python -m fluxopt.benchmark --solve                # also time the HiGHS solve
    python -m fluxopt.benchmark --json                 # machine-readable output

The reference systems are realistic, readable models — constant and
time-varying data, several effects and cross-effect couplings — so the numbers
reflect real workloads and the builders double as examples:

- ``district_heating`` — municipal utility: gas boiler, ramp-limited CHP and
  a heat pump with a weather-driven COP feed a heat network backed by a
  hot-water tank; seasonal gas tariff, day-ahead electricity prices, CO2
  priced into cost.
- ``industry_park`` — factory site: a steam boiler fleet with on/off unit
  commitment, a gas-engine CHP with a piecewise part-load curve, investment
  decisions for an electrode boiler and a steam accumulator, and an annual
  CO2 cap; three-shift steam demand.
- ``green_city`` — sector-coupled city: wind (PPA with a contracted energy
  cap), rooftop PV and a grid connection supply a battery (sized by the
  optimizer) and two district-heating networks; cost, CO2 and primary-energy
  accounting.
- ``energy_transition`` — ``green_city`` planned over eight five-year
  investment periods: growing demand, a decarbonizing grid, a rising carbon
  price, and the battery as a multi-period ``Investment`` (15-year lifetime,
  capex learning curve, recurring O&M).
- ``stress`` — the exception to realistic and readable: an abstract,
  structure-only stress workload with neutral ids and fixed-seed random
  parameters. ~190 flows over 16 investment periods, a 30-effect graph,
  optional ``Sizing`` in bulk, multi-node balances and
  piecewise/status/storage features; its ``--timesteps`` budget is split
  across the periods.

All data is deterministic (any randomness is drawn from fixed seeds), and
each system is built in a fresh subprocess so peak memory is attributed per
model. Memory is whole-process peak RSS — the number that has to fit in your
RAM; for allocator-level profiles use pytest-benchmem on
``benchmark/test_reference.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta
from importlib.metadata import version
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    Investment,
    ModelData,
    PiecewiseConversion,
    Port,
    Sizing,
    Status,
    Storage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

Elements = dict[str, Any]

HOURS_PER_YEAR = 8760
CARBON_PRICE = 0.045
"""EUR per kg CO2 (45 EUR/t), fed into ``cost`` via ``Effect.contribution_from``."""

GAS_PRICE = 35.0
"""EUR per MWh of natural gas (flat supply tariff)."""

GAS_CO2 = 202.0
"""kg CO2 per MWh of natural gas burned."""


def _hourly_index(n: int) -> list[datetime]:
    """``n`` hourly timesteps starting Monday, 2024-01-01."""
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def _clock(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hour-of-day, day index and weekend mask for the hourly index."""
    t = np.arange(n)
    hour = t % 24
    day = t // 24
    weekend = (day % 7) >= 5
    return hour, day, weekend


def _winter(day: np.ndarray) -> np.ndarray:
    """Seasonal factor: 1 at mid-winter, 0 at mid-summer."""
    return 0.5 + 0.5 * np.cos(2 * np.pi * day / 365.0)


def _gas_price(n: int) -> np.ndarray:
    """Indexed gas tariff [EUR/MWh]: winter premium on a 30 EUR base."""
    _, day, _ = _clock(n)
    return 30.0 + 11.0 * _winter(day)


def _heat_demand(n: int) -> np.ndarray:
    """Relative heat load: seasonal base plus morning and evening peaks."""
    hour, day, _ = _clock(n)
    peaks = 0.12 * np.exp(-((hour - 7.0) ** 2) / 8.0) + 0.10 * np.exp(-((hour - 19.0) ** 2) / 12.0)
    return np.clip(0.12 + 0.6 * _winter(day) + peaks, 0.05, 1.0)


def _elec_price(n: int) -> np.ndarray:
    """Day-ahead electricity price [EUR/MWh]: peaks at 8 h and 19 h, a solar dip at midday, cheaper weekends."""
    hour, day, weekend = _clock(n)
    peaks = 18.0 * np.exp(-((hour - 8.0) ** 2) / 6.0) + 22.0 * np.exp(-((hour - 19.0) ** 2) / 8.0)
    solar_dip = 14.0 * np.exp(-((hour - 13.0) ** 2) / 10.0) * (1.0 - _winter(day))
    return 52.0 + 14.0 * _winter(day) - 8.0 * weekend + peaks - solar_dip


def _grid_co2(n: int) -> np.ndarray:
    """Grid CO2 intensity [kg/MWh]: higher in winter, dips at midday with solar."""
    hour, day, _ = _clock(n)
    solar_dip = 130.0 * np.exp(-((hour - 13.0) ** 2) / 12.0) * (1.0 - 0.7 * _winter(day))
    return np.clip(340.0 + 70.0 * _winter(day) - solar_dip, 90.0, 600.0)


def _heat_pump_cop(n: int) -> np.ndarray:
    """Air-source heat-pump COP from a seasonal + daily ambient-temperature curve."""
    hour, day, _ = _clock(n)
    temp = 11.0 - 11.0 * np.cos(2 * np.pi * (day - 15.0) / 365.0) + 3.0 * np.sin(2 * np.pi * (hour - 15.0) / 24.0)
    return np.clip(2.9 + 0.09 * temp, 1.6, 5.0)


def _solar(n: int) -> np.ndarray:
    """Relative PV availability: daylight bell, stronger in summer."""
    hour, day, _ = _clock(n)
    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.5) / 13.0))
    return daylight * (0.9 - 0.55 * _winter(day))


def _wind(n: int) -> np.ndarray:
    """Relative wind availability: overlapping weather fronts, windier in winter."""
    t = np.arange(n)
    fronts = 0.38 + 0.25 * np.sin(t / 9.3) + 0.2 * np.sin(t / 37.0 + 2.0) + 0.15 * np.sin(t / 171.0 + 1.0)
    return np.clip(fronts + 0.15 * _winter(t // 24), 0.0, 1.0)


def _steam_demand(n: int) -> np.ndarray:
    """Relative steam load: three-shift weekdays, reduced weekend crew."""
    hour, _, weekend = _clock(n)
    weekday_shift = np.where((hour >= 6) & (hour < 22), 0.85, 0.6)
    return np.where(weekend, 0.35, weekday_shift)


def _city_elec_demand(n: int) -> np.ndarray:
    """Relative city electricity load: business-hours peak, evening shoulder, weekend reduction."""
    hour, day, weekend = _clock(n)
    business = 0.25 * np.exp(-((hour - 11.0) ** 2) / 24.0) + 0.15 * np.exp(-((hour - 19.0) ** 2) / 10.0)
    return np.clip(0.45 + 0.08 * _winter(day) + business - 0.1 * weekend, 0.2, 1.0)


def district_heating(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Municipal district-heating utility.

    A 15 MW gas boiler, a gas CHP and an 8 MW air-source heat pump with a
    weather-driven COP feed a 20 MW-peak heat network backed by an 80 MWh
    hot-water tank. Gas is bought at an indexed tariff with a winter premium;
    electricity is bought and sold at a day-ahead price profile; every kg of
    CO2 — burned on site or embodied in grid power — is priced into cost at
    45 EUR/t.
    """
    n = timesteps
    price = _elec_price(n)
    return {
        'timesteps': _hourly_index(n),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='heat'), Carrier(id='ambient')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg'),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[
                    Flow(
                        carrier='gas', size=60.0, effects_per_flow_hour={'cost': _gas_price(n).tolist(), 'co2': GAS_CO2}
                    )
                ],
            ),
            Port(
                id='power_exchange',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=30.0,
                        effects_per_flow_hour={'cost': price.tolist(), 'co2': _grid_co2(n).tolist()},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=30.0,
                        effects_per_flow_hour={'cost': (-0.95 * price).tolist()},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='heat_network',
                exports=[Flow(carrier='heat', size=20.0, fixed_relative_profile=_heat_demand(n).tolist())],
            ),
        ],
        'converters': [
            Converter.boiler('gas_boiler', 0.92, Flow(carrier='gas'), Flow(carrier='heat', size=15.0)),
            Converter.chp(
                'chp',
                0.38,
                0.45,
                Flow(carrier='gas', size=25.0, ramp_up_per_hour=0.4, ramp_down_per_hour=0.4),
                Flow(carrier='elec'),
                Flow(carrier='heat'),
            ),
            Converter.heat_pump(
                'heat_pump',
                _heat_pump_cop(n).tolist(),
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(carrier='heat', size=8.0),
            ),
        ],
        'storages': [
            Storage(
                id='hot_water_tank',
                charging=Flow(carrier='heat', size=10.0),
                discharging=Flow(carrier='heat', size=10.0),
                capacity=80.0,
                relative_loss_per_hour=0.003,
            ),
        ],
    }


def industry_park(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Industrial steam-and-power site with unit commitment and investment.

    Two 20 MW steam boilers with minimum load, minimum up/down times and
    startup costs cover a three-shift steam demand alongside a gas-engine CHP
    whose part-load efficiency follows a piecewise-linear curve. The optimizer
    may additionally invest in an electrode boiler (0-20 MW) and a steam
    accumulator (0-60 MWh), both carrying annualized capital cost and embodied
    CO2, and site emissions are capped at 80 kt CO2 per year.
    """
    n = timesteps
    price = _elec_price(n)
    steam_boilers = [
        Converter.boiler(
            f'steam_boiler_{i}',
            0.90,
            Flow(carrier='gas'),
            Flow(
                carrier='steam',
                size=20.0,
                relative_rate_min=0.35,
                status=Status(
                    uptime_min=4,
                    downtime_min=2,
                    effects_per_startup={'cost': 400.0},
                    effects_per_running_hour={'cost': 18.0},
                ),
            ),
        )
        for i in (1, 2)
    ]
    electrode_boiler = Converter.boiler(
        'electrode_boiler',
        0.99,
        Flow(carrier='elec'),
        Flow(
            carrier='steam',
            size=Sizing(size_min=0.0, size_max=20.0, effects_per_size={'cost': 9000.0, 'co2': 1800.0}),
        ),
    )
    site_chp = Converter(
        id='site_chp',
        inputs=[Flow(carrier='gas', size=30.0, ramp_up_per_hour=0.3, ramp_down_per_hour=0.3)],
        outputs=[Flow(carrier='elec'), Flow(carrier='steam')],
        conversion=PiecewiseConversion(
            points={
                'gas': [0.0, 12.0, 20.0, 30.0],
                'elec': [0.0, 3.6, 7.4, 12.0],
                'steam': [0.0, 6.0, 9.2, 12.6],
            }
        ),
    )
    return {
        'timesteps': _hourly_index(n),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='steam')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg', total_max=8.0e7),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[Flow(carrier='gas', size=90.0, effects_per_flow_hour={'cost': GAS_PRICE, 'co2': GAS_CO2})],
            ),
            Port(
                id='power_grid',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=40.0,
                        effects_per_flow_hour={'cost': price.tolist(), 'co2': _grid_co2(n).tolist()},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=15.0,
                        effects_per_flow_hour={'cost': (-0.9 * price).tolist()},
                    )
                ],
            ),
            Port(
                id='process_steam',
                exports=[Flow(carrier='steam', size=45.0, fixed_relative_profile=_steam_demand(n).tolist())],
            ),
            Port(
                id='machinery',
                exports=[Flow(carrier='elec', size=12.0, fixed_relative_profile=_city_elec_demand(n).tolist())],
            ),
        ],
        'converters': [*steam_boilers, electrode_boiler, site_chp],
        'storages': [
            Storage(
                id='steam_accumulator',
                charging=Flow(carrier='steam', size=15.0),
                discharging=Flow(carrier='steam', size=15.0),
                capacity=Sizing(size_min=0.0, size_max=60.0, effects_per_size={'cost': 1200.0, 'co2': 300.0}),
                relative_loss_per_hour=0.01,
            ),
        ],
    }


def green_city(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Sector-coupled city energy system.

    A wind PPA, rooftop PV and a grid connection (hourly prices and CO2
    intensity) supply the city load, a battery sized by the optimizer, and two
    district-heating networks — each served by a heat pump with weather-driven
    COP, a gas peak boiler and a hot-water tank. Tracks cost, CO2 and primary
    energy; CO2 is priced into cost at 45 EUR/t.
    """
    n = timesteps
    price = _elec_price(n)
    cop = _heat_pump_cop(n).tolist()
    demand_north = _heat_demand(n)
    demand_south = np.roll(demand_north, 1)
    districts = [('north', demand_north, 25.0, 120.0), ('south', demand_south, 15.0, 60.0)]
    heat_ports = [
        Port(
            id=f'heat_network_{name}',
            exports=[Flow(carrier=f'heat_{name}', size=peak, fixed_relative_profile=demand.tolist())],
        )
        for name, demand, peak, _ in districts
    ]
    heat_plants = [
        converter
        for name, _, peak, _ in districts
        for converter in (
            Converter.heat_pump(
                f'heat_pump_{name}',
                cop,
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(carrier=f'heat_{name}', size=0.6 * peak),
            ),
            Converter.boiler(
                f'peak_boiler_{name}', 0.93, Flow(carrier='gas'), Flow(carrier=f'heat_{name}', size=0.8 * peak)
            ),
        )
    ]
    tanks = [
        Storage(
            id=f'tank_{name}',
            charging=Flow(carrier=f'heat_{name}', size=0.5 * peak),
            discharging=Flow(carrier=f'heat_{name}', size=0.5 * peak),
            capacity=capacity,
            relative_loss_per_hour=0.003,
        )
        for name, _, peak, capacity in districts
    ]
    return {
        'timesteps': _hourly_index(n),
        'carriers': [
            Carrier(id='elec'),
            Carrier(id='gas'),
            Carrier(id='ambient'),
            Carrier(id='heat_north'),
            Carrier(id='heat_south'),
        ],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg'),
            Effect(id='primary_energy', unit='MWh'),
        ],
        'ports': [
            Port(
                id='wind_farm',
                imports=[
                    Flow(
                        carrier='elec',
                        size=60.0,
                        relative_rate_max=_wind(n).tolist(),
                        flow_hours_max=150_000.0,
                        effects_per_flow_hour={'cost': 58.0, 'primary_energy': 0.03},
                    )
                ],
            ),
            Port(
                id='rooftop_pv',
                imports=[
                    Flow(
                        carrier='elec',
                        size=35.0,
                        relative_rate_max=_solar(n).tolist(),
                        effects_per_flow_hour={'cost': 21.0, 'primary_energy': 0.03},
                    )
                ],
            ),
            Port(
                id='transmission_grid',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=80.0,
                        effects_per_flow_hour={
                            'cost': price.tolist(),
                            'co2': _grid_co2(n).tolist(),
                            'primary_energy': 1.9,
                        },
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=40.0,
                        effects_per_flow_hour={'cost': (-0.9 * price).tolist()},
                    )
                ],
            ),
            Port(
                id='gas_grid',
                imports=[
                    Flow(
                        carrier='gas',
                        size=50.0,
                        effects_per_flow_hour={'cost': GAS_PRICE, 'co2': GAS_CO2, 'primary_energy': 1.1},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='city_load',
                exports=[Flow(carrier='elec', size=45.0, fixed_relative_profile=_city_elec_demand(n).tolist())],
            ),
            *heat_ports,
        ],
        'converters': heat_plants,
        'storages': [
            Storage(
                id='battery',
                charging=Flow(carrier='elec', size=25.0),
                discharging=Flow(carrier='elec', size=25.0),
                capacity=Sizing(size_min=0.0, size_max=200.0, effects_per_size={'cost': 14000.0, 'co2': 65000.0}),
                eta_charge=0.97,
                eta_discharge=0.97,
                relative_level_min=0.1,
            ),
            *tanks,
        ],
    }


def energy_transition(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """The ``green_city`` system planned over eight five-year investment periods.

    Each period 2025-2060 is represented by a full hourly year (the ``period``
    dimension multiplies every temporal variable): electricity and heat demand
    grow with electrification, grid CO2 intensity falls as the surrounding
    power system decarbonizes, and the carbon price rises from 45 to 130 EUR/t.
    The battery becomes a proper multi-period ``Investment``: 15-year lifetime
    (three periods), overnight capex falling along a learning curve, fixed O&M
    recurring over each build's life. At the default horizon this is a
    ~2 million variable model.
    """
    n = timesteps
    periods = list(range(2025, 2065, 5))
    demand_growth = np.linspace(0.55, 1.0, len(periods))
    grid_decarbonization = np.linspace(1.0, 0.25, len(periods))
    elements = green_city(n)
    time_index = pd.DatetimeIndex(elements['timesteps'], name='time')
    period_index = pd.Index(periods, name='period')

    def by_period(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=['period'], coords={'period': periods})

    def spread(profile: Any, per_period: np.ndarray) -> pd.DataFrame:
        """Hourly profile times per-period factors → a (time, period) DataFrame."""
        return pd.DataFrame(np.outer(np.asarray(profile), per_period), index=time_index, columns=period_index)

    ports = {port.id: port for port in elements['ports']}
    for port_id in ('city_load', 'heat_network_north', 'heat_network_south'):
        flow = ports[port_id].exports[0]
        grown = flow.model_copy(update={'fixed_relative_profile': spread(flow.fixed_relative_profile, demand_growth)})
        ports[port_id] = ports[port_id].model_copy(update={'exports': [grown]})
    grid = ports['transmission_grid']
    buy = grid.imports[0]
    grid_effects = dict(buy.effects_per_flow_hour)
    grid_effects['co2'] = spread(grid_effects['co2'], grid_decarbonization)
    cleaner_buy = buy.model_copy(update={'effects_per_flow_hour': grid_effects})
    ports['transmission_grid'] = grid.model_copy(update={'imports': [cleaner_buy]})
    elements['ports'] = list(ports.values())
    rising_carbon_price = by_period(np.linspace(CARBON_PRICE, 0.13, len(periods)))
    elements['effects'] = [
        Effect(id='cost', unit='EUR', contribution_from={'co2': rising_carbon_price}),
        Effect(id='co2', unit='kg'),
        Effect(id='primary_energy', unit='MWh'),
    ]
    storages = {storage.id: storage for storage in elements['storages']}
    learning_curve = Investment(
        size_min=0.0,
        size_max=200.0,
        lifetime=3,
        effects_per_size_at_build={'cost': by_period(np.linspace(220_000.0, 80_000.0, len(periods))), 'co2': 65_000.0},
        effects_per_size_recurring={'cost': 3_000.0},
    )
    storages['battery'] = storages['battery'].model_copy(update={'capacity': learning_curve})
    elements['storages'] = list(storages.values())
    elements['periods'] = periods
    elements['period_weights'] = [5.0] * len(periods)
    return elements


STRESS_PERIODS = 16
"""Investment periods of ``stress``; its ``timesteps`` budget is split across them."""

_STRESS_GAPS = (2, 1, 2, 2, 1, 3)
"""Yearly spacing between consecutive ``stress`` periods (irregular, like a staged plan)."""


def _stress_years(n: int, start: int = 2025) -> list[int]:
    """``n`` irregularly spaced investment years."""
    years = [start]
    for i in range(n - 1):
        years.append(years[-1] + _STRESS_GAPS[i % len(_STRESS_GAPS)])
    return years


def stress(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Abstract, structure-only stress workload for the model build.

    All ids are neutral (k*/conv_*/gen_*/leaf_*) and all parameters are
    rng-drawn from a fixed seed. The system concentrates the drivers that
    dominate build time and memory in large multi-period MILPs:

    - ~190 flows over (time, period); rate variables dominate the model
    - (time, period) coefficient arrays on most flows, per-period lump arrays
      on sizings
    - optional ``Sizing`` in bulk (capacity-charge sizings on inputs, range
      sizings on the per-period build fleets) -> the binary population
    - a 30-effect graph at ~1.2 effect entries per flow: weighted aggregation
      chains, one negative cross-effect factor, per-effect period weights,
      periodic bounds, two lifetime budget effects with unit period weights
    - multi-node carriers (12 carrier:node balance rows)
    - one piecewise converter with Status + availability (SOS2), one status
      flow with running-hour effects, a storage fleet with time-varying level
      caps, one load-factor bound, a fixed-profile sink

    Not exercised (unsupported in current fluxopt): period-varying conversion
    factors and period-varying storage level bounds. ``contribution_from``
    factors are binary-exact scalars (0.0625, 0.25, ...) on purpose — the
    time-variance check on the coefficient tensor is bit-exact, so a factor
    that picks up any float noise counts as time-varying and is rejected into
    lump-bearing source effects.

    ``timesteps`` is the total temporal budget, split across the 16 periods
    (the default full year gives 547 hourly steps per period).
    """
    n_timesteps = max(24, timesteps // STRESS_PERIODS)
    n_periods = STRESS_PERIODS
    rng = np.random.default_rng(7)
    time_index = pd.DatetimeIndex(_hourly_index(n_timesteps), name='time')
    years = _stress_years(n_periods)
    periods = pd.Index(years, name='period')
    period_weights = [float(w) for w in np.diff(years)] + [4.0]

    # -- generated data --------------------------------------------------------------------
    hours = np.arange(n_timesteps)
    seasonal = 0.5 + 0.4 * np.cos(2 * np.pi * hours / 8760)
    daily = 0.1 * np.sin(2 * np.pi * hours / 24)
    escalation = xr.DataArray([1.025 ** (y - years[0]) for y in years], coords=(periods,))
    npv = [float(1 / 1.035 ** (y - years[0])) for y in years]

    def tprofile(base: float, amp: float = 0.2) -> xr.DataArray:
        vals = base * (1 + amp * (seasonal - 0.5) + daily + 0.05 * rng.standard_normal(n_timesteps))
        return xr.DataArray(vals, coords=(time_index,))

    def tp_price(lo: float = 20, hi: float = 100) -> xr.DataArray:
        return tprofile(float(rng.uniform(lo, hi))) * escalation  # (time, period)

    def availability(floor: float = 0.05) -> xr.DataArray:
        return xr.DataArray(
            np.clip(0.4 + seasonal + 0.02 * rng.standard_normal(n_timesteps), floor, 1.0), coords=(time_index,)
        )

    def tp_coeff() -> xr.DataArray:
        return xr.DataArray(np.clip(tprofile(float(rng.uniform(1.5, 4.0)), 0.4).values, 1.2, 6.0), coords=(time_index,))

    demand = xr.DataArray(
        np.clip(seasonal + daily + 0.03 * rng.standard_normal(n_timesteps), 0.05, 1.0), coords=(time_index,)
    ) * xr.DataArray([0.985**i for i in range(n_periods)], coords=(periods,))
    peak = float(rng.uniform(400, 600))
    level_cap = xr.DataArray(np.clip(0.6 + 0.4 * seasonal, 0.3, 1.0), coords=(time_index,))

    def window(y0: int, y1: int) -> xr.DataArray:
        return xr.DataArray([1.0 if y0 <= y <= y1 else 0.0 for y in years], coords=(periods,))

    def lump_effects(y0: int) -> dict[str, xr.DataArray]:
        """Per-period lump arrays on a sizing: annualized + one-shot accounting pair,
        capacity credit and a small footprint term."""
        cost = float(rng.uniform(3e4, 3e6))
        funding = float(rng.uniform(0.2, 0.45))
        amort = int(rng.integers(10, 25))
        rate = 0.04
        af = rate * (1 + rate) ** amort / ((1 + rate) ** amort - 1)
        ann = window(y0, y0 + amort - 1) * cost * af
        one = xr.DataArray([cost if y == y0 else 0.0 for y in years], coords=(periods,))
        return {
            'leaf_ba': ann,
            'leaf_ga': ann * funding,
            'leaf_fi': ann * 0.08,
            'leaf_bt': one,
            'leaf_gt': one * funding,
            'cap_min': window(y0, y0 + amort + 4),
            'leaf_l': window(y0, y0 + amort + 4) * 0.02,
        }

    def cap_charge_input(carrier: str, node: str | None, sid: str, extra: dict[str, Any] | None = None) -> Flow:
        """Input flow with a (time, period) price and an optional capacity-charge sizing."""
        effects = {'leaf_f' if carrier in ('k2', 'k3') else 'leaf_p': tp_price(), **(extra or {})}
        return Flow(
            short_id=sid,
            carrier=carrier,
            node=node,
            effects_per_flow_hour=effects,
            size=Sizing(
                size_min=0,
                size_max=1000,
                mandatory=False,
                effects_per_size={'leaf_om': float(rng.uniform(2e4, 1e5)) * escalation},
            ),
        )

    # -- carriers: 12 balance rows ------------------------------------------------------------
    sites = ['n1', 'n2', 'n3']
    carriers = [
        Carrier(id='k0', unit='MW'),
        Carrier(id='k1', nodes=['m0', 'm1'], unit='MW'),
        Carrier(id='k2', nodes=['m0', *sites], unit='MW'),
        Carrier(id='k3', nodes=['m0', *sites], unit='MW'),
        Carrier(id='k4', unit='MW'),
        Carrier(id='k5', unit='MW'),
        Carrier(id='k6', unit='MW'),
        Carrier(id='k7', unit='MW'),
    ]

    # -- 30 effects: weighted chains, one negative cross factor, bounds, budgets ---------------
    annual_demand = float(demand.isel(period=0).sum('time')) * peak
    effects = [
        Effect(
            id='cost',
            unit='u',
            period_weights=npv,
            contribution_from={'agg_fix': 1, 'agg_op': 1, 'agg_env': 0.0625},
        ),
        Effect(
            id='agg_op',
            unit='u',
            contribution_from={'leaf_f': 1, 'leaf_p': 1, 'leaf_m': 1, 'leaf_r': -1, 'leaf_s': -1},
        ),
        Effect(id='leaf_f', unit='u'),
        Effect(id='leaf_p', unit='u'),
        Effect(id='leaf_m', unit='u'),
        Effect(id='leaf_r', unit='u'),
        Effect(id='leaf_s', unit='u'),
        Effect(
            id='agg_fix',
            unit='u',
            contribution_from={'leaf_ba': 1, 'leaf_om': 1, 'leaf_fi': 1, 'leaf_ga': -1},
        ),
        Effect(id='leaf_ba', unit='u'),
        Effect(id='leaf_om', unit='u'),
        Effect(id='leaf_fi', unit='u'),
        Effect(id='leaf_ga', unit='u'),
        Effect(
            id='agg_cap',
            unit='u',
            contribution_from={'leaf_bt': 1, 'leaf_gt': -1, 'agg_net': 1},
        ),
        Effect(id='leaf_bt', unit='u'),
        Effect(id='leaf_gt', unit='u'),
        Effect(id='agg_net', unit='u', contribution_from={'leaf_nc': 1, 'leaf_ng': -1}),
        Effect(id='leaf_nc', unit='u'),
        Effect(id='leaf_ng', unit='u'),
        Effect(
            id='agg_env',
            unit='u',
            contribution_from={'net_x': 1.0, 'leaf_w': 0.25, 'leaf_l': 0.125},
        ),
        Effect(id='leaf_xs', unit='u'),
        Effect(
            id='net_x',
            unit='u',
            periodic_max=xr.DataArray([999_999.0] * n_periods, coords=(periods,)),
            contribution_from={'leaf_x': 1, 'leaf_xs': -1},
        ),
        Effect(id='leaf_x', unit='u'),
        Effect(id='leaf_w', unit='u'),
        Effect(id='leaf_l', unit='u'),
        Effect(
            id='cap_min',
            unit='MW',
            periodic_min=xr.DataArray([peak * 1.05 * 0.985**i for i in range(n_periods)], coords=(periods,)),
        ),
        Effect(
            id='share_min',
            unit='MWh',
            periodic_min=xr.DataArray(
                [annual_demand * min(0.6, 0.05 + 0.04 * i) * 0.1 for i in range(n_periods)], coords=(periods,)
            ),
        ),
        Effect(id='zone_max', unit='MW', periodic_max=8.0),
        Effect(
            id='quota_a',
            unit='h',
            periodic_max=xr.DataArray([3000.0] * n_periods, coords=(periods,)),
            total_max=20_000.0,
            period_weights=[1.0] * n_periods,
        ),
        Effect(
            id='quota_b',
            unit='h',
            periodic_max=xr.DataArray([3000.0] * n_periods, coords=(periods,)),
            total_max=10_000.0,
            period_weights=[1.0] * n_periods,
        ),
        Effect(id='pair_limit', unit='', periodic_min=-0.15, periodic_max=0.15),
    ]

    # -- boundary ports --------------------------------------------------------------------
    ports = [
        Port(
            id='sink_0',
            exports=[Flow(short_id='load', carrier='k0', size=peak, fixed_relative_profile=demand)],
        ),
        Port(id='src_k2', imports=[Flow(short_id='buy', carrier='k2', node='m0', size=12_000)]),
        Port(
            id='src_k3',
            imports=[
                Flow(
                    short_id='buy',
                    carrier='k3',
                    node='m0',
                    size=Sizing(size_min=4000, size_max=4000, mandatory=False),
                )
            ],
        ),
        Port(
            id='hub_k1',
            imports=[Flow(short_id='buy', carrier='k1', node='m0', size=6000)],
            exports=[
                Flow(
                    short_id='sell',
                    carrier='k1',
                    node='m0',
                    size=6000,
                    effects_per_flow_hour={'leaf_r': tp_price()},
                )
            ],
        ),
        Port(id='src_k7', imports=[Flow(short_id='buy', carrier='k7', size=6000)]),
    ]

    converters: list[Converter] = []
    storages: list[Storage] = []

    # -- node bridges: 2-in/2-out with optional fixed-size sizing ------------------------------
    converters.extend(
        Converter(
            id=f'bridge_{site}',
            inputs=[
                Flow(short_id='a_in', carrier='k2', node='m0'),
                Flow(short_id='b_in', carrier='k3', node='m0'),
            ],
            outputs=[
                Flow(
                    short_id='a_out',
                    carrier='k2',
                    node=site,
                    size=Sizing(size_min=8_000, size_max=8_000, mandatory=False),
                ),
                Flow(
                    short_id='b_out',
                    carrier='k3',
                    node=site,
                    size=Sizing(size_min=8_000, size_max=8_000, mandatory=False),
                ),
            ],
            conversion_factors=[{'a_in': 1, 'a_out': -1}, {'b_in': 1, 'b_out': -1}],
        )
        for site in sites
    )

    # -- fixed 1-in/1-out fleet (mandatory sizing, scalar coefficient) --------------------------
    for i in range(5):
        eff = float(rng.uniform(0.75, 0.95))
        converters.append(
            Converter(
                id=f'conv_a{i}',
                inputs=[cap_charge_input('k2', sites[i % len(sites)], 'fuel', extra={'leaf_x': 0.2})],
                outputs=[
                    Flow(
                        short_id='out',
                        carrier='k0',
                        size=Sizing(
                            size_min=0,
                            size_max=float(rng.uniform(80, 260)),
                            mandatory=True,
                            effects_per_size={
                                'leaf_om': float(rng.uniform(3e3, 3e4)) * escalation,
                                **lump_effects(years[0]),
                            },
                        ),
                    )
                ],
                conversion_factors=[{'fuel': eff, 'out': -1}],
            )
        )

    # -- two 2-equation units with a split output feeding a lifetime quota ----------------------
    for name, budget, site in (('conv_b0', 'quota_a', 'n1'), ('conv_b1', 'quota_b', 'n2')):
        converters.append(
            Converter(
                id=name,
                conversion_factors=[
                    {'fuel': 0.5, 'out': -1},
                    {'fuel': 0.4, 'aux': -1, 'aux_q': -1},
                ],
                inputs=[cap_charge_input('k2', site, 'fuel', extra={'leaf_x': 0.4})],
                outputs=[
                    Flow(
                        short_id='out',
                        carrier='k0',
                        size=Sizing(
                            size_min=0,
                            size_max=float(rng.uniform(60, 120)),
                            mandatory=True,
                            effects_per_size={
                                'leaf_om': float(rng.uniform(1e4, 5e4)) * escalation,
                                **lump_effects(years[0]),
                            },
                        ),
                    ),
                    Flow(
                        short_id='aux',
                        carrier='k1',
                        node='m0',
                        effects_per_flow_hour={'leaf_r': tp_price(), 'leaf_xs': 0.4},
                    ),
                    Flow(
                        short_id='aux_q',
                        carrier='k1',
                        node='m0',
                        effects_per_flow_hour={
                            'leaf_r': tp_price(),
                            'leaf_s': 30.0 * escalation,
                            budget: 1 / 100,
                            'leaf_xs': 0.4,
                        },
                    ),
                ],
            )
        )

    # -- one 2-eq unit on the second network carrier, optional ----------------------------------
    converters.append(
        Converter(
            id='conv_b2',
            conversion_factors=[{'fuel': 0.5, 'aux': -1}, {'fuel': 0.4, 'out': -1}],
            inputs=[cap_charge_input('k3', 'n1', 'fuel')],
            outputs=[
                Flow(short_id='aux', carrier='k1', node='m0', effects_per_flow_hour={'leaf_r': tp_price()}),
                Flow(
                    short_id='out',
                    carrier='k0',
                    effects_per_flow_hour={'share_min': 1},
                    size=Sizing(
                        size_min=0,
                        size_max=80,
                        mandatory=False,
                        effects_per_size=lump_effects(years[min(4, n_periods - 1)]),
                    ),
                ),
            ],
        )
    )

    # -- one high-ratio unit with a load-factor bound -------------------------------------------
    converters.append(
        Converter(
            id='conv_d',
            inputs=[cap_charge_input('k1', 'm0', 'drive')],
            outputs=[
                Flow(
                    short_id='out',
                    carrier='k0',
                    effects_per_flow_hour={'share_min': 1},
                    load_factor_max=0.9,
                    size=Sizing(
                        size_min=0,
                        size_max=70,
                        mandatory=False,
                        effects_per_size=lump_effects(years[min(6, n_periods - 1)]),
                    ),
                )
            ],
            conversion_factors=[{'drive': 30, 'out': -1}],
        )
    )

    # -- piecewise unit with component status + maintenance availability -------------------------
    avail = np.ones(n_timesteps)
    for block_start in range(0, n_timesteps, max(1, n_timesteps // 5)):
        avail[block_start : block_start + max(1, n_timesteps // 50)] = 0.0
    converters.append(
        Converter(
            id='pw_unit',
            inputs=[Flow(short_id='fuel', carrier='k7', size=Sizing(size_min=0, size_max=50, mandatory=True))],
            outputs=[
                Flow(
                    short_id='aux',
                    carrier='k1',
                    node='m0',
                    size=6.0,
                    effects_per_flow_hour={'leaf_r': tp_price()},
                ),
                Flow(
                    short_id='out',
                    carrier='k0',
                    effects_per_flow_hour={'share_min': 1},
                    size=Sizing(size_min=0, size_max=43, mandatory=True),
                ),
            ],
            conversion=PiecewiseConversion(
                points={'fuel': [50, 50], 'aux': [6.0, 0.5], 'out': [37, 43]},
                status=Status(),
                availability=xr.DataArray(avail, coords=(time_index,)),
            ),
        )
    )

    # -- a paired unit with a status-gated second stage (pair_limit coupling) ---------------------
    converters.append(
        Converter(
            id='pair_unit',
            inputs=[cap_charge_input('k1', 'm0', 'drive'), cap_charge_input('k1', 'm0', 'drive_2')],
            outputs=[
                Flow(
                    short_id='out',
                    carrier='k0',
                    effects_per_flow_hour={'share_min': 1, 'pair_limit': 1},
                    size=Sizing(
                        size_min=0,
                        size_max=8.0,
                        mandatory=True,
                        effects_per_size=lump_effects(years[1]),
                        effects_fixed={'pair_limit': window(years[0], years[-1])},
                    ),
                ),
                Flow(
                    short_id='out_2',
                    carrier='k0',
                    effects_per_flow_hour={'share_min': 1},
                    relative_rate_min=0.02,
                    status=Status(effects_per_running_hour={'pair_limit': 4.0}),
                    size=Sizing(
                        size_min=2.5,
                        size_max=2.5,
                        mandatory=False,
                        effects_fixed={'pair_limit': -window(years[0], years[-1])},
                    ),
                ),
            ],
            conversion_factors=[
                {'drive': 3.0, 'out': -1},
                {'drive_2': tp_coeff(), 'out_2': -1},
            ],
        )
    )

    # -- fixed fleet with time-varying coefficients and availability ------------------------------
    for i in range(7):
        coeff = tp_coeff()
        converters.append(
            Converter(
                id=f'conv_e{i}',
                inputs=[cap_charge_input('k1', 'm0', 'drive', extra={'leaf_s': tp_price(10, 40)})],
                outputs=[
                    Flow(
                        short_id='out',
                        carrier='k0',
                        effects_per_flow_hour={'share_min': 1, 'leaf_m': 4 * (coeff - 1) / coeff * escalation},
                        relative_rate_max=availability(),
                        size=Sizing(
                            size_min=0,
                            size_max=float(rng.uniform(10, 60)),
                            mandatory=(i % 2 == 0),
                            effects_per_size={
                                'leaf_om': float(rng.uniform(2e4, 6e4)) * escalation,
                                'zone_max': 0.5 if i >= 4 else 0.0,
                                **lump_effects(years[min(i, n_periods - 1)]),
                            },
                        ),
                    )
                ],
                conversion_factors=[{'drive': coeff, 'out': -1}],
            )
        )

    # -- per-period build fleets (the bulk of the unit count) --------------------------------------
    coeff_fleet = tp_coeff()
    for pi, y in enumerate(years):
        site = sites[pi % len(sites)]
        if pi >= 2:
            converters.append(
                Converter(
                    id=f'gen_a{y}',
                    conversion_factors=[
                        {'b_in': 0.9, 'out_b': -1},
                        {'a_in': 0.9, 'out_a': -1},
                    ],
                    inputs=[
                        cap_charge_input('k2', site, 'a_in', extra={'leaf_x': 0.2}),
                        cap_charge_input('k3', site, 'b_in'),
                    ],
                    outputs=[
                        Flow(
                            short_id='out_b',
                            carrier='k0',
                            effects_per_flow_hour={'share_min': 1},
                            size=Sizing(
                                size_min=5,
                                size_max=120,
                                mandatory=False,
                                effects_per_size={
                                    'leaf_om': float(rng.uniform(2e3, 5e3)) * escalation,
                                    **lump_effects(y),
                                },
                            ),
                        ),
                        Flow(
                            short_id='out_a',
                            carrier='k0',
                            size=Sizing(size_min=0, size_max=120, mandatory=False),
                        ),
                    ],
                )
            )
            converters.append(
                Converter(
                    id=f'gen_b{y}',
                    inputs=[cap_charge_input('k1', 'm0', 'drive', extra={'leaf_s': tp_price(10, 40)})],
                    outputs=[
                        Flow(
                            short_id='out',
                            carrier='k0',
                            effects_per_flow_hour={'share_min': 1},
                            relative_rate_max=availability(),
                            size=Sizing(
                                size_min=5,
                                size_max=60,
                                mandatory=False,
                                effects_per_size={
                                    'leaf_om': float(rng.uniform(3e4, 6e4)) * escalation,
                                    **lump_effects(y),
                                },
                            ),
                        )
                    ],
                    conversion_factors=[{'drive': coeff_fleet, 'out': -1}],
                )
            )
        if pi >= 1:
            converters.append(
                Converter(
                    id=f'gen_c{y}',
                    inputs=[cap_charge_input('k1', 'm0', 'drive', extra={'leaf_s': tp_price(10, 40)})],
                    outputs=[
                        Flow(
                            short_id='out',
                            carrier='k0',
                            effects_per_flow_hour={'share_min': 1},
                            relative_rate_max=availability(),
                            size=Sizing(
                                size_min=2,
                                size_max=20,
                                mandatory=False,
                                effects_per_size={
                                    'leaf_om': float(rng.uniform(2e4, 5e4)) * escalation,
                                    'zone_max': 1.0,
                                    **lump_effects(y),
                                },
                            ),
                        )
                    ],
                    conversion_factors=[{'drive': coeff_fleet, 'out': -1}],
                )
            )

    # -- storages: two fixed + a per-period optional fleet -------------------------------------------
    def storage_flow(sid: str, cap: Sizing | float) -> Flow:
        effects: dict[str, Any] = {'leaf_p': tp_price(1, 5)}
        if sid == 'discharging':
            effects['leaf_w'] = 0.05
        return Flow(
            short_id=sid,
            carrier='k0',
            size=cap,
            relative_rate_max=level_cap,
            effects_per_flow_hour=effects,
        )

    for i, (cap, rate) in enumerate(((1000, 100), (3000, 150))):
        storages.append(
            Storage(
                id=f'store_fixed_{i}',
                eta_charge=float(rng.uniform(0.95, 0.99)),
                eta_discharge=float(rng.uniform(0.95, 0.99)),
                relative_loss_per_hour=float(rng.uniform(2e-4, 6e-4)),
                relative_level_max=level_cap,
                capacity=Sizing(
                    size_min=0,
                    size_max=float(cap),
                    mandatory=True,
                    effects_per_size=lump_effects(years[min(i, n_periods - 1)]) if i else {},
                ),
                charging=storage_flow('charging', Sizing(size_min=0, size_max=float(rate), mandatory=True)),
                discharging=storage_flow('discharging', Sizing(size_min=0, size_max=float(rate * 1.1), mandatory=True)),
            )
        )
    for pi, y in enumerate(years):
        if pi < 4:
            continue
        storages.append(
            Storage(
                id=f'store_{y}',
                eta_charge=0.99,
                eta_discharge=0.99,
                relative_loss_per_hour=0.00025,
                relative_level_max=level_cap,
                capacity=Sizing(
                    size_min=600,
                    size_max=40_000,
                    mandatory=False,
                    effects_per_size=lump_effects(y),
                    effects_fixed={'leaf_om': float(rng.uniform(2e5, 5e5)) * escalation * window(y, y + 29)},
                ),
                charging=storage_flow('charging', Sizing(size_min=25, size_max=800, mandatory=False)),
                discharging=storage_flow('discharging', Sizing(size_min=25, size_max=800, mandatory=False)),
            )
        )

    return {
        'timesteps': time_index,
        'carriers': carriers,
        'effects': effects,
        'ports': ports,
        'converters': converters,
        'storages': storages,
        'periods': years,
        'period_weights': period_weights,
    }


SYSTEMS: dict[str, Callable[[int], Elements]] = {
    'district_heating': district_heating,
    'industry_park': industry_park,
    'green_city': green_city,
    'energy_transition': energy_transition,
    'stress': stress,
}


def _count_time_series(value: Any, n_time: int) -> int:
    """Number of time-varying data arrays inside one element parameter value.

    Counts every array-valued leaf whose leading dimension is the time axis —
    xarray objects by their ``time`` dim, plain arrays/lists/pandas objects by
    length. Scalars, breakpoint lists and per-period arrays don't count.
    """
    if isinstance(value, xr.DataArray):
        return int('time' in value.dims)
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return int(len(value) == n_time)
    if isinstance(value, np.ndarray):
        return int(bool(value.shape) and value.shape[0] == n_time)
    if isinstance(value, list):
        if value and all(isinstance(v, (int, float)) for v in value):
            return int(len(value) == n_time)
        return sum(_count_time_series(v, n_time) for v in value)
    if isinstance(value, dict):
        return sum(_count_time_series(v, n_time) for v in value.values())
    if isinstance(value, BaseModel):
        return sum(_count_time_series(getattr(value, name), n_time) for name in type(value).model_fields)
    return 0


def _system_stats(elements: Elements) -> dict[str, Any]:
    """Element-layer size labels for one system.

    Stable properties of the system definition (component, flow, effect and
    time-series counts, temporal grid) — unlike variable/constraint counts,
    which are formulation output and belong to the measured row.
    """
    ports: list[Port] = elements['ports']
    converters: list[Converter] = elements.get('converters') or []
    storages: list[Storage] = elements.get('storages') or []
    flows = (
        sum(len(p.imports) + len(p.exports) for p in ports)
        + sum(len(c.inputs) + len(c.outputs) for c in converters)
        + 2 * len(storages)
    )
    n_time = len(elements['timesteps'])
    groups = (elements['carriers'], elements['effects'], ports, converters, storages)
    periods = elements.get('periods')
    return {
        'time': n_time,
        'periods': len(periods) if periods else 1,
        'components': len(ports) + len(converters) + len(storages),
        'flows': flows,
        'effects': len(elements['effects']),
        'series': sum(_count_time_series(element, n_time) for group in groups for element in group),
    }


def measure(model: str, timesteps: int = HOURS_PER_YEAR, solve: bool = False) -> dict[str, Any]:
    """Build one reference system and return its size labels, stage timings and peak memory.

    The row mixes two kinds of size: element-layer stats from
    :func:`_system_stats` (stable labels of the system definition) and the
    measured solver-model size (``variables``, ``nonzeros``, ``constraints``),
    which changes with the formulation and is re-measured every run.
    """
    builder = SYSTEMS[model]
    start = perf_counter()
    elements = builder(timesteps)
    elements_s = perf_counter() - start
    stats = _system_stats(elements)
    start = perf_counter()
    data = ModelData.build(**elements)
    data_s = perf_counter() - start
    import lpspec

    from fluxopt.relational import MATH_PROGRAM, build_sources
    from fluxopt.relational.results import objective_weights

    weights = objective_weights(data, 'cost')
    sources, coords = build_sources(data, weights)
    start = perf_counter()
    bound = lpspec.build(MATH_PROGRAM, {**sources, **coords})
    build_s = perf_counter() - start
    # Binaries are not a field the engine reports — it counts columns, and
    # integrality is a property of each rather than a second total.
    diagnostics = bound.diagnostics()
    row: dict[str, Any] = {
        'model': model,
        'timesteps': timesteps,
        **stats,
        'variables': diagnostics.columns,
        'nonzeros': diagnostics.nonzeros,
        'constraints': diagnostics.rows,
        'elements_s': elements_s,
        'data_s': data_s,
        'build_s': build_s,
    }
    if solve:
        start = perf_counter()
        bound.solve()
        row['solve_s'] = perf_counter() - start
    bound.close()
    row['peak_mib'] = _peak_rss_mib()
    return row


def _peak_rss_mib() -> float | None:
    """Peak resident memory of this process in MiB (None where unsupported, e.g. Windows).

    Whole-process, OS-level high-water: catches every allocation (numpy, solver
    C libraries, ...) but includes the interpreter + import footprint and
    allocator slack — the number that has to fit in RAM, not the build's own
    appetite.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == 'darwin' else 1024
    return peak * scale / 2**20


def _measure_in_subprocess(model: str, timesteps: int, solve: bool) -> dict[str, Any]:
    """Measure one system in a fresh interpreter so peak memory is attributed per model."""
    cmd = [sys.executable, '-m', 'fluxopt.benchmark', '--worker', model, '--timesteps', str(timesteps)]
    if solve:
        cmd.append('--solve')
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f'exit code {proc.returncode}'
        return {'model': model, 'error': detail}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f'{n / 1e6:.2f}M'
    if n >= 10_000:
        return f'{n / 1e3:.0f}k'
    return str(n)


def _fmt_seconds(s: float) -> str:
    return f'{s * 1000:.0f} ms' if s < 1.0 else f'{s:.1f} s'


def _fmt_mem(mib: float | None) -> str:
    if mib is None:
        return 'n/a'
    return f'{mib / 1024:.1f} GiB' if mib >= 1024 else f'{mib:.0f} MiB'


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Plain-text table; first column left-aligned, the rest right-aligned."""
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = []
    for cells in [headers, *rows]:
        first = cells[0].ljust(widths[0])
        rest = (cell.rjust(width) for cell, width in zip(cells[1:], widths[1:], strict=True))
        lines.append('  '.join([first, *rest]).rstrip())
    lines.insert(1, '-' * len(lines[0]))
    return '\n'.join(lines)


def _print_report(rows: list[dict[str, Any]], timesteps: int, solve: bool) -> None:
    print(f'fluxopt {version("fluxopt")} — build-pipeline benchmark')
    print(f'Python {platform.python_version()} · {platform.system()} {platform.machine()} · {os.cpu_count()} CPUs')
    print(f'{timesteps} hourly timesteps ({timesteps / HOURS_PER_YEAR:.1f} years)')
    print()
    headers = [
        'model',
        'grid',
        'comps',
        'flows',
        'effects',
        'series',
        'variables',
        'binary',
        'constraints',
        'elements',
        'data',
        'build',
        *(['solve'] if solve else []),
        'peak rss',
    ]
    table_rows = [
        [
            row['model'],
            f'{row["time"]}x{row["periods"]}',
            str(row['components']),
            str(row['flows']),
            str(row['effects']),
            str(row['series']),
            _fmt_count(row['variables']),
            _fmt_count(row['nonzeros']),
            _fmt_count(row['constraints']),
            _fmt_seconds(row['elements_s']),
            _fmt_seconds(row['data_s']),
            _fmt_seconds(row['build_s']),
            *([_fmt_seconds(row['solve_s'])] if solve else []),
            _fmt_mem(row['peak_mib']),
        ]
        for row in rows
        if 'error' not in row
    ]
    if table_rows:
        print(_render_table(headers, table_rows))
    for row in (r for r in rows if 'error' in r):
        print(f'{row["model"]}: FAILED — {row["error"]}')


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m fluxopt.benchmark',
        description='Build a few realistic reference energy systems and report speed and memory.',
    )
    parser.add_argument(
        'models',
        nargs='*',
        choices=sorted(SYSTEMS),
        metavar='model',
        help=f'reference systems to run (default: all — {", ".join(SYSTEMS)})',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=HOURS_PER_YEAR,
        help='number of hourly timesteps (default: 8760, one year)',
    )
    parser.add_argument('--solve', action='store_true', help='also solve each model with HiGHS and time it')
    parser.add_argument('--json', action='store_true', help='print JSON instead of the table')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    if args.worker:
        json.dump(measure(args.worker, args.timesteps, args.solve), sys.stdout)
        return 0
    models = args.models or list(SYSTEMS)
    rows = []
    for name in models:
        print(f'building {name} ({args.timesteps} timesteps) ...', file=sys.stderr, flush=True)
        rows.append(_measure_in_subprocess(name, args.timesteps, args.solve))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_report(rows, args.timesteps, args.solve)
    return 1 if any('error' in row for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
