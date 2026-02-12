from __future__ import annotations

from dataclasses import dataclass, field

from energysys.types import TimeSeries


@dataclass
class Flow:
    label: str
    bus: str
    size: float | None = None
    relative_minimum: TimeSeries = 0.0
    relative_maximum: TimeSeries = 1.0
    fixed_relative_profile: TimeSeries | None = None
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)

    # Set internally by components
    _component: str | None = field(default=None, repr=False, compare=False)
    _is_input: bool | None = field(default=None, repr=False, compare=False)


@dataclass
class Bus:
    label: str
    carrier: str | None = None


@dataclass
class Effect:
    label: str
    unit: str = ''
    is_objective: bool = False
    maximum_total: float | None = None
    minimum_total: float | None = None
    maximum_per_hour: TimeSeries | None = None
    minimum_per_hour: TimeSeries | None = None


@dataclass
class Storage:
    label: str
    charging: Flow
    discharging: Flow
    capacity: float | None = None
    eta_charge: TimeSeries = 1.0
    eta_discharge: TimeSeries = 1.0
    relative_loss_per_hour: TimeSeries = 0.0
    initial_charge_state: float | str | None = 0.0
    relative_minimum_charge_state: TimeSeries = 0.0
    relative_maximum_charge_state: TimeSeries = 1.0

    def __post_init__(self):
        self.charging._component = self.label
        self.charging._is_input = True  # charging takes energy from the bus
        self.discharging._component = self.label
        self.discharging._is_input = False  # discharging puts energy to the bus
