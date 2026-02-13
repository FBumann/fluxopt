from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.types import TimeSeries


@dataclass
class Flow:
    label: str
    bus: str
    size: float | None = None  # P̄_f  [MW]
    relative_minimum: TimeSeries = 0.0  # p̲_f  [-]
    relative_maximum: TimeSeries = 1.0  # p̄_f  [-]
    fixed_relative_profile: TimeSeries | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)  # c_{f,k}  [varies]

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
    maximum_total: float | None = None  # Φ̄_k  [unit]
    minimum_total: float | None = None  # Φ̲_k  [unit]
    maximum_per_hour: TimeSeries | None = None  # Φ̄_{k,t}  [unit]
    minimum_per_hour: TimeSeries | None = None  # Φ̲_{k,t}  [unit]


@dataclass
class Storage:
    """Energy storage with charge dynamics.

    Charge balance:
        E_{s,t+1} = E_{s,t} (1 - δ Δt) + P^c η^c Δt - P^d / η^d Δt
    """

    label: str
    charging: Flow
    discharging: Flow
    capacity: float | None = None  # Ē_s  [MWh]
    eta_charge: TimeSeries = 1.0  # η^c_s  [-]
    eta_discharge: TimeSeries = 1.0  # η^d_s  [-]
    relative_loss_per_hour: TimeSeries = 0.0  # δ_s  [1/h]
    initial_charge_state: float | str | None = 0.0  # E_{s,0}  [MWh]
    relative_minimum_charge_state: TimeSeries = 0.0  # e̲_s  [-]
    relative_maximum_charge_state: TimeSeries = 1.0  # ē_s  [-]

    def __post_init__(self) -> None:
        self.charging._component = self.label
        self.charging._is_input = True  # charging takes energy from the bus
        self.discharging._component = self.label
        self.discharging._is_input = False  # discharging puts energy to the bus
