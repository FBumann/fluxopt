from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.types import TimeSeries


@dataclass
class Sizing:
    """Capacity optimization parameters.

    The solver decides the optimal size within [min_size, max_size].

    - ``mandatory=True``: continuous, size in [min, max], no binary.
    - ``mandatory=False``: binary indicator gates size: 0 or [min, max].
    - ``min_size == max_size``: binary invest at exact size (yes/no).
    """

    min_size: float
    max_size: float
    mandatory: bool = False
    effects_per_size: dict[str, float] = field(default_factory=dict)
    effects_fixed: dict[str, float] = field(default_factory=dict)


@dataclass(eq=False)
class Flow:
    """A single energy flow on a bus.

    ``id`` is optional: defaults to the bus name. Set explicitly to
    disambiguate multiple flows on the same bus::

        Flow(bus='elec')  # id → 'elec' → boiler(elec)
        Flow(bus='elec', id='base')  # id → 'base' → chp(base)

    After ``__post_init__`` of the parent component, ``id`` is expanded
    to the qualified form ``component(id)``.
    """

    bus: str
    id: str = ''
    size: float | Sizing | None = None  # P̄_f  [MW]
    relative_minimum: TimeSeries = 0.0  # p̲_f  [-]
    relative_maximum: TimeSeries = 1.0  # p̄_f  [-]
    fixed_relative_profile: TimeSeries | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)  # c_{f,k}  [varies]

    def __post_init__(self) -> None:
        """Default id to bus name if not set."""
        if not self.id:
            self.id = self.bus


@dataclass
class Bus:
    id: str
    carrier: str | None = None


@dataclass
class Effect:
    id: str
    unit: str = ''
    is_objective: bool = False
    maximum_total: float | None = None  # Φ̄_k  [unit]
    minimum_total: float | None = None  # Φ̲_k  [unit]
    maximum_per_hour: TimeSeries | None = None  # Φ̄_{k,t}  [unit]
    minimum_per_hour: TimeSeries | None = None  # Φ̲_{k,t}  [unit]


@dataclass
class Storage:
    """Energy storage with charge dynamics.

    Flow ids are qualified as ``storage(flow)``. When both flows connect
    to the same bus, they are renamed to ``charge`` / ``discharge``::

        Storage('bat', Flow(bus='elec'), Flow(bus='elec'))  # bat(charge), bat(discharge)
        Storage('bat', Flow(bus='elec'), Flow(bus='heat'))  # bat(elec), bat(heat)

    Charge balance::

        E_{s,t+1} = E_{s,t} (1 - δ Δt) + P^c η^c Δt - P^d / η^d Δt
    """

    id: str
    charging: Flow
    discharging: Flow
    capacity: float | Sizing | None = None  # Ē_s  [MWh]
    eta_charge: TimeSeries = 1.0  # η^c_s  [-]
    eta_discharge: TimeSeries = 1.0  # η^d_s  [-]
    relative_loss_per_hour: TimeSeries = 0.0  # δ_s  [1/h]
    initial_charge_state: float | str | None = 0.0  # E_{s,0}  [MWh]
    relative_minimum_charge_state: TimeSeries = 0.0  # e̲_s  [-]
    relative_maximum_charge_state: TimeSeries = 1.0  # ē_s  [-]

    def __post_init__(self) -> None:
        """Rename colliding flow ids and qualify with storage id."""
        if self.charging.id == self.discharging.id:
            self.charging.id = 'charge'
            self.discharging.id = 'discharge'
        self.charging.id = f'{self.id}({self.charging.id})'
        self.discharging.id = f'{self.id}({self.discharging.id})'
