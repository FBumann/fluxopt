from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.types import TimeSeries


@dataclass
class Sizing:
    """Size optimization parameters for a flow or storage.

    When ``min_size == max_size`` the decision is binary (build at fixed size or not).
    """

    min_size: float
    max_size: float
    mandatory: bool = False
    effects_per_size: dict[str, float] = field(default_factory=dict)
    effects_of_size: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.min_size < 0:
            raise ValueError(f'min_size must be >= 0, got {self.min_size}')
        if self.max_size <= 0:
            raise ValueError(f'max_size must be > 0, got {self.max_size}')
        if self.min_size > self.max_size:
            raise ValueError(f'min_size ({self.min_size}) must be <= max_size ({self.max_size})')


@dataclass(eq=False)
class Flow:
    """A single energy flow on a bus.

    ``id`` is optional: leave empty for the common single-flow-per-bus case and
    the parent component will default it to the bus name.  Set it explicitly to
    disambiguate when a component has multiple flows on the same bus::

        Flow(bus='elec')  # → boiler(elec)
        Flow(bus='elec', id='base')  # → chp(base)

    After the parent component's ``__post_init__`` runs, ``id`` is expanded to
    the qualified form ``{component.id}({id or bus})``.  For storage flows the
    default is ``charge`` / ``discharge`` instead of the bus name.
    """

    bus: str
    id: str = ''
    size: float | Sizing | None = None  # P̄_f  [MW]
    relative_minimum: TimeSeries = 0.0  # p̲_f  [-]
    relative_maximum: TimeSeries = 1.0  # p̄_f  [-]
    fixed_relative_profile: TimeSeries | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)  # c_{f,k}  [varies]

    _is_input: bool = field(default=False, init=False, repr=False)


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

    Charge balance:
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
        self.charging.id = f'{self.id}({self.charging.id or "charge"})'
        self.charging._is_input = True  # charging takes energy from the bus
        self.discharging.id = f'{self.id}({self.discharging.id or "discharge"})'
        self.discharging._is_input = False  # discharging puts energy to the bus
