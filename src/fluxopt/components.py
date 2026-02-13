from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.elements import Flow
    from fluxopt.types import TimeSeries


@dataclass
class Port:
    id: str
    imports: list[Flow] = field(default_factory=list)
    exports: list[Flow] = field(default_factory=list)

    def __post_init__(self) -> None:
        for f in self.imports:
            f.id = f'{self.id}({f.bus})'
            f._is_input = False
        for f in self.exports:
            f.id = f'{self.id}({f.bus})'
            f._is_input = True


@dataclass
class Converter:
    """Linear conversion between input and output flows.

    Conversion equation (per equation index):
        sum_f(a_f * P_{f,t}) = 0   for all t
    """

    id: str
    inputs: list[Flow]
    outputs: list[Flow]
    conversion_factors: list[dict[Flow, TimeSeries]] = field(default_factory=list)  # a_f

    def __post_init__(self) -> None:
        for f in self.inputs:
            f.id = f'{self.id}({f.bus})'
            f._is_input = True
        for f in self.outputs:
            f.id = f'{self.id}({f.bus})'
            f._is_input = False

    @classmethod
    def boiler(cls, id: str, thermal_efficiency: TimeSeries, fuel_flow: Flow, thermal_flow: Flow) -> Converter:
        return cls(
            id,
            inputs=[fuel_flow],
            outputs=[thermal_flow],
            conversion_factors=[{fuel_flow: thermal_efficiency, thermal_flow: -1}],
        )

    @classmethod
    def heat_pump(cls, id: str, cop: TimeSeries, electrical_flow: Flow, thermal_flow: Flow) -> Converter:
        return cls(
            id,
            inputs=[electrical_flow],
            outputs=[thermal_flow],
            conversion_factors=[{electrical_flow: cop, thermal_flow: -1}],
        )

    @classmethod
    def chp(
        cls,
        id: str,
        eta_el: TimeSeries,
        eta_th: TimeSeries,
        fuel_flow: Flow,
        electrical_flow: Flow,
        thermal_flow: Flow,
    ) -> Converter:
        return cls(
            id,
            inputs=[fuel_flow],
            outputs=[electrical_flow, thermal_flow],
            conversion_factors=[
                {fuel_flow: eta_el, electrical_flow: -1},
                {fuel_flow: eta_th, thermal_flow: -1},
            ],
        )
