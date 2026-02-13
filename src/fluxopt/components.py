from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.elements import Flow
    from fluxopt.types import TimeSeries


@dataclass
class Source:
    label: str
    outputs: list[Flow]

    def __post_init__(self) -> None:
        for f in self.outputs:
            f._component = self.label
            f._is_input = False


@dataclass
class Sink:
    label: str
    inputs: list[Flow]

    def __post_init__(self) -> None:
        for f in self.inputs:
            f._component = self.label
            f._is_input = True


@dataclass
class LinearConverter:
    """Linear conversion between input and output flows.

    Conversion equation (per equation index):
        sum_f(a_f * P_{f,t}) = 0   for all t
    """

    label: str
    inputs: list[Flow]
    outputs: list[Flow]
    conversion_factors: list[dict[str, TimeSeries]] = field(default_factory=list)  # a_f

    def __post_init__(self) -> None:
        for f in self.inputs:
            f._component = self.label
            f._is_input = True
        for f in self.outputs:
            f._component = self.label
            f._is_input = False

    @classmethod
    def boiler(cls, label: str, thermal_efficiency: TimeSeries, fuel_flow: Flow, thermal_flow: Flow) -> LinearConverter:
        return cls(
            label,
            inputs=[fuel_flow],
            outputs=[thermal_flow],
            conversion_factors=[{fuel_flow.label: thermal_efficiency, thermal_flow.label: -1}],
        )

    @classmethod
    def heat_pump(cls, label: str, cop: TimeSeries, electrical_flow: Flow, thermal_flow: Flow) -> LinearConverter:
        return cls(
            label,
            inputs=[electrical_flow],
            outputs=[thermal_flow],
            conversion_factors=[{electrical_flow.label: cop, thermal_flow.label: -1}],
        )

    @classmethod
    def chp(
        cls,
        label: str,
        eta_el: TimeSeries,
        eta_th: TimeSeries,
        fuel_flow: Flow,
        electrical_flow: Flow,
        thermal_flow: Flow,
    ) -> LinearConverter:
        return cls(
            label,
            inputs=[fuel_flow],
            outputs=[electrical_flow, thermal_flow],
            conversion_factors=[
                {fuel_flow.label: eta_el, electrical_flow.label: -1},
                {fuel_flow.label: eta_th, thermal_flow.label: -1},
            ],
        )
