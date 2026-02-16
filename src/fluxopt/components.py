from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fluxopt.types import IdList

if TYPE_CHECKING:
    from fluxopt.elements import Flow
    from fluxopt.types import TimeSeries


def _qualify_flows(component_id: str, flows: list[Flow]) -> IdList[Flow]:
    """Set qualified id on each flow and return as IdList.

    Args:
        component_id: Parent component id used as prefix.
        flows: Flows to qualify.
    """
    for f in flows:
        f.id = f'{component_id}({f.id})'
    return IdList(flows)


@dataclass
class Port:
    """System boundary that imports from or exports to buses."""

    id: str
    imports: list[Flow] | IdList[Flow] = field(default_factory=list)
    exports: list[Flow] | IdList[Flow] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Qualify flow ids with the port id."""
        self.imports = _qualify_flows(self.id, list(self.imports))
        self.exports = _qualify_flows(self.id, list(self.exports))


@dataclass
class Converter:
    """Linear conversion between input and output flows.

    Conversion equation (per equation index)::

        sum_f(a_f * P_{f,t}) = 0   for all t
    """

    id: str
    inputs: list[Flow] | IdList[Flow]
    outputs: list[Flow] | IdList[Flow]
    conversion_factors: list[dict[Flow, TimeSeries]] = field(default_factory=list)  # a_f

    def __post_init__(self) -> None:
        """Qualify flow ids with the converter id."""
        self.inputs = _qualify_flows(self.id, list(self.inputs))
        self.outputs = _qualify_flows(self.id, list(self.outputs))

    @classmethod
    def boiler(cls, id: str, thermal_efficiency: TimeSeries, fuel_flow: Flow, thermal_flow: Flow) -> Converter:
        """Create a boiler converter: fuel * eta = thermal.

        Args:
            id: Converter id.
            thermal_efficiency: Thermal efficiency eta.
            fuel_flow: Input fuel flow.
            thermal_flow: Output thermal flow.
        """
        return cls(
            id,
            inputs=[fuel_flow],
            outputs=[thermal_flow],
            conversion_factors=[{fuel_flow: thermal_efficiency, thermal_flow: -1}],
        )

    @classmethod
    def heat_pump(cls, id: str, cop: TimeSeries, electrical_flow: Flow, thermal_flow: Flow) -> Converter:
        """Create a heat pump converter: electrical * COP = thermal.

        Args:
            id: Converter id.
            cop: Coefficient of performance.
            electrical_flow: Input electrical flow.
            thermal_flow: Output thermal flow.
        """
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
        """Create a CHP converter with separate electrical and thermal outputs.

        Args:
            id: Converter id.
            eta_el: Electrical efficiency.
            eta_th: Thermal efficiency.
            fuel_flow: Input fuel flow.
            electrical_flow: Output electrical flow.
            thermal_flow: Output thermal flow.
        """
        return cls(
            id,
            inputs=[fuel_flow],
            outputs=[electrical_flow, thermal_flow],
            conversion_factors=[
                {fuel_flow: eta_el, electrical_flow: -1},
                {fuel_flow: eta_th, thermal_flow: -1},
            ],
        )
