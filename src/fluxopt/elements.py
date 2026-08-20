from __future__ import annotations

from typing import Any, Literal, NamedTuple, override

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fluxopt.types import PiecewiseMethod, Variate

# Element models hold arbitrary xarray/numpy/pandas values (Variate);
# pydantic validates ids/scalars/structure while passing those through by
# isinstance.
_PYDANTIC_CFG = ConfigDict(arbitrary_types_allowed=True)


class Element(BaseModel):
    """Base for user-facing element models (shared pydantic config)."""

    model_config = _PYDANTIC_CFG


PENALTY_EFFECT_ID = 'penalty'

NODE_SEP = ':'
QUAL_FMT = '{component}({flow})'


def qualified_id(component: str, flow: str) -> str:
    """Format a qualified flow id: ``component(flow)``."""
    return QUAL_FMT.format(component=component, flow=flow)


def node_id(carrier: str, node: str) -> str:
    """Format a carrier-node id: ``carrier:node``."""
    return f'{carrier}{NODE_SEP}{node}'


class _BoundFlow(NamedTuple):
    """A flow bound to its place in a component (internal build record).

    Produced by the ``_qualified_flows()`` method of Port, Converter, and
    Storage — the single place where component-local ``short_id``s become
    system-global qualified ids. The flow declaration itself is never
    modified.
    """

    id: str
    """Qualified flow id ``component(short_id)``."""
    flow: Flow
    """The flow declaration (unmodified)."""
    sign: Literal[1, -1]
    """Carrier-balance coefficient of the placement: +1 produces into the
    carrier (imports, converter outputs, discharging), -1 consumes from it
    (exports, converter inputs, charging)."""


class Carrier(Element):
    """Physical energy medium (electricity, heat, gas, …)."""

    id: str
    """Unique identifier used as xarray coordinate."""
    nodes: list[str] = Field(default_factory=list)
    """Sub-nodes for multi-node balancing. Empty means single-node."""
    unit: str = 'MWh'
    """Energy unit label."""
    color: str | None = None
    """Optional color for plotting."""
    description: str = ''
    """Human-readable description."""


class Sizing(Element):
    """Capacity optimization parameters.

    The solver decides the optimal size within [size_min, size_max].

    - ``mandatory=True``: continuous, size in [min, max], no binary.
    - ``mandatory=False``: binary indicator gates size: 0 or [min, max].
    - ``size_min == size_max`` with ``mandatory=False``: binary invest
      at exact size (yes/no).

    See: docs/math/sizing.md
    """

    size_min: float = Field(ge=0)
    """Minimum capacity if invested."""
    size_max: float = Field(ge=0)
    """Maximum capacity."""
    mandatory: bool = True
    """If True, must be built (no binary indicator)."""
    effects_per_size: dict[str, Variate] = Field(default_factory=dict)
    """Effect cost per unit size (e.g. €/MW)."""
    effects_fixed: dict[str, Variate] = Field(default_factory=dict)
    """Fixed effect cost if built (optional only)."""

    @model_validator(mode='after')
    def _bounds_are_ordered(self) -> Sizing:
        """An empty range is a typo, not a model."""
        if self.size_max < self.size_min:
            msg = f'size_max ({self.size_max}) is below size_min ({self.size_min})'
            raise ValueError(msg)
        return self


class Investment(Element):
    """Singular discrete build-timing optimization.

    The solver decides WHEN to build (which period) and at what size.
    Once built, capacity is available for ``lifetime`` periods.
    Size is decided once — no growth or partial retirement.
    """

    size_min: float = Field(ge=0)
    """Minimum capacity if built."""
    size_max: float = Field(ge=0)
    """Maximum capacity."""
    mandatory: bool = True
    """If True, must build exactly once; if False, may build at most once."""
    lifetime: int | None = Field(default=None, gt=0)
    """Periods active after build; None = forever."""
    prior_size: float = Field(default=0.0, ge=0)
    """Pre-existing capacity available from period 0."""
    effects_per_size_at_build: dict[str, Variate] = Field(default_factory=dict)
    """One-time per-MW costs charged in the build period."""
    effects_fixed_at_build: dict[str, Variate] = Field(default_factory=dict)
    """One-time fixed costs charged in the build period."""
    effects_per_size_recurring: dict[str, Variate] = Field(default_factory=dict)
    """Recurring per-MW costs charged every active period."""
    effects_fixed_recurring: dict[str, Variate] = Field(default_factory=dict)
    """Recurring fixed costs charged every active period."""

    @model_validator(mode='after')
    def _bounds_are_ordered(self) -> Investment:
        """An empty range is a typo, not a model."""
        if self.size_max < self.size_min:
            msg = f'size_max ({self.size_max}) is below size_min ({self.size_min})'
            raise ValueError(msg)
        return self


class Status(Element):
    """Binary on/off behavior parameters.

    Together with relative bounds, gives semi-continuous behavior:
    ``{0} U [min, max] * size``.

    See: docs/math/status.md
    """

    uptime_min: float | None = Field(default=None, ge=0)  # [h]
    """Minimum consecutive on-hours."""
    uptime_max: float | None = Field(default=None, ge=0)  # [h]
    """Maximum consecutive on-hours."""
    downtime_min: float | None = Field(default=None, ge=0)  # [h]
    """Minimum consecutive off-hours."""
    downtime_max: float | None = Field(default=None, ge=0)  # [h]
    """Maximum consecutive off-hours."""
    effects_per_running_hour: dict[str, Variate] = Field(default_factory=dict)
    """Effect cost per running hour."""
    effects_per_startup: dict[str, Variate] = Field(default_factory=dict)
    """Effect cost per startup event."""

    @model_validator(mode='after')
    def _durations_are_ordered(self) -> Status:
        """An empty window is a typo, not a model."""
        for lo, hi, what in (
            (self.uptime_min, self.uptime_max, 'uptime'),
            (self.downtime_min, self.downtime_max, 'downtime'),
        ):
            if lo is not None and hi is not None and hi < lo:
                msg = f'{what}_max ({hi}) is below {what}_min ({lo})'
                raise ValueError(msg)
        return self


class Flow(Element):
    """A single energy flow on a carrier.

    ``short_id`` defaults to ``carrier`` (or ``carrier:node`` when ``node``
    is set).  Set explicitly to disambiguate multiple flows on the same
    carrier::

        Flow(carrier='elec')  # short_id='elec'
        Flow(carrier='heat', node='A')  # short_id='heat:A'
        Flow(carrier='elec', short_id='base')  # short_id='base'

    ``short_id`` must be unique within a component.  The system-global
    qualified id ``component(short_id)`` is derived at build time and
    appears as the ``flow`` coordinate in ``ModelData`` and results; the
    flow itself is never modified. Storage resolves colliding short_ids
    to ``charge`` / ``discharge`` at that point.

    See: docs/math/flows.md
    """

    carrier: str
    """Carrier this flow connects to."""
    short_id: str = ''
    """Component-local identifier; defaults to ``carrier``
    (or ``carrier:node``). The qualified form ``component(short_id)``
    is derived at build time.
    """
    node: str | None = None
    """Sub-node for multi-node carrier balancing."""
    size: float | Sizing | Investment | None = None  # P̄_f  [MW]
    """Nominal capacity [MW], ``Sizing`` for investment optimization,
    or None (unsized / unbounded).
    """
    relative_rate_min: Variate = 0.0  # p̲_f  [-]
    """Lower bound as fraction of size."""
    relative_rate_max: Variate = 1.0  # p̄_f  [-]
    """Upper bound as fraction of size."""
    fixed_relative_profile: Variate | None = None  # π_f  [-]
    """Fixed profile as fraction of size; sets both
    lower and upper bounds equal to the profile value.
    """
    effects_per_flow_hour: dict[str, Variate] = Field(default_factory=dict)  # c_{f,k}  [varies]
    """Effect coefficients per flow-hour
    (e.g. €/MWh).
    """
    flow_hours_min: float | None = None  # H̲_f  [MWh] — per period
    """Lower bound on total flow-hours ``Σ P·Δt`` [MWh],
    applied to each period independently.
    """
    flow_hours_max: float | None = None  # H̄_f  [MWh] — per period
    """Upper bound on total flow-hours ``Σ P·Δt`` [MWh],
    applied to each period independently.
    """
    load_factor_min: float | None = None  # λ̲_f  [-] — per period
    """Lower bound on utilization ``Σ P·Δt / (size·T)``,
    applied to each period independently. Requires ``size``.
    """
    load_factor_max: float | None = None  # λ̄_f  [-] — per period
    """Upper bound on utilization ``Σ P·Δt / (size·T)``,
    applied to each period independently. Requires ``size``.
    """
    ramp_up_per_hour: Variate | None = None  # r⁺_f  [1/h] — fraction of size per hour
    """Maximum rate increase between consecutive
    timesteps, as fraction of size per hour. Requires ``size``.
    """
    ramp_down_per_hour: Variate | None = None  # r⁻_f  [1/h] — fraction of size per hour
    """Maximum rate decrease between consecutive
    timesteps, as fraction of size per hour. Requires ``size``.
    """
    status: Status | None = None
    """On/off behavior (semi-continuous, startup costs, durations)."""
    prior_rates: list[float] | None = None  # flow rates before horizon [MW]
    """Flow rates [MW] before the horizon, used for
    status initial conditions.
    """

    # Identity semantics — array-valued fields make value equality ill-defined,
    # and set/dict usage needs hashability.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    @override
    def model_post_init(self, __context: Any) -> None:
        """Default short_id from carrier/node and validate field combinations."""
        if not self.short_id:
            self.short_id = node_id(self.carrier, self.node) if self.node else self.carrier
        if self.status is not None and isinstance(self.relative_rate_min, (int, float)) and self.relative_rate_min <= 0:
            msg = (
                f'Flow {self.short_id!r}: relative_rate_min must be > 0 when status is set, '
                f'otherwise on/off is indistinguishable (got {self.relative_rate_min})'
            )
            raise ValueError(msg)
        if (
            self.status is not None
            and self.fixed_relative_profile is not None
            and isinstance(self.size, (Sizing, Investment))
        ):
            msg = (
                f'Flow {self.short_id!r}: fixed_relative_profile is not supported together with '
                f'Status and Sizing/Investment — fix the size or drop the profile'
            )
            raise ValueError(msg)
        if (self.load_factor_min is not None or self.load_factor_max is not None) and self.size is None:
            msg = (
                f'Flow {self.short_id!r}: load_factor bounds require a size '
                f'(fixed or Sizing/Investment) — the load factor is relative to size'
            )
            raise ValueError(msg)
        if (self.ramp_up_per_hour is not None or self.ramp_down_per_hour is not None) and self.size is None:
            msg = (
                f'Flow {self.short_id!r}: ramp limits require a size '
                f'(fixed or Sizing/Investment) — ramps are relative to size'
            )
            raise ValueError(msg)


class Effect(Element):
    """A tracked quantity across the optimization horizon (cost, CO₂, …).

    One effect is designated as the objective to minimize via the
    ``objective`` argument of ``optimize()``. Others can be bounded
    to enforce budgets (e.g. emission caps).

    Effects accumulate contributions from two domains:

    - **Temporal** — per-timestep flow costs, running costs, startup costs.
    - **Lump** — one-time sizing costs and fixed costs.

    Cross-effect chains (e.g. CO₂ → cost) are supported via
    ``contribution_from``.

    See: docs/math/effects.md
    """

    id: str
    """Unique identifier."""
    unit: str = ''
    """Unit label (e.g. ``'€'``, ``'kg'``)."""
    total_max: float | None = None  # Φ̄_k  [unit] — weighted total across all periods
    """Upper bound on weighted total across all periods."""
    total_min: float | None = None  # Φ̲_k  [unit] — weighted total across all periods
    """Lower bound on weighted total across all periods."""
    periodic_max: Variate | None = None  # Φ̄_{k,p}  [unit] — each period independently
    """Upper bound applied to each period independently.
    Scalar or per-period values (multi-period only).
    """
    periodic_min: Variate | None = None  # Φ̲_{k,p}  [unit] — each period independently
    """Lower bound applied to each period independently.
    Scalar or per-period values (multi-period only).
    """
    contribution_from: dict[str, Variate] = Field(default_factory=dict)
    """Cross-effect factors ``{source_effect: factor}``.
    Scalar factors apply identically to both domains. Time-varying factors
    apply per-timestep in the temporal domain and are rejected at build time
    when the source effect carries lump (sizing/fixed) contributions.
    """
    period_weights: list[float] | None = None  # ω[p] — scales total across periods
    """Per-period weights ω for total aggregation;
    overrides global ``period_weights``.
    """


class Storage(Element):
    """Energy storage with level dynamics.

    Flow ids are qualified as ``storage(flow)`` at build time. When both
    flows share a short_id, the qualified ids use ``charge`` /
    ``discharge`` instead (the flows themselves are never modified)::

        Storage(
            id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='elec')
        )  # bat(charge), bat(discharge)
        Storage(
            id='bat',
            charging=Flow(carrier='elec', short_id='in'),
            discharging=Flow(carrier='elec', short_id='out'),
        )  # bat(in), bat(out)

    Level balance::

        E_{s,t+1} = E_{s,t} (1 - δ)^Δt + P^c η^c Δt - P^d / η^d Δt

    See: docs/math/storage.md
    """

    id: str
    """Storage identifier."""
    charging: Flow
    """Charging flow."""
    discharging: Flow
    """Discharging flow."""
    capacity: float | Sizing | Investment | None = None  # Ē_s  [MWh]
    """Maximum stored energy [MWh], ``Sizing`` for investment
    optimization, or None.
    """
    eta_charge: Variate = 1.0  # η^c_s  [-]
    """Charging efficiency."""
    eta_discharge: Variate = 1.0  # η^d_s  [-]
    """Discharging efficiency."""
    relative_loss_per_hour: Variate = 0.0  # δ_s  [1/h]
    """Self-discharge rate [1/h]."""
    prior_level: float | None = None  # E_{s,0}  [MWh]
    """Initial energy level [MWh]; None = unconstrained."""
    cyclic: bool = True  # E_{s,first} == E_{s,last}
    """If True, end level must equal start level."""
    relative_level_min: Variate = 0.0  # e̲_s  [-]
    """Min SOC as fraction of capacity."""
    relative_level_max: Variate = 1.0  # ē_s  [-]
    """Max SOC as fraction of capacity."""
    final_level_min: float | None = None  # E̲^end_s  [MWh]
    """Lower bound on the level at the last timestep
    [MWh], per period; None = unconstrained.
    """
    final_level_max: float | None = None  # Ē^end_s  [MWh]
    """Upper bound on the level at the last timestep
    [MWh], per period; None = unconstrained.
    """
    prevent_simultaneous: bool = False
    """If True, a binary per timestep excludes
    charging and discharging at once. Requires sized flows
    (fixed or Sizing/Investment) for the big-M bound.
    """
    status: Status | None = None
    """Component-level on/off behavior gating both charging and
    discharging. Forbids flow-level ``status`` on the child flows
    (the two switches would have no defined precedence).
    """

    def _local_short_ids(self) -> tuple[str, str]:
        """Component-local flow names; a colliding pair becomes ``charge`` / ``discharge``."""
        if self.charging.short_id == self.discharging.short_id:
            return 'charge', 'discharge'
        return self.charging.short_id, self.discharging.short_id

    @property
    def _charging_id(self) -> str:
        """Qualified id of the charging flow, derived at build time."""
        return qualified_id(self.id, self._local_short_ids()[0])

    @property
    def _discharging_id(self) -> str:
        """Qualified id of the discharging flow, derived at build time."""
        return qualified_id(self.id, self._local_short_ids()[1])

    def _qualified_flows(self) -> list[_BoundFlow]:
        """Both flows with qualified ids; charging consumes, discharging produces."""
        return [
            _BoundFlow(self._charging_id, self.charging, -1),
            _BoundFlow(self._discharging_id, self.discharging, 1),
        ]

    @override
    def model_post_init(self, __context: Any) -> None:
        """Validate carrier match and status/size requirements."""
        if self.charging.carrier != self.discharging.carrier:
            msg = (
                f'Storage {self.id!r}: charging carrier {self.charging.carrier!r} '
                f'!= discharging carrier {self.discharging.carrier!r}'
            )
            raise ValueError(msg)
        if self.status is not None:
            for f in (self.charging, self.discharging):
                if f.status is not None:
                    msg = (
                        f'Storage {self.id!r}: flow {f.short_id!r} cannot have flow-level '
                        f'status when Storage.status is set; the component status already gates both flows'
                    )
                    raise ValueError(msg)
                if f.size is None:
                    msg = (
                        f'Storage {self.id!r}: flow {f.short_id!r} must have a size when '
                        f'Storage.status is set — without it, the on/off binary cannot gate '
                        f'the rate (no upper bound to scale)'
                    )
                    raise ValueError(msg)
        if self.prevent_simultaneous:
            for f in (self.charging, self.discharging):
                if f.size is None:
                    msg = (
                        f'Storage {self.id!r}: flow {f.short_id!r} must have a size when '
                        f'prevent_simultaneous is set — the exclusion binary needs an '
                        f'upper bound on the rate (big-M)'
                    )
                    raise ValueError(msg)


_CurveTuple = tuple[str, 'list[Variate]'] | tuple[str, 'list[Variate]', Literal['==', '<=', '>=']]


class PiecewiseConversion(Element):
    """Piecewise-linear conversion linking N flows.

    Wraps :func:`linopy.piecewise.add_piecewise_formulation`. All flows
    share interpolation weights — every operating point lies on the same
    piece of the curve.

    Two input forms:

    - **Dict** — equality-only, terse for the common case::

          PiecewiseConversion(points={'fuel': [0, 50, 100], 'Heat': [0, 45, 70]})

    - **List of tuples** — supports per-flow inequality bounds::

          PiecewiseConversion(
              points=[
                  ('fuel', [0, 50, 100]),
                  ('Heat', [0, 45, 70], '>='),
              ]
          )

    See: docs/math/converters.md
    """

    points: dict[str, list[Variate]] | list[_CurveTuple]
    """Per-flow breakpoints. Either ``{flow: [bp...]}`` (equality
    only) or a list of ``(flow, [bp...])`` / ``(flow, [bp...], '<='|'>=')``
    tuples. Need >=2 flows; all breakpoint lists must share the same
    length (>=2). At most one tuple may carry a non-equality bound,
    and only when exactly two flows are present.
    """
    method: PiecewiseMethod = 'auto'
    """Formulation. ``"auto"`` picks LP (2 flows + bounded +
    matching convexity), else incremental (monotonic) or sos2.
    Override with ``"sos2"`` / ``"incremental"`` / ``"lp"``.
    """
    status: Status | None = None
    """Component-level on/off behavior gating the curve."""
    availability: Variate = 1.0
    """Time-varying scaling of the upper breakpoint."""

    @override
    def model_post_init(self, __context: Any) -> None:
        """Validate normalized breakpoints and bound combinations."""
        flows_pts_bounds = list(self._iter_normalized())

        if len(flows_pts_bounds) < 2:
            msg = f'PiecewiseConversion needs >=2 flows, got {len(flows_pts_bounds)}'
            raise ValueError(msg)

        n = len(flows_pts_bounds[0][1])
        if n < 2:
            msg = f'PiecewiseConversion needs >=2 breakpoints per flow, got {n}'
            raise ValueError(msg)

        if any(len(pts) != n for _, pts, _ in flows_pts_bounds):
            lengths = {flow: len(pts) for flow, pts, _ in flows_pts_bounds}
            msg = f'PiecewiseConversion breakpoint lists must all have the same length, got {lengths}'
            raise ValueError(msg)

        flows = [flow for flow, _, _ in flows_pts_bounds]
        if len(set(flows)) != len(flows):
            dupes = [f for f in flows if flows.count(f) > 1]
            msg = f'PiecewiseConversion has duplicate flow ids: {sorted(set(dupes))}'
            raise ValueError(msg)

        nonequal = [b for _, _, b in flows_pts_bounds if b != '==']
        if len(nonequal) > 1:
            msg = f'At most one bounded flow per PiecewiseConversion, got {len(nonequal)}'
            raise ValueError(msg)
        if nonequal and len(flows_pts_bounds) > 2:
            msg = f'Inequality bounds require exactly 2 flows, got {len(flows_pts_bounds)}'
            raise ValueError(msg)
        if self.method == 'lp' and not nonequal:
            msg = "method='lp' requires one flow with bound '<=' or '>='"
            raise ValueError(msg)

    def _iter_normalized(
        self,
    ) -> list[tuple[str, list[Variate], Literal['==', '<=', '>=']]]:
        """Return the curve as a list of ``(flow, breakpoints, bound)`` tuples."""
        if isinstance(self.points, dict):
            return [(flow, list(pts), '==') for flow, pts in self.points.items()]
        result: list[tuple[str, list[Variate], Literal['==', '<=', '>=']]] = []
        for i, t in enumerate(self.points):
            match t:
                case (flow, pts):
                    result.append((flow, list(pts), '=='))
                case (flow, pts, bound):
                    if bound not in ('==', '<=', '>='):
                        msg = f'PiecewiseConversion tuple {i} has invalid bound {bound!r}; expected one of (==, <=, >=)'
                        raise ValueError(msg)
                    result.append((flow, list(pts), bound))
                case _:
                    msg = f'PiecewiseConversion tuple {i} has length {len(t)}; expected 2 or 3'
                    raise ValueError(msg)
        return result
