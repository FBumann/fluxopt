from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linopy import Model, Variable

from fluxopt.constraints.sparse import sparse_weighted_sum
from fluxopt.constraints.status import add_duration_tracking, add_switch_transitions
from fluxopt.constraints.storage import add_accumulation_constraints
from fluxopt.contract import BoundType, Dim, Var
from fluxopt.contributions import _apply_leontief, _leontief
from fluxopt.effect_terms import effect_terms
from fluxopt.results import Result
from fluxopt.types import as_dataarray

if TYPE_CHECKING:
    from collections.abc import Callable

    from fluxopt.effect_terms import EffectTerm
    from fluxopt.model_data import ModelData


def _normalize_objective(value: str | dict[str, float] | None) -> dict[str, float]:
    """Coerce an objective spec into a ``{effect: weight}`` dict.

    A bare effect name becomes ``{name: 1.0}``; ``None`` becomes ``{}``.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return {value: 1.0}
    return {k: float(v) for k, v in value.items()}


def _validate_objective(effects: dict[str, float]) -> None:
    """Require the objective to name at least one non-penalty effect.

    The built-in penalty effect is added automatically as soft-constraint
    steering (see :meth:`FlowSystemModel._set_objective`) and cannot stand in
    for a real objective, so an empty or penalty-only spec is rejected.

    Raises:
        ValueError: If no non-penalty effect is named.
    """
    from fluxopt.elements import PENALTY_EFFECT_ID

    if not any(k != PENALTY_EFFECT_ID for k in effects):
        msg = (
            'objective must name at least one non-penalty effect to minimize '
            '(a name like "cost", or a weight dict like {"cost": 1, "co2": 50}). '
            'The built-in penalty effect is added automatically and cannot be the sole objective.'
        )
        raise ValueError(msg)


def _lump_bearing_effects(terms: list[EffectTerm], cf_lump: xr.DataArray) -> xr.DataArray:
    """Boolean mask over ``effect``: which effects receive lump contributions.

    An effect is lump-bearing when a lump-domain term contributes to it
    directly, or when it receives from a lump-bearing effect through the
    (acyclic) cross-effect matrix ``(effect, source_effect[, ...])``.
    """
    effect_ids = cf_lump.indexes['effect']
    bearing = xr.DataArray(np.zeros(len(effect_ids), dtype=bool), coords={'effect': effect_ids}, dims='effect')
    for term in (t for t in terms if t.domain == 'lump'):
        nonzero = (term.coeff.notnull() & (term.coeff != 0)).any([d for d in term.coeff.dims if d != 'effect'])
        bearing = bearing | nonzero.reindex_like(bearing, fill_value=False)

    adjacency = (cf_lump.notnull() & (cf_lump != 0)).any(
        [d for d in cf_lump.dims if d not in ('effect', 'source_effect')]
    )
    for _ in range(bearing.sizes['effect']):
        grown = bearing | (adjacency & bearing.rename({'effect': 'source_effect'})).any('source_effect')
        if grown.equals(bearing):
            break
        bearing = grown
    return bearing


class FlowSystemModel:
    # Sizing variables — None when no sizing is configured
    flow_size: Variable | None = None
    flow_size_indicator: Variable | None = None
    storage_capacity: Variable | None = None
    storage_capacity_indicator: Variable | None = None

    # Investment variables — None when no Investment is configured
    invest_size: Variable | None = None
    invest_build: Variable | None = None
    invest_active: Variable | None = None
    invest_size_at_build: Variable | None = None

    # Storage variables — None when no storages
    storage_level: Variable | None = None
    prior_storage_level: Variable | None = None

    # Effect / objective — set via optimize() or defaults to ['cost']

    # Status variables — None when no status is configured
    flow_on: Variable | None = None
    flow_startup: Variable | None = None
    flow_shutdown: Variable | None = None

    # Component-level status variables — None when no component status is configured
    component_on: Variable | None = None
    component_startup: Variable | None = None
    component_shutdown: Variable | None = None

    def __init__(self, data: ModelData, objective: str | dict[str, float] | None = None) -> None:
        """Initialize the flow system optimization model.

        Args:
            data: Pre-built model data.
            objective: Effect(s) the objective minimizes. A single effect name,
                or a dict mapping effect names to objective weights
                (``{'cost': 1, 'co2': 50}``). May be deferred (left ``None``)
                and supplied later via the :attr:`objective` property or
                :meth:`optimize` — but a real (non-penalty) objective is
                required by the time :meth:`build` runs. See :meth:`optimize`
                for the full weighting semantics.

        Raises:
            ValueError: If ``objective`` is given but names no non-penalty effect.
        """
        self.data = data
        self.m = Model()
        self._objective: dict[str, float] = _normalize_objective(objective)
        self._objective_weights: dict[str, float] = {}
        self._piecewise: dict[str, Any] = {}  # conv_id -> linopy.PiecewiseFormulation
        self._built = False
        if objective is not None:
            _validate_objective(self._objective)

    @property
    def objective(self) -> dict[str, float]:
        """Effect(s) the objective minimizes, as ``{effect: weight}``.

        Assign a name or a dict to retarget the objective; call :meth:`build`
        again for the change to take effect in the linopy model. Assigning an
        empty objective is rejected — a model must always minimize something.
        """
        return dict(self._objective)

    @objective.setter
    def objective(self, value: str | dict[str, float]) -> None:
        normalized = _normalize_objective(value)
        _validate_objective(normalized)
        self._objective = normalized

    def _add_variables(
        self,
        *,
        lower: Any = None,
        upper: Any = None,
        coords: dict[str, xr.DataArray],
        name: str,
        binary: bool = False,
    ) -> Variable:
        """Add a variable with bounds auto-aligned to coords via as_dataarray.

        Args:
            lower: Lower bound (scalar, array, or DataArray).
            upper: Upper bound (scalar, array, or DataArray).
            coords: Coordinate dict mapping dim names to DataArrays.
            name: Variable name.
            binary: Create a binary variable instead.
        """
        coord_dict = {k: v.values for k, v in coords.items()}
        kwargs: dict[str, Any] = {'coords': coords, 'name': name, 'binary': binary}
        if not binary:
            if lower is not None:
                kwargs['lower'] = as_dataarray(lower, coord_dict)
            if upper is not None:
                kwargs['upper'] = as_dataarray(upper, coord_dict)
        return self.m.add_variables(**kwargs)

    def build(self) -> None:
        """Build all variables, constraints, and the objective.

        Idempotent with respect to retargeting: rebuilding starts from a
        fresh linopy model, so assigning :attr:`objective` and calling
        ``build()`` again is supported.

        Raises:
            ValueError: If no real (non-penalty) objective has been set
                (see :attr:`objective`).
        """
        _validate_objective(self._objective)  # fail fast, before building anything
        if self._built:
            self.m = Model()
            self._piecewise = {}
        # Phase 1: Decision variables
        self._create_flow_variables()
        self._create_sizing_variables()
        self._create_investment_variables()
        self._create_status_variables()
        self._create_component_status_variables()
        # Phase 2: Flow rate constraints
        self._constrain_flow_rates_plain()
        self._constrain_flow_rates_sizing()
        self._constrain_flow_rates_status()
        self._constrain_flow_rates_component_status()
        self._constrain_flow_aggregates()
        self._constrain_flow_ramps()
        # Phase 3: Feature constraints
        self._constrain_sizing()
        self._constrain_investment()
        self._constrain_status()
        self._constrain_component_status()
        # Phase 4: System
        self._create_balance()
        self._create_converter_constraints()
        self._create_piecewise_constraints()
        self._create_storage()
        self._create_effects()
        self._set_objective()
        self._builtin_var_names: frozenset[str] = frozenset(self.m.variables)
        self._built = True

    def optimize(
        self,
        objective: str | dict[str, float] | None = None,
        customize: Callable[[FlowSystemModel], None] | None = None,
        *,
        solver: str = 'highs',
        **kwargs: Any,
    ) -> Result:
        """Build, optionally customize, and solve the model.

        Args:
            objective: Effect(s) to minimize, overriding any objective
                already set on this FlowSystemModel. A single name, or a dict
                mapping effect names to objective weights
                (``{'cost': 1, 'co2': 50}``) — tracked effect totals are
                unaffected by the weighting. The built-in ``'penalty'``
                effect is added at weight 1.0 unless the dict names it:
                ``{'cost': 1, 'penalty': 0}`` opts out, other values scale
                the steering pressure. If None, the current :attr:`objective`
                is used.
            customize: Optional callback to modify the linopy model between build and solve.
                Receives ``self``; use ``model.m`` to add variables/constraints.
            solver: Solver backend name.
            **kwargs: Passed through to ``linopy.Model.solve()``.
        """
        if objective is not None:
            self.objective = objective
        self.build()
        if customize is not None:
            customize(self)
        return self.solve(solver_name=solver, **kwargs)

    def solve(self, **kwargs: Any) -> Result:
        """Solve the built model and return results.

        Thin wrapper around ``linopy.Model.solve()``. Call :meth:`build` first.

        Args:
            **kwargs: Passed through to ``linopy.Model.solve()``.

        Raises:
            RuntimeError: If the model has not been built yet.
        """
        if not self._built:
            msg = 'Model not built — call build() (or optimize()) before solve().'
            raise RuntimeError(msg)
        self.m.solve(**kwargs)
        return Result.from_model(self)

    def _create_flow_variables(self) -> None:
        """Create flow rate decision variables P_{f,t} >= 0."""
        ds = self.data.flows
        self.flow_rate = self.m.add_variables(
            lower=0,
            coords={'flow': ds.rel_lb.coords['flow'], **self.data.dims.coords(time=True)},
            name=Var.FLOW_RATE,
        )

    def _create_sizing_variables(self) -> None:
        """Create sizing decision variables for flows and storages.

        Both Sizing and Investment flows get entries in ``flow_size``.
        Investment adds extra variables (invest_size, invest_build, etc.)
        with constraints linking flow_size to invest_size * active.
        """
        fds = self.data.flows
        pc = self.data.dims.coords(period=True)

        # Collect all flow ids and upper bounds that need a flow_size variable
        all_ids: list[str] = []
        upper_parts: list[xr.DataArray] = []

        # --- Sizing flows ---
        if fds.sizing is not None:
            sizing_ids = list(fds.sizing.min.coords[Dim.SIZING_FLOW].values)
            all_ids.extend(sizing_ids)
            upper_parts.append(fds.sizing.max.rename({Dim.SIZING_FLOW: 'flow'}))

            mandatory = fds.sizing.mandatory
            optional_ids = np.array(sizing_ids)[~mandatory.values]
            if len(optional_ids):
                self.flow_size_indicator = self._add_variables(
                    binary=True,
                    coords={'flow': xr.DataArray(optional_ids, dims=['flow']), **pc},
                    name=Var.FLOW_SIZE_INDICATOR,
                )

        # --- Investment flows ---
        if fds.invest is not None:
            invest_ids = list(fds.invest.min.coords[Dim.INVEST_FLOW].values)
            all_ids.extend(invest_ids)
            upper_parts.append(fds.invest.max.rename({Dim.INVEST_FLOW: 'flow'}))

        # Create unified flow_size variable
        if all_ids:
            upper = xr.concat(upper_parts, dim='flow') if len(upper_parts) > 1 else upper_parts[0]
            flow_coord = xr.DataArray(all_ids, dims=['flow'])
            self.flow_size = self._add_variables(
                lower=0, upper=upper, coords={'flow': flow_coord, **pc}, name=Var.FLOW_SIZE
            )

        # --- Storage capacity sizing ---
        sds = self.data.storages
        if sds is not None and sds.sizing is not None:
            sizing_ids = sds.sizing.min.coords[Dim.SIZING_STORAGE].values
            stor_coord = xr.DataArray(sizing_ids, dims=['storage'])
            upper = sds.sizing.max.rename({Dim.SIZING_STORAGE: 'storage'})
            pc = self.data.dims.coords(period=True)
            self.storage_capacity = self._add_variables(
                lower=0, upper=upper, coords={'storage': stor_coord, **pc}, name=Var.STORAGE_CAPACITY
            )
            mandatory = sds.sizing.mandatory
            optional_ids = sizing_ids[~mandatory.values]
            if len(optional_ids):
                self.storage_capacity_indicator = self._add_variables(
                    binary=True,
                    coords={'storage': xr.DataArray(optional_ids, dims=['storage']), **pc},
                    name=Var.STORAGE_SIZE_INDICATOR,
                )

    def _create_investment_variables(self) -> None:
        """Create investment-specific decision variables.

        Investment flows already have entries in ``flow_size`` (created by
        ``_create_sizing_variables``).  This method adds the extra variables:
        - invest_size[flow]: chosen capacity (period-independent)
        - invest_build[flow, period]: binary, build in this period?
        - invest_active[flow, period]: binary, operational in this period?
        - invest_size_at_build[flow, period]: invest_size * build (big-M linked)
        """
        fds = self.data.flows
        if fds.invest is None:
            return

        invest_ids = list(fds.invest.min.coords[Dim.INVEST_FLOW].values)
        pc = self.data.dims.coords(period=True)

        if not pc:
            msg = 'Investment requires multi-period optimization (periods must be specified)'
            raise ValueError(msg)

        flow_coord = xr.DataArray(invest_ids, dims=['flow'])
        upper = fds.invest.max.rename({Dim.INVEST_FLOW: 'flow'})

        # invest_size[flow]: single capacity decision (no period dim)
        self.invest_size = self._add_variables(lower=0, upper=upper, coords={'flow': flow_coord}, name=Var.INVEST_SIZE)

        # invest_build[flow, period]: binary build indicator
        self.invest_build = self._add_variables(binary=True, coords={'flow': flow_coord, **pc}, name=Var.INVEST_BUILD)

        # invest_active[flow, period]: binary active indicator
        self.invest_active = self._add_variables(binary=True, coords={'flow': flow_coord, **pc}, name=Var.INVEST_ACTIVE)

        # invest_size_at_build[flow, period]: invest_size * build (big-M)
        self.invest_size_at_build = self._add_variables(
            lower=0, upper=upper, coords={'flow': flow_coord, **pc}, name=Var.INVEST_SIZE_AT_BUILD
        )

    def _create_status_variables(self) -> None:
        """Create binary on/off variables for flows with Status."""
        ds = self.data.flows
        if ds.status is None:
            return

        status_ids = ds.status.uptime_min.coords[Dim.STATUS_FLOW].values
        flow_coord = xr.DataArray(status_ids, dims=['flow'])
        tp = {'flow': flow_coord, **self.data.dims.coords(time=True)}

        self.flow_on = self._add_variables(binary=True, coords=tp, name=Var.FLOW_ON)
        self.flow_startup = self._add_variables(binary=True, coords=tp, name=Var.FLOW_STARTUP)
        self.flow_shutdown = self._add_variables(binary=True, coords=tp, name=Var.FLOW_SHUTDOWN)

    def _status_flow_ids(self) -> set[str]:
        """Return ids of flows with Status, or empty set."""
        ds = self.data.flows
        if ds.status is None:
            return set()
        return set(ds.status.uptime_min.coords[Dim.STATUS_FLOW].values)

    def _create_component_status_variables(self) -> None:
        """Create binary on/off variables for components with Status."""
        ds = self.data.flows
        if ds.cstatus is None:
            return

        comp_ids = ds.cstatus.uptime_min.coords[Dim.CSTATUS_COMPONENT].values
        comp_coord = xr.DataArray(comp_ids, dims=['component'])
        tp = {'component': comp_coord, **self.data.dims.coords(time=True)}

        self.component_on = self._add_variables(binary=True, coords=tp, name=Var.COMPONENT_ON)
        self.component_startup = self._add_variables(binary=True, coords=tp, name=Var.COMPONENT_STARTUP)
        self.component_shutdown = self._add_variables(binary=True, coords=tp, name=Var.COMPONENT_SHUTDOWN)

    def _governed_flows_map(self) -> dict[str, list[str]]:
        """Return ``{component_id: [flow_ids governed]}`` from data, or empty."""
        cst = self.data.flows.cstatus
        gf = cst.governed_flows if cst is not None else None
        if gf is None:
            return {}
        result: dict[str, list[str]] = {}
        for comp_id in gf.coords[Dim.CSTATUS_COMPONENT].values:
            row = gf.sel(cstatus_component=comp_id).values
            result[str(comp_id)] = [str(f) for f in row if str(f)]
        return result

    def _component_status_flow_ids(self) -> set[str]:
        """Return ids of flows governed by component-level Status, or empty set."""
        return {fid for fids in self._governed_flows_map().values() for fid in fids}

    def _sizing_flow_ids(self) -> set[str]:
        """Return ids of sizing flows, or empty set."""
        if self.flow_size is None:
            return set()
        return set(self.flow_size.coords['flow'].values)

    def _constrain_flow_rates_plain(self) -> None:
        """Apply flow rate bounds for fixed-size flows without Status.

        P in [size * rel_lb, size * rel_ub] or P = size * profile.
        """
        ds = self.data.flows
        sizing_ids = self._sizing_flow_ids()
        status_ids = self._status_flow_ids()
        cstatus_ids = self._component_status_flow_ids()
        exclude = sizing_ids | status_ids | cstatus_ids

        size = ds.size
        rel_lb = ds.rel_lb
        rel_ub = ds.rel_ub
        fixed = ds.fixed_profile
        is_bounded = ds.bound_type == BoundType.BOUNDED
        is_profile = ds.bound_type == BoundType.PROFILE

        # Mask: has fixed size AND not sizing AND not status
        is_plain = size.notnull()
        if exclude:
            exclude_mask = xr.DataArray(
                [fid in exclude for fid in size.coords['flow'].values],
                dims=['flow'],
                coords={'flow': size.coords['flow']},
            )
            is_plain = is_plain & ~exclude_mask

        plain_bounded = is_bounded & is_plain
        if plain_bounded.any():
            mask = plain_bounded.broadcast_like(rel_lb)
            self.m.add_constraints(self.flow_rate >= size * rel_lb, name='flow_lb', mask=mask)
            self.m.add_constraints(self.flow_rate <= size * rel_ub, name='flow_ub', mask=mask)

        plain_profile = is_profile & is_plain
        if plain_profile.any():
            mask = plain_profile.broadcast_like(fixed) & fixed.notnull()
            self.m.add_constraints(self.flow_rate == size * fixed, name='flow_fix', mask=mask)

    def _constrain_flow_rates_sizing(self) -> None:
        """Apply flow rate bounds for investable flows without Status.

        P in [S * rel_lb, S * rel_ub] or P = S * profile.
        """
        if self.flow_size is None:
            return

        ds = self.data.flows
        status_ids = self._status_flow_ids()

        all_invest_ids = list(self.flow_size.coords['flow'].values)
        invest_ids = [fid for fid in all_invest_ids if fid not in status_ids]
        if not invest_ids:
            return

        fr = self.flow_rate.sel(flow=invest_ids)
        rl = ds.rel_lb.sel(flow=invest_ids)
        ru = ds.rel_ub.sel(flow=invest_ids)
        fp = ds.fixed_profile.sel(flow=invest_ids)
        inv_bounded = (ds.bound_type == BoundType.BOUNDED).sel(flow=invest_ids)
        inv_profile = (ds.bound_type == BoundType.PROFILE).sel(flow=invest_ids)
        fs = self._flow_size_on_time(invest_ids)

        var_mask = inv_bounded.broadcast_like(rl)
        if var_mask.any():
            self.m.add_constraints(fr >= rl * fs, name='flow_lb_sizing', mask=var_mask)
            self.m.add_constraints(fr <= ru * fs, name='flow_ub_sizing', mask=var_mask)

        if inv_profile.any():
            fix_mask = inv_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == fp * fs, name='flow_fix_sizing', mask=fix_mask)

    def _constrain_flow_rates_status(self) -> None:
        """Apply semi-continuous flow rate bounds for flows with Status.

        Fixed-size: P <= size * rel_ub * on, P >= size * rel_lb * on.
        Sizing: big-M formulation via _constrain_flow_rates_status_sizing().
        """
        if self.flow_on is None:
            return

        ds = self.data.flows
        all_status_ids = list(self.flow_on.coords['flow'].values)
        sizing_ids = self._sizing_flow_ids()
        overlap = sizing_ids & set(all_status_ids)

        # Dispatch status+sizing flows to dedicated method
        if overlap:
            self._constrain_flow_rates_status_sizing(sorted(overlap))

        # Status-only flows (fixed size)
        status_ids = [fid for fid in all_status_ids if fid not in overlap]
        if not status_ids:
            return

        fr = self.flow_rate.sel(flow=status_ids)
        size = ds.size.sel(flow=status_ids)
        rl = ds.rel_lb.sel(flow=status_ids)
        ru = ds.rel_ub.sel(flow=status_ids)
        fp = ds.fixed_profile.sel(flow=status_ids)
        is_bounded = (ds.bound_type == BoundType.BOUNDED).sel(flow=status_ids)
        is_profile = (ds.bound_type == BoundType.PROFILE).sel(flow=status_ids)

        stat_bounded = is_bounded & size.notnull()
        if stat_bounded.any():
            mask = stat_bounded.broadcast_like(rl)
            self.m.add_constraints(fr >= size * rl * self.flow_on, name='flow_lb_status', mask=mask)
            self.m.add_constraints(fr <= size * ru * self.flow_on, name='flow_ub_status', mask=mask)

        stat_profile = is_profile & size.notnull()
        if stat_profile.any():
            mask = stat_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == size * fp * self.flow_on, name='flow_fix_status', mask=mask)

    def _constrain_flow_rates_status_sizing(self, flow_ids: list[str]) -> None:
        """Apply big-M flow rate bounds for flows with both Status and Sizing.

        Three constraints decouple the binary on/off from the continuous size:
          P <= on * M⁺          (big-M: forces on=1 when P>0)
          P <= S  * rel_ub      (rate limited by invested size)
          P >= (on - 1) * M⁻ + S * rel_lb  (enforces minimum when on=1)

        Where M⁺ = size_max * rel_ub, M⁻ = size_max * rel_lb.

        A ``fixed_relative_profile`` on such a flow has no formulation here
        and is rejected (the profile path exists only for fixed sizes).

        Args:
            flow_ids: Flows that have both Status and Sizing.
        """
        ds = self.data.flows
        assert self.flow_on is not None
        assert self.flow_size is not None

        profile_ids = [fid for fid in flow_ids if str(ds.bound_type.sel(flow=fid).values) == BoundType.PROFILE]
        if profile_ids:
            raise ValueError(
                f'fixed_relative_profile is not supported on flows with both Status and '
                f'Sizing/Investment: {profile_ids} — fix the size or drop the profile'
            )

        fr = self.flow_rate.sel(flow=flow_ids)
        on = self.flow_on.sel(flow=flow_ids)
        fs = self._flow_size_on_time(flow_ids)
        rl = ds.rel_lb.sel(flow=flow_ids)
        ru = ds.rel_ub.sel(flow=flow_ids)

        # size_max: combine from sizing and invest sources
        sizing_ids = set(ds.sizing.max.coords[Dim.SIZING_FLOW].values) if ds.sizing is not None else set()
        invest_ids = set(ds.invest.max.coords[Dim.INVEST_FLOW].values) if ds.invest is not None else set()
        parts: list[xr.DataArray] = []
        for fid in flow_ids:
            if fid in sizing_ids:
                assert ds.sizing is not None
                parts.append(ds.sizing.max.sel(sizing_flow=fid).rename('flow'))
            elif fid in invest_ids:
                assert ds.invest is not None
                parts.append(ds.invest.max.sel(invest_flow=fid).rename('flow'))
            else:
                raise ValueError(f'Flow {fid!r} has Status+Sizing but no size_max in sizing or investment data')
        size_max = xr.DataArray([float(p) for p in parts], dims=['flow'], coords={'flow': flow_ids})

        big_m_ub = size_max * ru  # M⁺
        big_m_lb = size_max * rl  # M⁻

        self.m.add_constraints(fr <= on * big_m_ub, name='flow_ub_status_sizing_bigm')
        self.m.add_constraints(fr <= fs * ru, name='flow_ub_status_sizing_size')
        self.m.add_constraints(fr >= (on - 1) * big_m_lb + fs * rl, name='flow_lb_status_sizing')

        self._gate_status_to_build(flow_ids, sizing_ids, invest_ids)

    def _gate_status_to_build(self, flow_ids: list[str], sizing_ids: set[str], invest_ids: set[str]) -> None:
        """Prevent on=1 for units that were not built.

        Replaces a former ``on <= S`` constraint, which was vacuous for
        sizes >= 1 and wrongly forced on=0 for legitimately built sizes < 1.

        - Investment flows: ``on <= invest_active`` (off in inactive periods).
        - Sizing with ``size_min > 0``: ``S >= size_min * on``.
        - Optional sizing with ``size_min == 0``: ``on <= size_indicator``.
        - Mandatory sizing with ``size_min == 0`` stays ungated: a zero-size
          "unit" already has its rate forced to 0, so an on=1 there is
          cosmetic unless running costs are negative.

        Args:
            flow_ids: Flows with both Status and Sizing/Investment.
            sizing_ids: Flow ids sized via ``Sizing``.
            invest_ids: Flow ids sized via ``Investment``.
        """
        ds = self.data.flows
        assert self.flow_on is not None

        if gated := sorted(set(flow_ids) & invest_ids):
            assert self.invest_active is not None
            self.m.add_constraints(
                self.flow_on.sel(flow=gated) <= self.invest_active.sel(flow=gated), name='flow_on_requires_active'
            )

        status_sizing = sorted(set(flow_ids) & sizing_ids)
        if not status_sizing:
            return
        assert ds.sizing is not None and self.flow_size is not None
        size_min = ds.sizing.min.rename({Dim.SIZING_FLOW: 'flow'}).sel(flow=status_sizing)

        if min_pos := [fid for fid in status_sizing if float(size_min.sel(flow=fid)) > 0]:
            self.m.add_constraints(
                self.flow_size.sel(flow=min_pos) >= size_min.sel(flow=min_pos) * self.flow_on.sel(flow=min_pos),
                name='flow_on_requires_size',
            )
        if self.flow_size_indicator is not None:
            indicator_ids = set(map(str, self.flow_size_indicator.coords['flow'].values))
            if min_zero := [
                fid for fid in status_sizing if float(size_min.sel(flow=fid)) == 0 and fid in indicator_ids
            ]:
                self.m.add_constraints(
                    self.flow_on.sel(flow=min_zero) <= self.flow_size_indicator.sel(flow=min_zero),
                    name='flow_on_requires_indicator',
                )

    def _constrain_flow_rates_component_status(self) -> None:
        """Apply gating constraints for flows governed by component-level Status.

        For each component with Status and each governed flow:
          bounded:  size * rel_lb * on <= P <= size * rel_ub * on
          profile:  P == size * profile * on

        Sizing/Investment governed flows are not yet supported and raise.
        """
        if self.component_on is None:
            return

        ds = self.data.flows
        sizing_ids = self._sizing_flow_ids()
        # Piecewise converters: their flows are already gated via the
        # add_piecewise_formulation `active=` parameter (when active=0, all
        # auxiliary weights -> 0 -> all curve flows pinned to 0). No extra
        # per-flow gating needed.
        piecewise_comps = set(self.data.piecewise.converter_ids()) if self.data.piecewise is not None else set()

        for comp_id, flow_ids in self._governed_flows_map().items():
            if comp_id in piecewise_comps:
                continue
            on = self.component_on.sel(component=comp_id)  # (time, period?)
            for fid in flow_ids:
                if fid in sizing_ids:
                    msg = (
                        f'Component {comp_id!r}: governed flow {fid!r} has Sizing/Investment, '
                        f'which is not yet supported with component-level status'
                    )
                    raise NotImplementedError(msg)

                fr = self.flow_rate.sel(flow=fid)
                bt = str(ds.bound_type.sel(flow=fid).values)
                size_val = ds.size.sel(flow=fid).values
                if np.isnan(size_val):  # pragma: no cover
                    # Defensive invariant check — callers (Storage.__post_init__, future
                    # PiecewiseConversion) ensure governed flows are sized. Reaching this branch
                    # means an invariant was violated upstream.
                    msg = f'Component {comp_id!r}: governed flow {fid!r} is unsized'
                    raise ValueError(msg)
                size = float(size_val)
                rl = ds.rel_lb.sel(flow=fid)
                ru = ds.rel_ub.sel(flow=fid)
                if bt == BoundType.BOUNDED:
                    self.m.add_constraints(fr >= size * rl * on, name=f'flow_lb_cstatus_{fid}')
                    self.m.add_constraints(fr <= size * ru * on, name=f'flow_ub_cstatus_{fid}')
                elif bt == BoundType.PROFILE:
                    fp = ds.fixed_profile.sel(flow=fid)
                    self.m.add_constraints(fr == size * fp * on, name=f'flow_fix_cstatus_{fid}')
                else:  # pragma: no cover
                    # Defensive — UNSIZED is caught by the NaN-size check above. Reaching
                    # this branch means a new BoundType was added without updating this dispatch.
                    msg = f'Component {comp_id!r}: governed flow {fid!r} has unsupported bound_type {bt!r}'
                    raise ValueError(msg)

    def _constrain_flow_aggregates(self) -> None:
        """Bound per-period flow-hour aggregates.

        Flow hours (absolute):   H̲_f <= Σ_t P_{f,t}·Δt_t <= H̄_f
        Load factor (relative):  λ̲_f·S_f·T <= Σ_t P_{f,t}·Δt_t <= λ̄_f·S_f·T

        Each period is bounded independently. For Sizing/Investment flows
        the load factor multiplies the size *variable*; T = Σ_t Δt_t.

        See: docs/math/flows.md
        """
        ds = self.data.flows
        bounds = (ds.flow_hours_min, ds.flow_hours_max, ds.load_factor_min, ds.load_factor_max)
        if all(b is None for b in bounds):
            return
        dims = self.data.dims
        w = dims.dt * dims.weights
        flow_hours = dims.sum_time(self.flow_rate * w)  # (flow[, period])

        if ds.flow_hours_min is not None:
            self.m.add_constraints(
                flow_hours >= ds.flow_hours_min, name='flow_hours_min', mask=ds.flow_hours_min.notnull()
            )
        if ds.flow_hours_max is not None:
            self.m.add_constraints(
                flow_hours <= ds.flow_hours_max, name='flow_hours_max', mask=ds.flow_hours_max.notnull()
            )

        total_duration = dims.sum_time(w)  # T [h] per period
        sized_flow_ids = self._sizing_flow_ids()
        for lf, name, sign in (
            (ds.load_factor_min, 'load_factor_min', 1),
            (ds.load_factor_max, 'load_factor_max', -1),
        ):
            if lf is None:
                continue
            lf_ids = list(lf.coords['flow'].values[lf.notnull().values])
            fixed_ids = [fid for fid in lf_ids if fid not in sized_flow_ids]
            var_ids = [fid for fid in lf_ids if fid in sized_flow_ids]
            if fixed_ids:
                # Fixed size: constant RHS λ·S·T
                rhs = lf.sel(flow=fixed_ids) * ds.size.sel(flow=fixed_ids) * total_duration
                lhs = flow_hours.sel(flow=fixed_ids)
                if sign > 0:
                    self.m.add_constraints(lhs >= rhs, name=name)
                else:
                    self.m.add_constraints(lhs <= rhs, name=name)
            if var_ids:
                # Sizing/Investment: size is a variable — move to LHS
                assert self.flow_size is not None
                coeff = lf.sel(flow=var_ids) * total_duration
                expr = flow_hours.sel(flow=var_ids) - coeff * self.flow_size.sel(flow=var_ids)
                if sign > 0:
                    self.m.add_constraints(expr >= 0, name=f'{name}_sized')
                else:
                    self.m.add_constraints(expr <= 0, name=f'{name}_sized')

    def _constrain_flow_ramps(self) -> None:
        """Limit flow rate changes between consecutive timesteps.

        Ramp up:   P_{f,t} - P_{f,t-1} <= r⁺_{f,t}·S_f·Δt_t + M_f·startup_{f,t}
        Ramp down: P_{f,t-1} - P_{f,t} <= r⁻_{f,t}·S_f·Δt_t + M_f·shutdown_{f,t}

        Applies from the second timestep onward; periods are independent.
        For Sizing/Investment flows the size variable enters the constraint
        (r·Δt moves to the LHS as a coefficient).

        For status flows the ramp does not bind across on/off transitions:
        the startup/shutdown binary relaxes it with M = static size bound
        (transitions are pinned to actual state changes by the
        ``status|exclusive`` constraint). Component-status flows relax via
        their component's transition binaries.

        See: docs/math/flows.md
        """
        ds = self.data.flows
        if ds.ramp_up is None and ds.ramp_down is None:
            return
        time_vals = self.flow_rate.coords['time'].values
        if len(time_vals) < 2:
            return
        coords_from_1 = time_vals[1:]
        curr = self.flow_rate.isel(time=slice(1, None))
        prev = self.flow_rate.isel(time=slice(None, -1)).assign_coords(time=coords_from_1)
        dt_t = self.data.dims.dt.isel(time=slice(1, None))
        status_ids = self._status_flow_ids()
        flow_to_comp = {fid: cid for cid, fids in self._governed_flows_map().items() for fid in fids}

        for ramp, name, delta, trans_flow, trans_comp in (
            (ds.ramp_up, 'ramp_up', curr - prev, self.flow_startup, self.component_startup),
            (ds.ramp_down, 'ramp_down', prev - curr, self.flow_shutdown, self.component_shutdown),
        ):
            if ramp is None:
                continue
            non_flow_dims = [d for d in ramp.dims if d != 'flow']
            has_ramp = ramp.notnull().any(non_flow_dims)
            ramp_ids = [str(f) for f in ramp.coords['flow'].values[has_ramp.values]]
            limit = ramp.isel(time=slice(1, None)) * dt_t  # r·Δt  (flow, time[, period])

            plain_ids = [fid for fid in ramp_ids if fid not in status_ids and fid not in flow_to_comp]
            flow_status_ids = [fid for fid in ramp_ids if fid in status_ids]
            self._add_ramp_constraints(plain_ids, name, '', limit, delta, None)
            if flow_status_ids:
                assert trans_flow is not None
                relax = trans_flow.sel(flow=flow_status_ids).isel(time=slice(1, None))
                self._add_ramp_constraints(flow_status_ids, name, '_status', limit, delta, relax)
            # Component-status flows share their component's transition binary
            comp_groups: dict[str, list[str]] = {}
            for fid in ramp_ids:
                if fid in flow_to_comp:
                    comp_groups.setdefault(flow_to_comp[fid], []).append(fid)
            for cid, fids in comp_groups.items():
                assert trans_comp is not None
                relax = trans_comp.sel(component=cid).isel(time=slice(1, None))
                self._add_ramp_constraints(fids, name, f'_cstatus_{cid}', limit, delta, relax)

    def _add_ramp_constraints(
        self,
        ids: list[str],
        name: str,
        suffix: str,
        limit: xr.DataArray,
        delta: Any,
        relax: Any,
    ) -> None:
        """Add ramp constraints for *ids*, optionally relaxed at transitions.

        Args:
            ids: Flow ids to constrain.
            name: Base constraint name (``ramp_up`` / ``ramp_down``).
            suffix: Name suffix distinguishing the status regime.
            limit: r·Δt coefficient, (flow, time[, period]).
            delta: Rate change expression over t >= 1.
            relax: Transition binary relaxing the ramp (M·relax added to the
                allowance), or None for the strict constraint.
        """
        ds = self.data.flows
        sized_flow_ids = self._sizing_flow_ids()
        fixed_ids = [fid for fid in ids if fid not in sized_flow_ids]
        var_ids = [fid for fid in ids if fid in sized_flow_ids]
        chain = self.data.dims.episodes.chain_mask
        if fixed_ids:
            # Fixed size: constant RHS r·S̄·Δt
            rhs = limit.sel(flow=fixed_ids) * ds.size.sel(flow=fixed_ids)
            lhs = delta.sel(flow=fixed_ids)
            if relax is not None:
                lhs = lhs - self._flow_size_bounds(fixed_ids) * relax
            self.m.add_constraints(lhs <= rhs, name=f'flow_{name}{suffix}', mask=rhs.notnull() & chain)
        if var_ids:
            # Sizing/Investment: size is a variable — move to LHS
            coeff = limit.sel(flow=var_ids)
            fs = self._flow_size_on_time(var_ids)
            if 'time' in fs.dims:
                fs = fs.isel(time=slice(1, None))
            expr = delta.sel(flow=var_ids) - coeff * fs
            if relax is not None:
                expr = expr - self._flow_size_bounds(var_ids) * relax
            self.m.add_constraints(expr <= 0, name=f'flow_{name}_sized{suffix}', mask=coeff.notnull() & chain)

    def _flow_size_on_time(self, flow_ids: list[str]) -> Any:
        """Selected flow sizes expanded onto the flat time axis.

        Size is a per-period decision; operational constraints need it per
        timestep.

        Args:
            flow_ids: Qualified flow ids; each must have a size variable.
        """
        assert self.flow_size is not None
        return self.data.dims.map_to_time(self.flow_size.sel(flow=flow_ids))

    def _flow_size_bounds(self, flow_ids: list[str]) -> xr.DataArray:
        """Static per-flow upper size bounds: fixed value or sizing/invest max.

        Args:
            flow_ids: Qualified flow ids; each must be sized.
        """
        fds = self.data.flows
        vals: list[float] = []
        for fid in flow_ids:
            v = fds.size.sel(flow=fid).values
            if not np.isnan(v):
                vals.append(float(v))
            elif fds.sizing is not None and fid in fds.sizing.max.coords[Dim.SIZING_FLOW].values:
                vals.append(float(fds.sizing.max.sel(sizing_flow=fid).values))
            elif fds.invest is not None and fid in fds.invest.max.coords[Dim.INVEST_FLOW].values:
                vals.append(float(fds.invest.max.sel(invest_flow=fid).values))
            else:  # pragma: no cover — element validation guards sized flows
                raise ValueError(f'Flow {fid!r} has no static size bound for big-M')
        return xr.DataArray(vals, dims=['flow'], coords={'flow': flow_ids})

    def _constrain_sizing(self) -> None:
        """Constrain sizing variables: S in [min, max] gated by indicator."""
        # --- Flow sizing (Sizing only, not Investment) ---
        fds = self.data.flows
        if fds.sizing is not None:
            assert self.flow_size is not None
            sizing_ids = fds.sizing.min.coords[Dim.SIZING_FLOW].values
            smin = fds.sizing.min.rename({Dim.SIZING_FLOW: 'flow'})
            mandatory = fds.sizing.mandatory

            mand_ids = sizing_ids[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.flow_size.sel(flow=mand_ids) >= smin.sel(flow=mand_ids),
                    name='invest_mand_lb',
                )

            opt_ids = sizing_ids[~mandatory.values]
            if len(opt_ids):
                assert self.flow_size_indicator is not None
                smax = fds.sizing.max.rename({Dim.SIZING_FLOW: 'flow'})
                fs = self.flow_size.sel(flow=opt_ids)
                self.m.add_constraints(fs >= smin.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_lb')
                self.m.add_constraints(fs <= smax.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_ub')

        # --- Storage capacity sizing ---
        if self.storage_capacity is not None:
            sds = self.data.storages
            assert sds is not None
            assert sds.sizing is not None
            smin = sds.sizing.min.rename({Dim.SIZING_STORAGE: 'storage'})
            mandatory = sds.sizing.mandatory.rename({Dim.SIZING_STORAGE: 'storage'})

            mand_ids = self.storage_capacity.coords['storage'].values[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.storage_capacity.sel(storage=mand_ids) >= smin.sel(storage=mand_ids),
                    name='stor_invest_mand_lb',
                )

            opt_ids = self.storage_capacity.coords['storage'].values[~mandatory.values]
            if len(opt_ids):
                assert self.storage_capacity_indicator is not None
                smax = sds.sizing.max.rename({Dim.SIZING_STORAGE: 'storage'})
                sc = self.storage_capacity.sel(storage=opt_ids)
                self.m.add_constraints(
                    sc >= smin.sel(storage=opt_ids) * self.storage_capacity_indicator, name='stor_invest_lb'
                )
                self.m.add_constraints(
                    sc <= smax.sel(storage=opt_ids) * self.storage_capacity_indicator, name='stor_invest_ub'
                )

    def _constrain_investment(self) -> None:
        """Constrain investment variables: build-once, active logic, size linking."""
        if self.invest_size is None:
            return

        fds = self.data.flows
        assert fds.invest is not None
        assert self.invest_build is not None
        assert self.invest_active is not None
        assert self.invest_size_at_build is not None

        invest_ids = list(fds.invest.min.coords[Dim.INVEST_FLOW].values)
        smin = fds.invest.min.rename({Dim.INVEST_FLOW: 'flow'})
        smax = fds.invest.max.rename({Dim.INVEST_FLOW: 'flow'})
        mandatory = fds.invest.mandatory.rename({Dim.INVEST_FLOW: 'flow'})
        lifetime = fds.invest.lifetime.rename({Dim.INVEST_FLOW: 'flow'})
        prior_size = fds.invest.prior_size.rename({Dim.INVEST_FLOW: 'flow'})

        build = self.invest_build
        active = self.invest_active
        inv_size = self.invest_size
        assert self.flow_size is not None
        fs = self.flow_size.sel(flow=invest_ids)
        sab = self.invest_size_at_build

        periods = self.data.dims.period
        assert periods is not None
        period_vals = list(periods.values)
        n_periods = len(period_vals)

        # --- Build-once constraints ---
        build_sum = build.sum('period')
        mand_ids = [fid for fid, m in zip(invest_ids, mandatory.values, strict=True) if m]
        opt_ids = [fid for fid, m in zip(invest_ids, mandatory.values, strict=True) if not m]

        if mand_ids:
            self.m.add_constraints(build_sum.sel(flow=mand_ids) == 1, name='invest_build_once_mand')
        if opt_ids:
            self.m.add_constraints(build_sum.sel(flow=opt_ids) <= 1, name='invest_build_once_opt')

        # --- Active logic per flow ---
        for f_idx, fid in enumerate(invest_ids):
            lt = lifetime.values[f_idx]
            has_prior = prior_size.values[f_idx] > 0
            lt_int = int(lt) if not np.isnan(lt) else None

            for p_idx, p in enumerate(period_vals):
                b_sel = build.sel(flow=fid)
                a_sel = active.sel(flow=fid, period=p)

                if lt_int is None:
                    # No lifetime: once built, active forever
                    # active[p] = sum_{tau <= p} build[tau] + (1 if prior)
                    contributing = [period_vals[t] for t in range(p_idx + 1)]
                    rhs = b_sel.sel(period=contributing).sum('period')
                    if has_prior:
                        self.m.add_constraints(a_sel == rhs + 1, name=f'invest_active_{fid}_p{p}')
                    else:
                        self.m.add_constraints(a_sel == rhs, name=f'invest_active_{fid}_p{p}')
                else:
                    # With lifetime: active for lt_int periods after build
                    # active[p] = sum_{tau: p in [tau, tau+lt)} build[tau] + (1 if prior and p < lt)
                    contributing = [period_vals[t_idx] for t_idx in range(n_periods) if t_idx <= p_idx < t_idx + lt_int]
                    rhs = b_sel.sel(period=contributing).sum('period') if contributing else 0
                    if has_prior and p_idx < lt_int:
                        self.m.add_constraints(a_sel == rhs + 1, name=f'invest_active_{fid}_p{p}')
                    else:
                        self.m.add_constraints(a_sel == rhs, name=f'invest_active_{fid}_p{p}')

            # --- Prevent build when prior is active and no lifetime (can't build twice) ---
            if has_prior:
                self.m.add_constraints(build.sel(flow=fid).sum('period') == 0, name=f'invest_no_build_prior_{fid}')

        # --- Prior size: fix invest_size to prior_size ---
        prior_ids = [fid for fid, ps in zip(invest_ids, prior_size.values, strict=True) if ps > 0]
        if prior_ids:
            ps_vals = prior_size.sel(flow=prior_ids)
            self.m.add_constraints(inv_size.sel(flow=prior_ids) == ps_vals, name='invest_size_prior')

        # --- Size bounds ---
        # invest_size >= size_min (if mandatory or built)
        non_prior_mand = [fid for fid in mand_ids if fid not in prior_ids]
        if non_prior_mand:
            self.m.add_constraints(
                inv_size.sel(flow=non_prior_mand) >= smin.sel(flow=non_prior_mand),
                name='invest_size_lb_mand',
            )
        if opt_ids:
            non_prior_opt = [fid for fid in opt_ids if fid not in prior_ids]
            if non_prior_opt:
                # invest_size >= size_min * sum(build) — only if built
                self.m.add_constraints(
                    inv_size.sel(flow=non_prior_opt)
                    >= smin.sel(flow=non_prior_opt) * build_sum.sel(flow=non_prior_opt),
                    name='invest_size_lb_opt',
                )
                # invest_size <= size_max * sum(build) — zero when not built
                self.m.add_constraints(
                    inv_size.sel(flow=non_prior_opt)
                    <= smax.sel(flow=non_prior_opt) * build_sum.sel(flow=non_prior_opt),
                    name='invest_size_ub_opt',
                )

        # invest_size <= size_max (already via variable upper bound)

        # --- Size ↔ active linking (big-M: flow_size = invest_size when active, 0 when not) ---
        self.m.add_constraints(fs <= smax * active, name='invest_fs_ub_active')
        self.m.add_constraints(fs <= inv_size, name='invest_fs_ub_size')
        self.m.add_constraints(fs >= inv_size - smax * (1 - active), name='invest_fs_lb')

        # --- size_at_build: linearization of invest_size * build ---
        self.m.add_constraints(sab <= smax * build, name='invest_sab_ub_build')
        self.m.add_constraints(sab <= inv_size, name='invest_sab_ub_size')
        self.m.add_constraints(sab >= inv_size - smax * (1 - build), name='invest_sab_lb')

    def _constrain_status(self) -> None:
        """Add switch transition and duration tracking constraints for status flows."""
        if self.flow_on is None:
            return
        assert self.flow_startup is not None
        assert self.flow_shutdown is not None

        ds = self.data.flows
        assert ds.status is not None

        # Rename status_flow -> flow to align with variable dims
        min_up = ds.status.uptime_min.rename({Dim.STATUS_FLOW: 'flow'})
        max_up = ds.status.uptime_max.rename({Dim.STATUS_FLOW: 'flow'})
        min_down = ds.status.downtime_min.rename({Dim.STATUS_FLOW: 'flow'})
        max_down = ds.status.downtime_max.rename({Dim.STATUS_FLOW: 'flow'})
        initial = ds.status.initial.rename({Dim.STATUS_FLOW: 'flow'})

        prev_up = (
            ds.status.previous_uptime.rename({Dim.STATUS_FLOW: 'flow'})
            if ds.status.previous_uptime is not None
            else None
        )
        prev_down = (
            ds.status.previous_downtime.rename({Dim.STATUS_FLOW: 'flow'})
            if ds.status.previous_downtime is not None
            else None
        )

        # Filter to flows with known initial state
        has_initial = initial.notnull()
        previous_state = initial.sel(flow=initial.coords['flow'][has_initial]) if has_initial.any() else None

        episodes = self.data.dims.episodes
        add_switch_transitions(
            self.m,
            self.flow_on,
            self.flow_startup,
            self.flow_shutdown,
            name='status',
            previous_state=previous_state,
            episodes=episodes,
        )

        dt = self.data.dims.dt

        # Uptime tracking
        has_any_up = min_up.notnull().any() | max_up.notnull().any()
        if has_any_up:
            add_duration_tracking(
                self.m,
                self.flow_on,
                dt,
                name='uptime',
                minimum=min_up,
                maximum=max_up,
                previous=prev_up,
                episodes=episodes,
            )

        # Downtime tracking: state = 1 - on
        has_any_down = min_down.notnull().any() | max_down.notnull().any()
        if has_any_down:
            add_duration_tracking(
                self.m,
                1 - self.flow_on,
                dt,
                name='downtime',
                minimum=min_down,
                maximum=max_down,
                previous=prev_down,
                episodes=episodes,
            )

    def _constrain_component_status(self) -> None:
        """Add switch transition and duration tracking constraints for component status."""
        if self.component_on is None:
            return
        assert self.component_startup is not None
        assert self.component_shutdown is not None

        ds = self.data.flows
        assert ds.cstatus is not None

        min_up = ds.cstatus.uptime_min.rename({Dim.CSTATUS_COMPONENT: 'component'})
        max_up = ds.cstatus.uptime_max.rename({Dim.CSTATUS_COMPONENT: 'component'})
        min_down = ds.cstatus.downtime_min.rename({Dim.CSTATUS_COMPONENT: 'component'})
        max_down = ds.cstatus.downtime_max.rename({Dim.CSTATUS_COMPONENT: 'component'})
        initial = ds.cstatus.initial.rename({Dim.CSTATUS_COMPONENT: 'component'})

        prev_up = (
            ds.cstatus.previous_uptime.rename({Dim.CSTATUS_COMPONENT: 'component'})
            if ds.cstatus.previous_uptime is not None
            else None
        )
        prev_down = (
            ds.cstatus.previous_downtime.rename({Dim.CSTATUS_COMPONENT: 'component'})
            if ds.cstatus.previous_downtime is not None
            else None
        )

        has_initial = initial.notnull()
        previous_state = initial.sel(component=initial.coords['component'][has_initial]) if has_initial.any() else None

        episodes = self.data.dims.episodes
        add_switch_transitions(
            self.m,
            self.component_on,
            self.component_startup,
            self.component_shutdown,
            name='cstatus',
            element_dim='component',
            previous_state=previous_state,
            episodes=episodes,
        )

        dt = self.data.dims.dt

        has_any_up = min_up.notnull().any() | max_up.notnull().any()
        if has_any_up:
            add_duration_tracking(
                self.m,
                self.component_on,
                dt,
                name='c_uptime',
                element_dim='component',
                minimum=min_up,
                maximum=max_up,
                previous=prev_up,
                episodes=episodes,
            )

        has_any_down = min_down.notnull().any() | max_down.notnull().any()
        if has_any_down:
            add_duration_tracking(
                self.m,
                1 - self.component_on,
                dt,
                name='c_downtime',
                element_dim='component',
                minimum=min_down,
                maximum=max_down,
                previous=prev_down,
                episodes=episodes,
            )

    def _create_balance(self) -> None:
        """Create carrier balance: ``sum_f(coeff * P) = 0`` for all carriers and timesteps."""
        d = self.data
        coeff = d.carriers.flow_coeff  # (carrier, flow) — NaN for unconnected

        # Replace NaN with 0 for summation (unconnected flows contribute nothing)
        coeff_filled = coeff.fillna(0)

        # Carrier balance: sum over flow dim of (coeff * flow_rate) == 0
        lhs = sparse_weighted_sum(self.flow_rate, coeff_filled, sum_dim='flow', group_dim='carrier')
        self.m.add_constraints(lhs == 0, name='carrier_balance')

    def _create_converter_constraints(self) -> None:
        """Create conversion constraints: ``sum_f(a_f * P) = 0`` per converter and equation."""
        d = self.data
        if d.converters is None:
            return

        ds = d.converters

        # Select flow rates for each (converter, flow) pair
        selected = self.flow_rate.sel(flow=ds.pair_flow)  # (pair, time)
        weighted = selected * ds.pair_coeff  # (pair, eq_idx, time)

        # Group by converter and sum over pairs
        mapping = xr.DataArray(ds.pair_converter.values, dims=['pair'], name='converter')
        lhs = weighted.groupby(mapping).sum()  # (converter, eq_idx, time)

        # Restore original converter order (groupby sorts alphabetically)
        conv_ids = list(ds.eq_mask.coords['converter'].values)
        lhs = lhs.sel(converter=conv_ids)

        # Drop flow coord left by vectorized sel
        lhs = lhs.drop_vars('flow', errors='ignore')

        # Broadcast eq_mask (converter, eq_idx) to (converter, eq_idx, time)
        mask_3d = ds.eq_mask.expand_dims(time=ds.pair_coeff.coords['time'])
        self.m.add_constraints(lhs == 0, name='conversion', mask=mask_3d)

    def _create_piecewise_constraints(self) -> None:
        """Create piecewise-linear conversion constraints via linopy's piecewise API.

        For each converter with ``PiecewiseConversion``: builds one
        ``add_piecewise_formulation`` call linking all curve flows through
        shared interpolation weights. The optional ``PiecewiseConversion.status``
        wires in the existing ``component_on`` binary as the ``active`` gate.
        Per-converter availability is enforced separately as
        ``flow_rate <= avail * max_bp * active``.
        """
        import warnings
        from typing import cast

        from linopy import EvolvingAPIWarning
        from linopy.piecewise import add_piecewise_formulation

        from fluxopt.types import PiecewiseMethod

        pw = self.data.piecewise
        if pw is None:
            return

        for conv_id in pw.converter_ids():
            method = cast('PiecewiseMethod', str(pw.method.sel(pw_converter=conv_id).values))
            avail = pw.availability.sel(pw_converter=conv_id)  # (time,)
            has_status = bool(pw.has_status.sel(pw_converter=conv_id).values)

            active = self.component_on.sel(component=conv_id) if has_status and self.component_on is not None else None

            mask = pw.pair_converter.values == conv_id
            pair_indices = np.where(mask)[0]

            pairs: list[tuple[Any, ...]] = []
            for idx in pair_indices:
                fid = str(pw.pair_flow.values[idx])
                bound = str(pw.pair_bound.values[idx])
                # Wrap as LinearExpression and drop the per-flow scalar coord so
                # linopy can broadcast pairs without merge conflicts on 'flow'.
                expr = (1.0 * self.flow_rate.sel(flow=fid)).drop_vars('flow', errors='ignore')
                bps = (
                    pw.breakpoints.isel(pw_pair=idx)
                    .drop_vars(Dim.PW_PAIR, errors='ignore')
                    .rename({'breakpoint': '_breakpoint'})
                )
                pairs.append((expr, bps) if bound == '==' else (expr, bps, bound))

            # fluxopt owns the API risk of add_piecewise_formulation — surface it
            # in our changelog instead of leaking warnings into user output.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', EvolvingAPIWarning)
                formulation = add_piecewise_formulation(
                    self.m,
                    *pairs,
                    method=method,
                    active=active,
                    name=f'pw_{conv_id}',
                )
            self._piecewise[conv_id] = formulation

            # Availability constraint: scale upper envelope, not the curve.
            # Use max-over-breakpoint, not last — SOS2 allows non-monotonic breakpoints.
            ref_expr = pairs[0][0]
            ref_idx = pair_indices[0]
            max_bp = pw.breakpoints.isel(pw_pair=ref_idx).max('breakpoint')  # (time,)
            if active is not None:
                self.m.add_constraints(
                    ref_expr <= avail * max_bp * active,
                    name=f'pw_avail_{conv_id}',
                )
            else:
                self.m.add_constraints(
                    ref_expr <= avail * max_bp,
                    name=f'pw_avail_{conv_id}',
                )

    def _term_variable(self, term: EffectTerm) -> Variable:
        """Resolve a term's solver variable, selected to the entity ids its coeff covers."""
        by_name: dict[str, Variable | None] = {
            Var.FLOW_RATE: self.flow_rate,
            Var.FLOW_ON: self.flow_on,
            Var.FLOW_STARTUP: self.flow_startup,
            Var.COMPONENT_ON: self.component_on,
            Var.COMPONENT_STARTUP: self.component_startup,
            Var.FLOW_SIZE: self.flow_size,
            Var.FLOW_SIZE_INDICATOR: self.flow_size_indicator,
            Var.STORAGE_CAPACITY: self.storage_capacity,
            Var.STORAGE_SIZE_INDICATOR: self.storage_capacity_indicator,
            Var.INVEST_SIZE_AT_BUILD: self.invest_size_at_build,
            Var.INVEST_BUILD: self.invest_build,
            Var.INVEST_ACTIVE: self.invest_active,
        }
        assert term.var is not None
        var = by_name[term.var]
        assert var is not None, f'effect term {term.key!r} references variable {term.var!r} before it was created'
        if term.select is not None:
            var = var.sel({term.entity_dim: list(term.select)})
        return var

    def _temporal_effect_rhs(self, terms: list[EffectTerm]) -> Any:
        """Sum of all temporal effect contributions, folded into effect--total.

        Builds one expression per declared temporal term,
        ``sum_entity(coeff * var [* dt])`` — no per-timestep variables.

        Cross-effect chains ``E = D + C·E`` are resolved via the closed form
        ``E = (I - C)^{-1}·D``, contracting the Leontief inverse into each
        term's numeric coefficient array: the fold is linear in the terms,
        while multiplying the assembled linopy expression instead would fan
        every term out across all source effects.

        Returns ``0`` when no temporal terms are declared.
        """
        d = self.data
        cf = d.effects.cf_temporal
        leontief = _leontief(cf) if cf is not None else None

        rhs: Any = 0
        for term in (t for t in terms if t.domain == 'temporal'):
            var = self._term_variable(term)
            coeff = term.coeff * d.dims.dt if term.scale_dt else term.coeff
            if leontief is not None:
                coeff = _apply_leontief(leontief, coeff)
            if term.sparse:
                expr = sparse_weighted_sum(var, coeff, sum_dim=term.entity_dim, group_dim='effect')
            else:
                expr = (coeff * var).sum(term.entity_dim)
            rhs = rhs + expr
        return rhs

    def _create_effects(self) -> None:
        """Effect tracking: temporal and lump domains.

        Both the expressions built here and the post-solve decomposition in
        ``contributions.py`` derive from the same term declarations
        (:func:`fluxopt.effect_terms.effect_terms`).
        """
        d = self.data
        ds = d.effects

        effect_ids = ds.total_min.coords['effect']

        if len(effect_ids) == 0:
            return

        terms = effect_terms(d)
        temporal_rhs = self._temporal_effect_rhs(terms)

        # --- Lump domain: effect_lump[effect(, period)] ---
        # Combines all non-temporal contributions (sizing, investment recurring, investment at-build)
        pc = self.data.dims.coords(period=True)
        self.effect_lump = self.m.add_variables(coords={'effect': effect_ids, **pc}, name=Var.EFFECT_LUMP)

        # Accumulate direct lump contributions per effect. Variable terms are
        # summed first so the linopy expression stays the left operand;
        # constant terms (mandatory fixed costs) are folded in afterwards.
        lump_direct: Any = 0
        lump_const: Any = 0
        for term in (t for t in terms if t.domain == 'lump'):
            if term.var is None:
                lump_const = lump_const + term.coeff.sum(term.entity_dim)
            else:
                lump_direct = lump_direct + (term.coeff * self._term_variable(term)).sum(term.entity_dim)
        if not isinstance(lump_const, int):
            lump_direct = lump_direct + lump_const if not isinstance(lump_direct, int) else lump_const

        # Cross-effect lump: mean(cf_temporal, 'time')[k,j] * effect_lump[j].
        # A time-varying factor has no meaning for one-time (lump) quantities,
        # so it is rejected when the source effect carries lump contributions;
        # the mean is only ever applied where the source's lump is structurally zero.
        lump_rhs: Any = lump_direct
        if ds.cf_temporal is not None:
            cf_lump = d.dims.mean_time(ds.cf_temporal)  # (effect, source_effect[, period])
            varying = d.dims.varies_within_period(ds.cf_temporal)  # (effect, source_effect)
            if bool(varying.any().item()):
                bearing = _lump_bearing_effects(terms, cf_lump)
                mask = varying & bearing.rename({'effect': 'source_effect'})
                mask = mask.any([dim for dim in mask.dims if dim not in ('effect', 'source_effect')])
                if bool(mask.any().item()):
                    pairs = ', '.join(
                        f'{mask.effect.values[i]}<-{mask.source_effect.values[j]}'
                        for i, j in zip(*mask.values.nonzero(), strict=True)
                    )
                    raise ValueError(
                        f'Time-varying contribution_from for {pairs} is ill-defined: the source effect '
                        'carries lump (sizing/fixed) contributions, and a per-timestep factor has no meaning '
                        'for one-time quantities. Use a scalar factor, or move the lump share into a '
                        'separate effect with a scalar factor.'
                    )
            source_p = self.effect_lump.rename({'effect': 'source_effect'})
            cross = (cf_lump * source_p).sum('source_effect')
            lump_rhs = cross + lump_direct  # linopy expr must be left operand

        self.m.add_constraints(self.effect_lump == lump_rhs, name='effect_lump_eq')

        # --- Total: effect_total[effect(, period)] ---
        self.effect_total = self.m.add_variables(coords={'effect': effect_ids, **pc}, name=Var.EFFECT_TOTAL)
        rhs: Any = self.effect_lump
        if not isinstance(temporal_rhs, int):
            rhs = d.dims.sum_time(temporal_rhs * d.dims.weights) + self.effect_lump
        self.m.add_constraints(self.effect_total == rhs, name='effect_total_eq')

        # Per-period bounds on effect_total
        min_pp = ds.periodic_min  # (effect[, period]) — NaN = unbounded
        max_pp = ds.periodic_max
        has_min_pp = min_pp.notnull()
        if has_min_pp.any():
            self.m.add_constraints(self.effect_total >= min_pp, name='effect_periodic_min', mask=has_min_pp)
        has_max_pp = max_pp.notnull()
        if has_max_pp.any():
            self.m.add_constraints(self.effect_total <= max_pp, name='effect_periodic_max', mask=has_max_pp)

        # Weighted total bounds (across all periods)
        # Single-period: effect_total has no period dim, bound applies directly.
        # Multi-period: weighted sum across periods, using per-effect period_weights
        # if set, else global period_weights, else unweighted.
        total_min = ds.total_min  # (effect,) — NaN = unbounded
        total_max = ds.total_max
        total_sum: Any
        if 'period' in self.effect_total.dims:
            # Multi-period: weighted sum across periods.
            # Per-effect weights override global; both are always set in multi-period.
            assert d.dims.period_weights is not None
            if ds.period_weights is not None:
                w_per_effect = ds.period_weights.fillna(d.dims.period_weights)
                total_sum = (self.effect_total * w_per_effect).sum('period')
            else:
                total_sum = (self.effect_total * d.dims.period_weights).sum('period')
        else:
            total_sum = self.effect_total
        has_min = total_min.notnull()
        if has_min.any():
            self.m.add_constraints(total_sum >= total_min, name='effect_total_min', mask=has_min)
        has_max = total_max.notnull()
        if has_max.any():
            self.m.add_constraints(total_sum <= total_max, name='effect_total_max', mask=has_max)

    def _create_storage(self) -> None:
        """Create storage variables, level balance, and prior/cyclic conditions."""
        d = self.data
        if d.storages is None:
            return
        ds = d.storages

        stor_ids = ds.capacity.coords['storage']

        # storage_level[storage, time(, period)] >= 0  (end-of-period convention)
        self.storage_level = self.m.add_variables(
            lower=0,
            coords={'storage': stor_ids, **self.data.dims.coords(time=True)},
            name=Var.STORAGE_LEVEL,
        )

        # --- Capacity bounds on storage_level ---
        cap = ds.capacity  # (storage,) — NaN for uncapped/investable
        has_fixed_cap = cap.notnull()
        has_invest_cap = self.storage_capacity is not None

        # Fixed-capacity storages: level <= capacity (parameter)
        if has_fixed_cap.any():
            cap_2d = cap.broadcast_like(xr.DataArray(np.nan, dims=['storage', 'time'], coords=[stor_ids, d.dims.time]))
            mask_cap = has_fixed_cap.broadcast_like(cap_2d)
            self.m.add_constraints(self.storage_level <= cap_2d, name='level_cap', mask=mask_cap)

        # Investable storages: level <= capacity (variable)
        if has_invest_cap:
            assert self.storage_capacity is not None
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            level_invest = self.storage_level.sel(storage=invest_ids)
            cap_t = d.dims.map_to_time(self.storage_capacity)
            self.m.add_constraints(level_invest <= cap_t, name='level_cap_invest')

        # --- Relative level bounds ---
        # For fixed-capacity storages
        if has_fixed_cap.any():
            rel_lb = ds.rel_level_lb
            rel_ub = ds.rel_level_ub

            abs_lb = rel_lb * cap
            has_lb = has_fixed_cap.broadcast_like(rel_lb) & (abs_lb > 1e-12)
            if has_lb.any():
                self.m.add_constraints(self.storage_level >= abs_lb, name='level_lb', mask=has_lb)

            abs_ub = rel_ub * cap
            cap_2d_check = cap.broadcast_like(rel_ub)
            has_ub = has_fixed_cap.broadcast_like(rel_ub) & (abs_ub < cap_2d_check - 1e-12)
            if has_ub.any():
                self.m.add_constraints(self.storage_level <= abs_ub, name='level_ub', mask=has_ub)

        # For investable storages: relative bounds use capacity variable
        if has_invest_cap:
            assert self.storage_capacity is not None
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            rel_lb_inv = ds.rel_level_lb.sel(storage=invest_ids)
            rel_ub_inv = ds.rel_level_ub.sel(storage=invest_ids)
            level_invest = self.storage_level.sel(storage=invest_ids)
            cap_t = d.dims.map_to_time(self.storage_capacity)

            has_lb = (rel_lb_inv > 1e-12).any('time')
            if has_lb.any():
                lb_mask = has_lb.broadcast_like(rel_lb_inv) & (rel_lb_inv > 1e-12)
                self.m.add_constraints(
                    level_invest >= rel_lb_inv * cap_t,
                    name='level_lb_invest',
                    mask=lb_mask,
                )

            has_ub = (rel_ub_inv < 1 - 1e-12).any('time')
            if has_ub.any():
                ub_mask = has_ub.broadcast_like(rel_ub_inv) & (rel_ub_inv < 1 - 1e-12)
                self.m.add_constraints(
                    level_invest <= rel_ub_inv * cap_t,
                    name='level_ub_invest',
                    mask=ub_mask,
                )

        # Map charge/discharge flows to storage dimension via sel + rename
        charge_fids = [str(v) for v in ds.charge_flow.values]
        discharge_fids = [str(v) for v in ds.discharge_flow.values]
        stor_vals = stor_ids.values

        charge_rates = self.flow_rate.sel(flow=charge_fids).rename({'flow': 'storage'})
        charge_rates = charge_rates.assign_coords({'storage': stor_vals})
        discharge_rates = self.flow_rate.sel(flow=discharge_fids).rename({'flow': 'storage'})
        discharge_rates = discharge_rates.assign_coords({'storage': stor_vals})

        # Precompute pure-xarray coefficients (no linopy overhead)
        loss_factor = (1 - ds.loss) ** d.dims.dt  # (storage, time)
        charge_factor = ds.eta_c * d.dims.dt  # (storage, time)
        discharge_factor = d.dims.dt / ds.eta_d  # (storage, time)

        inflow = charge_rates * charge_factor
        outflow = discharge_rates * discharge_factor

        # --- Prior variable + single balance for ALL storages ---
        # prior[storage] is a free variable representing the state before period 0.
        # Cyclic and fixed-prior constraints pin it; otherwise it's free.
        self.prior_storage_level = self.m.add_variables(
            lower=0, coords={'storage': stor_ids, **self.data.dims.coords(period=True)}, name=Var.STORAGE_PRIOR
        )

        add_accumulation_constraints(
            self.m,
            self.storage_level,
            inflow=inflow,
            outflow=outflow,
            decay=loss_factor,
            initial=self.prior_storage_level,
            name='storage_balance',
            episodes=d.dims.episodes,
        )

        level_end = d.dims.last_time(self.storage_level)

        cyclic_mask = ds.cyclic.values.astype(bool)
        if np.any(cyclic_mask):
            cyc_ids = [str(s) for s, c in zip(stor_ids.values, cyclic_mask, strict=True) if c]
            self.m.add_constraints(
                self.prior_storage_level.sel(storage=cyc_ids) == level_end.sel(storage=cyc_ids),
                name='storage_prior_cyc',
            )

        # Fixed prior: prior == value
        has_prior = ds.prior_level.notnull().values
        if np.any(has_prior):
            prior_ids = [str(s) for s, p in zip(stor_ids.values, has_prior, strict=True) if p]
            self.m.add_constraints(
                self.prior_storage_level.sel(storage=prior_ids) == ds.prior_level.sel(storage=prior_ids),
                name='storage_prior_fix',
            )

        # --- Final level bounds: E̲^end <= E[last] <= Ē^end (per period) ---
        final_level = level_end
        if ds.final_level_min is not None:
            self.m.add_constraints(
                final_level >= ds.final_level_min, name='storage_final_min', mask=ds.final_level_min.notnull()
            )
        if ds.final_level_max is not None:
            self.m.add_constraints(
                final_level <= ds.final_level_max, name='storage_final_max', mask=ds.final_level_max.notnull()
            )

        # --- Prevent simultaneous charge and discharge ---
        # Binary b per timestep: P^c <= M^c·b, P^d <= M^d·(1-b).
        # M is the static flow size bound (fixed size or sizing/invest max).
        if ds.prevent_simultaneous is not None:
            prevent = ds.prevent_simultaneous.values.astype(bool)
            prev_ids = [str(s) for s, p in zip(stor_vals, prevent, strict=True) if p]
            stor_coord = xr.DataArray(prev_ids, dims=['storage'])
            charging_on = self.m.add_variables(
                binary=True,
                coords={'storage': stor_coord, **self.data.dims.coords(time=True)},
                name=Var.STORAGE_CHARGING,
            )
            m_c = xr.DataArray(
                [self._flow_size_upper(str(ds.charge_flow.sel(storage=s).values)) for s in prev_ids],
                dims=['storage'],
                coords={'storage': prev_ids},
            )
            m_d = xr.DataArray(
                [self._flow_size_upper(str(ds.discharge_flow.sel(storage=s).values)) for s in prev_ids],
                dims=['storage'],
                coords={'storage': prev_ids},
            )
            self.m.add_constraints(
                charge_rates.sel(storage=prev_ids) - m_c * charging_on <= 0,
                name='storage_no_simul_charge',
            )
            self.m.add_constraints(
                discharge_rates.sel(storage=prev_ids) + m_d * charging_on <= m_d,
                name='storage_no_simul_discharge',
            )

    def _flow_size_upper(self, fid: str) -> float:
        """Static upper bound on a flow's size: fixed value or sizing/invest max.

        Args:
            fid: Qualified flow id.
        """
        fds = self.data.flows
        v = fds.size.sel(flow=fid).values
        if not np.isnan(v):
            return float(v)
        if fds.sizing is not None and fid in fds.sizing.max.coords[Dim.SIZING_FLOW].values:
            return float(fds.sizing.max.sel(sizing_flow=fid).values)
        if fds.invest is not None and fid in fds.invest.max.coords[Dim.INVEST_FLOW].values:
            return float(fds.invest.max.sel(invest_flow=fid).values)
        # Element validation guards sized flows; reaching this means an invariant broke upstream.
        raise ValueError(f'Flow {fid!r} has no static size bound for big-M')  # pragma: no cover

    def _set_objective(self) -> None:
        """Set objective: minimize the sum of (period-weighted) effect totals.

        Objective = sum_k sum_p( ω[k,p] * effect_total[k,p] )

        ω falls back to global period_weights (or 1 in single-period).
        The built-in penalty effect enters at weight 1.0 unless named in
        ``objective`` (see :meth:`optimize`), so
        ``effects_per_*={'penalty': ...}`` works as soft steering without
        touching the tracked effects. Zero-weight effects are validated but
        contribute no term.
        """
        from fluxopt.elements import PENALTY_EFFECT_ID

        ds = self.data.effects
        obj_expr: Any = 0
        effect_ids = list(ds.total_min.coords['effect'].values)

        weights = dict(self._objective)
        if PENALTY_EFFECT_ID not in weights and PENALTY_EFFECT_ID in effect_ids:
            weights[PENALTY_EFFECT_ID] = 1.0
        self._objective_weights = {k: float(v) for k, v in weights.items()}

        for k, weight in weights.items():
            if k not in effect_ids:
                raise ValueError(f'Objective effect {k!r} not found. Available: {effect_ids}')
            if weight == 0:
                continue

            # Resolve per-effect weight, falling back to global period_weights, then 1
            w: xr.DataArray | int = 1
            if ds.period_weights is not None and not ds.period_weights.sel(effect=k).isnull().all():
                w = ds.period_weights.sel(effect=k)
            elif self.data.dims.period_weights is not None:
                w = self.data.dims.period_weights

            obj_expr = obj_expr + weight * (w * self.effect_total.sel(effect=k)).sum()

        self.m.add_objective(obj_expr)
