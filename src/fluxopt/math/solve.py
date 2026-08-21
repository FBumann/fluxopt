"""Run a :class:`~fluxopt.model_data.ModelData` through the streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lpspec

from fluxopt.math.results import objective_weights, to_result
from fluxopt.math.sources import PROGRAM, build_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fluxopt.model_data import ModelData
    from fluxopt.results import Result


def solve(
    data: ModelData,
    objective: str | dict[str, float],
    *,
    solver_name: str = 'highs',
    solver_options: Mapping[str, Any] | None = None,
    math: Any = None,
    dimensions: Mapping[str, Any] | None = None,
    lookups: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Result:
    """Build *data* as a streaming program and solve it.

    Args:
        data: The model data. The same object the linopy backend consumes.
        objective: Effect id, or effect ids mapped to objective weights.
        solver_name: ``highs``, which ships with lpspec, or ``gurobi``, which
            needs lpspec's ``[gurobi]`` extra.
        solver_options: Forwarded to the solver verbatim, in *its own*
            vocabulary rather than linopy's (``{'mip_rel_gap': 1e-9}`` is
            HiGHS's own spelling).
        math: An edited program to solve instead of the shipped one — the
            model returned by :meth:`~fluxopt.flow_system.FlowSystem.math`,
            with whatever the caller added to it. It goes through validation
            and lowering exactly as a file does.
        dimensions: Labels for the dimensions *math* adds, as
            ``{name: labels}``.
        lookups: Data for the lookups *math* adds, as ``{name: frame}`` with
            the ``over`` dimension and the space its values are labels of,
            holding one row per label it maps and none for a label it does
            not.
        parameters: Data for the parameters *math* adds, as
            ``{name: frame}`` with the declared dims and a ``value`` column.

            The three are the language's own declaration blocks, and
            :class:`~fluxopt.math.parameters.Parameters` reports them under
            the same three names — so what a caller reads and what they supply
            use one vocabulary.

            None of the three may overwrite a name the program already binds:
            a caller who could silently replace ``rate_max`` could change the
            model without editing it.

    Returns:
        The same :class:`~fluxopt.results.Result` the linopy lane returns.

    Raises:
        UnsupportedFeatureError: If *data* uses a feature the program does not
            express yet.
        RuntimeError: If the solve produced no primal solution.
    """
    weights = objective_weights(data, objective)
    sources, coords = build_sources(data, weights)
    bound = {**sources, **coords}
    # All three are source keys to the binder, so they merge alike here and are
    # named apart only at the call site — which is what the language declares
    # and what `system.parameters()` reports.
    supplied = {**(dimensions or {}), **(lookups or {}), **(parameters or {})}
    if supplied:
        if clashes := sorted(set(supplied) & set(bound)):
            msg = f"these names are the program's own and cannot be supplied: {clashes}"
            raise ValueError(msg)
        bound |= supplied
    program = lpspec.load_model(PROGRAM if math is None else math)
    solved = lpspec.solve(program, bound, solver_name, solver_options=solver_options)
    return to_result(solved, data, weights, program)
