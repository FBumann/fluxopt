"""Run a :class:`~fluxopt.model_data.ModelData` through the streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lpspec

from fluxopt.relational.results import objective_weights, to_result
from fluxopt.relational.sources import MATH_PROGRAM, build_sources

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

    Returns:
        The same :class:`~fluxopt.results.Result` the linopy lane returns.

    Raises:
        UnsupportedFeatureError: If *data* uses a feature the program does not
            express yet.
        RuntimeError: If the solve produced no primal solution.
    """
    weights = objective_weights(data, objective)
    sources, coords = build_sources(data, weights)
    solved = lpspec.solve(
        MATH_PROGRAM,
        {**sources, **coords},
        solver_name,
        solver_options=solver_options,
    )
    return to_result(solved, data, weights)
