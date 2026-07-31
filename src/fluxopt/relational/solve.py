"""Run a :class:`~fluxopt.model_data.ModelData` through the streaming engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fluxopt.relational.sources import MATH_PROGRAM, build_sources

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fluxopt.model_data import ModelData

_MISSING = (
    "the relational backend needs 'farkas', which is not installed — install it with `uv sync --extra relational`"
)


def solve(
    data: ModelData,
    objective: str | dict[str, float],
    *,
    memory_limit: str = '2GB',
    solver_options: Mapping[str, Any] | None = None,
    **build_kwargs: Any,
) -> Any:
    """Build *data* as a streaming program and solve it.

    Args:
        data: The model data. The same object the linopy backend consumes.
        objective: Effect id, or effect ids mapped to objective weights.
        memory_limit: Hard build-memory budget for the engine. Peak build RAM
            follows this rather than the size of the model.
        solver_options: Forwarded verbatim to the solver, in linopy's shape
            (``{'mip_rel_gap': 1e-9}``).
        **build_kwargs: Passed through to ``farkas.solve``.

    Returns:
        The ``farkas`` result: ``.objective``, ``.status``, ``.primal(name)``
        and ``.to_dataarray(name)``. Not yet a :class:`~fluxopt.results.Result`
        — effect contributions and duals are not ported.

    Raises:
        ImportError: If `farkas` is not installed.
        UnsupportedFeatureError: If *data* uses a feature the program does not
            express yet.
    """
    try:
        import farkas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING) from exc

    weights = {objective: 1.0} if isinstance(objective, str) else dict(objective)
    sources, coords = build_sources(data, weights)
    return farkas.solve(
        MATH_PROGRAM,
        sources=sources,
        coords=coords,
        memory_limit=memory_limit,
        solver_options=solver_options,
        **build_kwargs,
    )
