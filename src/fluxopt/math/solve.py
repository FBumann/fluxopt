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
    parameters: Mapping[str, Any] | None = None,
    lookups: Mapping[str, Any] | None = None,
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
        parameters: Data for the parameters *math* adds. Merged with the
            program's own, which it may not overwrite: a caller who could
            silently replace `rate_max` could change the model without
            editing it.
        lookups: Data for the lookups *math* adds, as ``{name: frame}`` with
            the ``over`` dimension and the values. A lookup is a column on an
            index table rather than a source of its own, so it merges onto
            that table instead of arriving beside it — which is why it needs
            a channel and cannot go through *parameters*.

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
    if parameters:
        if clashes := sorted(set(parameters) & set(bound)):
            msg = f"these names are the program's own and cannot be supplied: {clashes}"
            raise ValueError(msg)
        bound |= dict(parameters)
    program = lpspec.load_model(PROGRAM if math is None else math)
    if lookups:
        bound |= _merged_lookups(lookups, bound, program)
    solved = lpspec.solve(program, bound, solver_name, solver_options=solver_options)
    return to_result(solved, data, weights, program)


def _merged_lookups(lookups: Mapping[str, Any], bound: dict[str, Any], program: Any) -> dict[str, Any]:
    """The index tables *lookups* land on, each with its column joined in.

    A lookup is not a source key — it travels as a column on the index of the
    dimension it runs over, which is why it cannot go through `parameters=`:
    supplying it there would mean handing back the whole index table, and that
    is a name the program owns.

    The one check here is the one lpspec cannot make. It knows what a lookup
    declares; it does not know which index tables *this* system built, so a
    lookup over a dimension that has none has nothing to land on.

    Args:
        lookups: Lookup name to a two-column frame: the ``over`` dimension,
            and the value it maps to. The value column is renamed to the
            lookup's own name, which is how an index table carries several
            lookups into the same dimension.
        bound: What is already bound, which is where the index tables are.
        program: The loaded program, which says what each lookup runs over.

    Raises:
        ValueError: If a name is not a declared lookup, or its ``over``
            dimension has no index table to merge onto.
    """
    import polars as pl

    merged: dict[str, Any] = {}
    for name, frame in lookups.items():
        if (declared := program.lookups.get(name)) is None:
            known = sorted(program.lookups)
            msg = f'{name!r} is not a lookup this program declares. Declared: {known}'
            raise ValueError(msg)
        over = declared.over
        index = merged.get(over, bound.get(over))
        if index is None:
            msg = f'lookup {name!r} runs over {over!r}, which this system has no index for'
            raise ValueError(msg)
        supplied = pl.DataFrame(frame)
        if supplied.width != 2 or over not in supplied.columns:
            msg = (
                f'lookup {name!r} needs a two-column table — {over!r} and the value it maps to — '
                f'and got {supplied.columns}'
            )
            raise ValueError(msg)
        # Keyed by the lookup's own name, not its target's: an index table may
        # carry several lookups into the same dimension.
        value = next(c for c in supplied.columns if c != over)
        merged[over] = pl.DataFrame(index).join(supplied.select([over, pl.col(value).alias(name)]), on=over, how='left')
    return merged
