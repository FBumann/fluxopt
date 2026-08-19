"""Relational backend: build and solve fluxopt's math as a streaming program.

fluxopt's three layers are Elements -> ModelData -> Model. This package is a
second implementation of that last step. Where
:class:`~fluxopt.model.FlowSystemModel` builds a linopy model in memory, this
one binds the same :class:`~fluxopt.model_data.ModelData` to a declarative
YAML program (:data:`~fluxopt.relational.sources.MATH_PROGRAM`) executed by
`lpspec <https://github.com/fluxopt/lpspec>`_, which builds the model
relationally and hands it to the solver without an LP file in between.

Both backends answer with the same :class:`~fluxopt.results.Result`, so
``tests/test_relational_parity.py`` compares answers rather than shapes: it
solves the same ``ModelData`` through each and requires the objectives to
agree, with a solution that never beats the linopy lane's proven optimum.

It is the *second* implementation only for as long as the migration runs.
``docs/design/lpspec-direction.md`` is where that ends: this program becomes
fluxopt's only math and ``model.py`` is deleted rather than ported, which is
also what retires the parity test.

lpspec is pinned to a tag — its language surface is pre-1.0 and still moves, so
an unpinned ref would let a ``uv sync`` change what the program means.

Every feature the element layer can express, this program expresses. The two
that needed ``at(x, by=map)`` — an indexed lookup, the adjoint of ``sum(by=)``
— are component-level status, where a flow's own ``Status`` and a component's
share one ``status_entity`` axis, and piecewise conversion, where a curve's
weights are read back onto each of its flows.

One refusal is left, and it is a difference of formulation rather than a gap:
``PiecewiseConversion.method='lp'`` is linopy's tangent-line *relaxation*, and
this lane has only the exact formulation, so it would answer a different
question. :class:`UnsupportedFeatureError` says so rather than quietly
answering it.
"""

from fluxopt.relational.results import to_result
from fluxopt.relational.solve import solve
from fluxopt.relational.sources import MATH_PROGRAM, PERIOD_PARAMS, UnsupportedFeatureError, build_sources

__all__ = [
    'MATH_PROGRAM',
    'PERIOD_PARAMS',
    'UnsupportedFeatureError',
    'build_sources',
    'solve',
    'to_result',
]
