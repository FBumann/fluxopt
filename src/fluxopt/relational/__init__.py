"""Relational backend: build and solve fluxopt's math as a streaming program.

fluxopt's three layers are Elements -> ModelData -> Model. This package is a
second implementation of that last step. Where
:class:`~fluxopt.model.FlowSystemModel` builds a linopy model in memory, this
one binds the same :class:`~fluxopt.model_data.ModelData` to a declarative
YAML program (:data:`~fluxopt.relational.sources.MATH_PROGRAM`) executed by
`lpspec <https://github.com/fluxopt/lpspec>`_, which builds the model
relationally and hands it to the solver without an LP file in between.

Both backends are held to the same answer: ``tests/test_relational_parity.py``
solves the same ``ModelData`` through each and requires the objectives to
agree, with a solution that never beats the linopy lane's proven optimum.

It is the *second* implementation only for as long as the migration runs.
``docs/design/lpspec-direction.md`` is where that ends: this program becomes
fluxopt's only math and ``model.py`` is deleted rather than ported, which is
also what retires the parity test.

lpspec is pinned to a tag — its language surface is pre-1.0 and still moves, so
an unpinned ref would let a ``uv sync`` change what the program means.

Piecewise conversion and component-level status are not expressed yet, and
raise :class:`UnsupportedFeatureError` rather than being dropped: a missing
constraint still solves, just to the wrong answer. Both need ``at(x, by=map)``
— an indexed lookup, the adjoint of ``sum(by=)`` — which the pinned lpspec has;
porting them is the next step of the migration.
"""

from fluxopt.relational.solve import solve
from fluxopt.relational.sources import MATH_PROGRAM, PERIOD_PARAMS, UnsupportedFeatureError, build_sources

__all__ = [
    'MATH_PROGRAM',
    'PERIOD_PARAMS',
    'UnsupportedFeatureError',
    'build_sources',
    'solve',
]
