"""Relational backend: build and solve fluxopt's math as a streaming program.

fluxopt's three layers are Elements -> ModelData -> Model. This package is a
second implementation of that last step. Where
:class:`~fluxopt.model.FlowSystemModel` builds a linopy model in memory, this
one binds the same :class:`~fluxopt.model_data.ModelData` to a declarative
YAML program (:data:`~fluxopt.relational.sources.MATH_PROGRAM`) executed by
`farkas <https://github.com/FBumann/linopy-yaml>`_, which streams the model to
the solver under a fixed memory budget.

Both backends are held to the same answer: ``tests/test_relational_parity.py``
solves the same ``ModelData`` through each and requires the objectives to
agree, with a solution that never beats the linopy lane's proven optimum.

`farkas` is an optional dependency::

    uv sync --extra relational

Not every feature is expressed yet — investment, piecewise and component
status raise :class:`UnsupportedFeatureError` rather than being dropped.
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
