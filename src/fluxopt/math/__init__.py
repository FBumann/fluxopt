"""fluxopt's math, and the engine that builds it.

Three layers: Elements -> ModelData -> this. The math is
:data:`~fluxopt.math.sources.PROGRAM`, a declarative YAML program
built and solved by `lpspec <https://github.com/fluxopt/lpspec>`_;
:mod:`~fluxopt.math.sources` binds a ``ModelData`` to it, and
:mod:`~fluxopt.math.results` reads the answer back as a
:class:`~fluxopt.results.Result`.

There is no second implementation. ``model.py`` built the same math a second
time in linopy calls, and deleting it is what
``docs/design/lpspec-direction.md`` was written to argue for: the math is a
file now, so it is reviewed, diffed, typeset and extended as one.

lpspec is pinned to a tag — its language surface is pre-1.0 and still moves,
so an unpinned ref would let a ``uv sync`` change what the program means.

One refusal is left, and it is a difference of formulation rather than a gap:
``PiecewiseConversion.method='lp'`` is linopy's tangent-line *relaxation*, and
this program has only the exact formulation, so it would answer a different
question. :class:`UnsupportedFeatureError` says so rather than quietly
answering it.
"""

from fluxopt.math.parameters import Parameters
from fluxopt.math.results import to_result
from fluxopt.math.solve import solve
from fluxopt.math.sources import PERIOD_PARAMS, PROGRAM, UnsupportedFeatureError, build_sources, program

__all__ = [
    'PERIOD_PARAMS',
    'PROGRAM',
    'Parameters',
    'UnsupportedFeatureError',
    'build_sources',
    'program',
    'solve',
    'to_result',
]
