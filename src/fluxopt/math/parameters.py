"""The numbers a system binds, as an artifact.

fluxopt's math is a file. Its numbers were not — they were computed at every
solve and thrown away, which left the two halves of a solve with unequal
standing: a caller could read, edit and typeset the equations, then supply
data for a constraint of their own through a channel whose existing contents
were invisible to them.

:class:`Parameters` is the other half. It is what ``build_sources`` already
produces, given a name, a schema and a directory to live in — so
``system.math()`` and ``system.parameters()`` answer the same question about
the two halves, and ``(program.yaml, parameters/)`` is the whole problem.

**Derived, never authored.** The tables are persistable, not a writing
surface: the Leontief fold and the size pre-scaling have already dissolved
what the user declared into what the solver adds up, so an edit here is an
edit nobody can review. The authoring surfaces stay the element layer for
data and the program for math. See docs/design/parameters-as-artifact.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: What a pandas column's dtype means in polars. `build_sources` still emits
#: some tables through pandas, whose empty columns type as `float64` and read
#: as a numeric label space; the mapping is what `_empty` meant to say.
_DTYPES: dict[str, Any] = {'O': pl.String, 'b': pl.Boolean, 'i': pl.Int64, 'u': pl.Int64, 'f': pl.Float64}


def _as_frame(value: Any, name: str) -> pl.DataFrame:
    """One source table as polars, whatever `build_sources` handed back.

    A bare label array is a dimension's index, which is a one-column table
    like any other — storing it as one is what lets the whole set round-trip
    through a single format.
    """
    if isinstance(value, pl.DataFrame):
        return value
    if isinstance(value, pd.DataFrame):
        schema = {c: _DTYPES[value[c].dtype.kind] for c in value.columns}
        return pl.DataFrame({c: value[c].to_numpy() for c in value.columns}, schema=schema)
    labels = np.asarray(value)
    dtype = _DTYPES.get(labels.dtype.kind, pl.String)
    return pl.DataFrame({name: labels}, schema={name: dtype})


@dataclass(frozen=True)
class Parameters:
    """Every table a solve binds, in the three kinds the program declares.

    `dimensions:`, `lookups:` and `parameters:` are the language's own three
    blocks, and they are three different things: a dimension is a label set, a
    lookup is a map from one to another, and a parameter is a function of the
    dims it names. The binder hands lookups over as columns on the index table
    of the dimension they run over, which is what lpspec reads — but that is a
    transport detail, and an artifact that flattened the three into two would
    be keeping the transport instead of the meaning.
    """

    #: Dimension name -> its labels, one column named for the dimension
    dimensions: dict[str, pl.DataFrame]
    #: Lookup name -> `(over, into)` — or `(over, <name>)` where the lookup
    #: owns its label space rather than naming another dimension. One row per
    #: label the map is defined at: partial by nature, so a flow that charges
    #: no storage is simply absent rather than present and null.
    lookups: dict[str, pl.DataFrame]
    #: Parameter name -> its tidy `(dims..., value)` table
    parameters: dict[str, pl.DataFrame]

    def __getitem__(self, name: str) -> pl.DataFrame:
        """One table by name, of whichever kind."""
        for group in (self.parameters, self.lookups, self.dimensions):
            if name in group:
                return group[name]
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return any(name in g for g in (self.parameters, self.lookups, self.dimensions))

    @property
    def rows(self) -> int:
        """Total rows across every table — the size of the bound problem."""
        groups = (self.dimensions, self.lookups, self.parameters)
        return sum(len(f) for g in groups for f in g.values())

    @classmethod
    def of(cls, data: ModelData, objective: str | dict[str, float], math: Any = None) -> Parameters:
        """Bind *data* to a program and keep what was bound, split by kind.

        The program is what says which columns of an index table are lookups
        and what they map into, so the split is read off the declarations
        rather than guessed from the names.

        Args:
            data: The model data to bind.
            objective: Effect id, or effect ids mapped to objective weights —
                what :func:`~fluxopt.math.solve.solve` takes. The weights are
                a bound parameter like any other, so the set is not complete
                without knowing what is being minimised.
            math: The program to bind against. Defaults to the shipped one.
        """
        import lpspec

        from fluxopt.math.results import objective_weights
        from fluxopt.math.sources import PROGRAM, build_sources

        model = math if math is not None else lpspec.load_model(PROGRAM)
        sources, coords = build_sources(data, objective_weights(data, objective))

        # A lookup either names the dimension its values are labels of, or
        # owns an inline label space and targets nothing. The first is a
        # relation between two dimensions and reads best keyed by both; the
        # second has only its own name to be keyed by.
        declared: dict[str, list[tuple[str, str]]] = {}
        for name, lookup in model.lookups.items():
            declared.setdefault(lookup.over, []).append((name, lookup.into or name))

        dimensions: dict[str, pl.DataFrame] = {}
        lookups: dict[str, pl.DataFrame] = {}
        for dim, value in coords.items():
            frame = _as_frame(value, dim)
            dimensions[dim] = frame.select(dim)
            for name, into in declared.get(dim, []):
                if name not in frame.columns:
                    continue
                # A map is defined where it is defined: a flow that charges no
                # storage has no row here, rather than a row saying so.
                lookups[name] = frame.select([dim, pl.col(name).alias(into)]).drop_nulls()

        return cls(
            dimensions=dimensions,
            lookups=lookups,
            parameters={k: _as_frame(v, k) for k, v in sources.items()},
        )

    def save(self, path: str | Path) -> None:
        """Write the set as a directory of parquet, one file per table.

        Parquet rather than YAML for the reason the math is YAML: each format
        is asked what it is good at. A million-row table has no useful line
        diff and no dtype of its own in YAML, and it is the dtypes that the
        binder checks against the program's declarations.

        Args:
            path: Directory to write into. Created if absent.
        """
        root = Path(path)
        for group, frames in (
            ('dimensions', self.dimensions),
            ('lookups', self.lookups),
            ('parameters', self.parameters),
        ):
            (root / group).mkdir(parents=True, exist_ok=True)
            for name, frame in frames.items():
                frame.write_parquet(root / group / f'{name}.parquet')

    @classmethod
    def load(cls, path: str | Path) -> Parameters:
        """Read a set written by :meth:`save`.

        Args:
            path: The directory :meth:`save` wrote.

        Raises:
            OSError: If the directory holds no parameter set.
        """
        root = Path(path)
        if not (root / 'parameters').is_dir():
            raise OSError(f'No fluxopt parameter set found in {root} (missing parameters/)')

        def read(group: str) -> dict[str, pl.DataFrame]:
            return {f.stem: pl.read_parquet(f) for f in sorted((root / group).glob('*.parquet'))}

        return cls(dimensions=read('dimensions'), lookups=read('lookups'), parameters=read('parameters'))
