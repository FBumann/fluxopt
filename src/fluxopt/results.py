from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from fluxopt.model import FlowSystemModel


@dataclass
class SolvedModel:
    objective_value: float
    flow_rates: pl.DataFrame  # (flow, time, value)
    charge_states: pl.DataFrame  # (storage, time, value)
    effects: pl.DataFrame  # (effect, total)
    effects_per_timestep: pl.DataFrame  # (effect, time, value)
    contributions: pl.DataFrame  # (source, contributor, effect, time, value)

    def flow_rate(self, id: str) -> pl.DataFrame:
        """Get time series for a single flow."""
        return self.flow_rates.filter(pl.col('flow') == id).select('time', 'value')

    def charge_state(self, id: str) -> pl.DataFrame:
        """Get time series for a single storage."""
        return self.charge_states.filter(pl.col('storage') == id).select('time', 'value')

    @classmethod
    def from_model(cls, model: FlowSystemModel) -> SolvedModel:
        m = model.m
        d = model.data
        time_dtype = d.timesteps.schema['time']

        # Extract flow rates
        flow_sol = m.flow_rate.solution.rename({'solution': 'value'})

        # Extract charge states (if storages exist)
        if len(d.storages.index) > 0:
            cs_sol = m.charge_state.solution.rename({'solution': 'value'})
        else:
            cs_sol = pl.DataFrame(schema={'storage': pl.String, 'time': time_dtype, 'value': pl.Float64})

        # Extract effects
        if len(d.effects.index) > 0:
            effect_total_sol = m.effect_total.solution.rename({'solution': 'total'})
            effect_ts_sol = m.effect_per_timestep.solution.rename({'solution': 'value'})
        else:
            effect_total_sol = pl.DataFrame(schema={'effect': pl.String, 'total': pl.Float64})
            effect_ts_sol = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64})

        # Extract per-source contributions
        contrib_frames: list[pl.DataFrame] = []
        for src in model._temporal_sources:
            var_name = f'contributions({src.name})'
            var = getattr(m, var_name)
            sol = var.solution.rename({'solution': 'value', src.sum_dim: 'contributor'})
            sol = sol.with_columns(pl.lit(src.name).alias('source'))
            contrib_frames.append(sol.select('source', 'contributor', 'effect', 'time', 'value'))

        if contrib_frames:
            contributions = pl.concat(contrib_frames)
        else:
            contributions = pl.DataFrame(
                schema={
                    'source': pl.String,
                    'contributor': pl.String,
                    'effect': pl.String,
                    'time': time_dtype,
                    'value': pl.Float64,
                }
            )

        # Objective value
        obj_effect = d.effects.objective_effect
        obj_val = effect_total_sol.filter(pl.col('effect') == obj_effect)['total'][0]

        return cls(
            objective_value=obj_val,
            flow_rates=flow_sol,
            charge_states=cs_sol,
            effects=effect_total_sol,
            effects_per_timestep=effect_ts_sol,
            contributions=contributions,
        )
