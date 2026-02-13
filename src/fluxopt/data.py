from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from fluxopt.types import to_polars_series

if TYPE_CHECKING:
    from fluxopt.components import LinearConverter, Sink, Source
    from fluxopt.elements import Bus, Effect, Flow, Storage


@dataclass
class FlowsTable:
    index: pl.DataFrame  # (flow: str)
    bounds: pl.DataFrame  # (flow, time, lb, ub)
    fixed: pl.DataFrame  # (flow, time, value) — only fixed-profile flows

    @classmethod
    def from_elements(cls, flows: list[Flow], timesteps: pl.Series) -> FlowsTable:
        flow_labels = [f.label for f in flows]
        index = pl.DataFrame({'flow': flow_labels})

        bounds_rows: list[dict[str, Any]] = []
        fixed_rows: list[dict[str, Any]] = []

        for f in flows:
            lb_series = to_polars_series(f.relative_minimum, timesteps, 'lb')
            ub_series = to_polars_series(f.relative_maximum, timesteps, 'ub')

            for i, t in enumerate(timesteps):
                lb = lb_series[i]
                ub = ub_series[i]
                if f.size is not None:
                    lb = lb * f.size
                    ub = ub * f.size
                else:
                    # No size: bounds are relative, use large M for upper
                    lb = lb * 1e9
                    ub = ub * 1e9
                bounds_rows.append({'flow': f.label, 'time': t, 'lb': float(lb), 'ub': float(ub)})

            if f.fixed_relative_profile is not None:
                profile = to_polars_series(f.fixed_relative_profile, timesteps, 'value')
                for i, t in enumerate(timesteps):
                    val = float(profile[i])
                    if f.size is not None:
                        val = val * f.size
                    fixed_rows.append({'flow': f.label, 'time': t, 'value': val})

        bounds = pl.DataFrame(
            bounds_rows, schema={'flow': pl.String, 'time': pl.String, 'lb': pl.Float64, 'ub': pl.Float64}
        )
        fixed = pl.DataFrame(fixed_rows, schema={'flow': pl.String, 'time': pl.String, 'value': pl.Float64})
        return cls(index=index, bounds=bounds, fixed=fixed)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, timesteps: pl.Series) -> FlowsTable:
        """df schema: (flow, bus, size, rel_min, rel_max, [fixed_profile columns])"""
        flow_labels = df['flow'].to_list()
        index = pl.DataFrame({'flow': flow_labels})

        bounds_rows: list[dict[str, Any]] = []
        fixed_rows: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            size = row.get('size')
            rel_min = row.get('rel_min', 0.0)
            rel_max = row.get('rel_max', 1.0)
            for t in timesteps:
                lb = rel_min * (size if size is not None else 1e9)
                ub = rel_max * (size if size is not None else 1e9)
                bounds_rows.append({'flow': row['flow'], 'time': t, 'lb': float(lb), 'ub': float(ub)})

        bounds = pl.DataFrame(
            bounds_rows, schema={'flow': pl.String, 'time': pl.String, 'lb': pl.Float64, 'ub': pl.Float64}
        )
        fixed = pl.DataFrame(fixed_rows, schema={'flow': pl.String, 'time': pl.String, 'value': pl.Float64})
        return cls(index=index, bounds=bounds, fixed=fixed)


@dataclass
class BusesTable:
    index: pl.DataFrame  # (bus: str)
    flow_coefficients: pl.DataFrame  # (bus, flow, coeff)

    @classmethod
    def from_elements(cls, buses: list[Bus], flows: list[Flow]) -> BusesTable:
        index = pl.DataFrame({'bus': [b.label for b in buses]})

        rows: list[dict[str, Any]] = []
        for flow in flows:
            # Output to bus: +1, Input from bus: -1
            coeff = -1.0 if flow._is_input else 1.0
            rows.append({'bus': flow.bus, 'flow': flow.label, 'coeff': coeff})

        flow_coefficients = pl.DataFrame(rows, schema={'bus': pl.String, 'flow': pl.String, 'coeff': pl.Float64})
        return cls(index=index, flow_coefficients=flow_coefficients)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> BusesTable:
        """df schema: (bus, flow, coeff)"""
        index = pl.DataFrame({'bus': df['bus'].unique().sort()})
        return cls(index=index, flow_coefficients=df)


@dataclass
class ConvertersTable:
    index: pl.DataFrame  # (converter: str)
    flow_coefficients: pl.DataFrame  # (converter, eq_idx, flow, coeff [,time])

    @classmethod
    def from_elements(cls, converters: list[LinearConverter], timesteps: pl.Series) -> ConvertersTable:
        index = pl.DataFrame({'converter': [c.label for c in converters]})

        rows: list[dict[str, Any]] = []
        for conv in converters:
            for eq_idx, equation in enumerate(conv.conversion_factors):
                # Check if any factor is time-varying
                is_time_varying = any(not isinstance(v, (int, float)) for v in equation.values())

                if is_time_varying:
                    for flow_label, factor in equation.items():
                        factor_series = to_polars_series(factor, timesteps, 'coeff')
                        for i, t in enumerate(timesteps):
                            rows.append(
                                {
                                    'converter': conv.label,
                                    'eq_idx': eq_idx,
                                    'flow': flow_label,
                                    'time': t,
                                    'coeff': float(factor_series[i]),
                                }
                            )
                else:
                    for flow_label, factor in equation.items():
                        scalar = float(factor)  # type: ignore[arg-type]  # guarded by is_time_varying check
                        rows.extend(
                            {
                                'converter': conv.label,
                                'eq_idx': eq_idx,
                                'flow': flow_label,
                                'time': t,
                                'coeff': scalar,
                            }
                            for t in timesteps
                        )

        flow_coefficients = pl.DataFrame(
            rows,
            schema={
                'converter': pl.String,
                'eq_idx': pl.Int64,
                'flow': pl.String,
                'time': pl.String,
                'coeff': pl.Float64,
            },
        )
        return cls(index=index, flow_coefficients=flow_coefficients)

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> ConvertersTable:
        """df schema: (converter, eq_idx, flow, coeff [,time])"""
        index = pl.DataFrame({'converter': df['converter'].unique().sort()})
        return cls(index=index, flow_coefficients=df)


@dataclass
class EffectsTable:
    index: pl.DataFrame  # (effect: str)
    flow_coefficients: pl.DataFrame  # (flow, effect, time, coeff)
    objective_effect: str
    bounds: pl.DataFrame  # (effect, min_total, max_total)
    time_bounds: pl.DataFrame  # (effect, time, min_per_hour, max_per_hour)

    @classmethod
    def from_elements(cls, effects: list[Effect], flows: list[Flow], timesteps: pl.Series) -> EffectsTable:
        index = pl.DataFrame({'effect': [e.label for e in effects]})

        objective_effects = [e for e in effects if e.is_objective]
        objective_effect = objective_effects[0].label

        # Flow-effect coefficients
        coeff_rows: list[dict[str, Any]] = []
        for flow in flows:
            for effect_label, factor in flow.effects_per_flow_hour.items():
                factor_series = to_polars_series(factor, timesteps, 'coeff')
                for i, t in enumerate(timesteps):
                    coeff_rows.append(
                        {
                            'flow': flow.label,
                            'effect': effect_label,
                            'time': t,
                            'coeff': float(factor_series[i]),
                        }
                    )

        flow_coefficients = pl.DataFrame(
            coeff_rows,
            schema={'flow': pl.String, 'effect': pl.String, 'time': pl.String, 'coeff': pl.Float64},
        )

        # Bounds
        bounds_rows: list[dict[str, Any]] = []
        time_bounds_rows: list[dict[str, Any]] = []
        for e in effects:
            bounds_rows.append(
                {
                    'effect': e.label,
                    'min_total': e.minimum_total,
                    'max_total': e.maximum_total,
                }
            )
            if e.minimum_per_hour is not None or e.maximum_per_hour is not None:
                min_ph = (
                    to_polars_series(e.minimum_per_hour, timesteps, 'min') if e.minimum_per_hour is not None else None
                )
                max_ph = (
                    to_polars_series(e.maximum_per_hour, timesteps, 'max') if e.maximum_per_hour is not None else None
                )
                for i, t in enumerate(timesteps):
                    time_bounds_rows.append(
                        {
                            'effect': e.label,
                            'time': t,
                            'min_per_hour': float(min_ph[i]) if min_ph is not None else None,
                            'max_per_hour': float(max_ph[i]) if max_ph is not None else None,
                        }
                    )

        bounds = pl.DataFrame(
            bounds_rows, schema={'effect': pl.String, 'min_total': pl.Float64, 'max_total': pl.Float64}
        )
        time_bounds = pl.DataFrame(
            time_bounds_rows,
            schema={'effect': pl.String, 'time': pl.String, 'min_per_hour': pl.Float64, 'max_per_hour': pl.Float64},
        )
        return cls(
            index=index,
            flow_coefficients=flow_coefficients,
            objective_effect=objective_effect,
            bounds=bounds,
            time_bounds=time_bounds,
        )

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, flow_effects_df: pl.DataFrame) -> EffectsTable:
        """df: (effect, is_objective, min_total, max_total)
        flow_effects_df: (flow, effect, time, coeff)"""
        index = pl.DataFrame({'effect': df['effect']})
        obj_row = df.filter(pl.col('is_objective') == True)  # noqa: E712
        objective_effect = obj_row['effect'][0]
        bounds = df.select('effect', 'min_total', 'max_total')
        time_bounds = pl.DataFrame(
            schema={'effect': pl.String, 'time': pl.String, 'min_per_hour': pl.Float64, 'max_per_hour': pl.Float64}
        )
        return cls(
            index=index,
            flow_coefficients=flow_effects_df,
            objective_effect=objective_effect,
            bounds=bounds,
            time_bounds=time_bounds,
        )


@dataclass
class StoragesTable:
    index: pl.DataFrame  # (storage: str)
    params: pl.DataFrame  # (storage, capacity, initial_charge, cyclic)
    time_params: pl.DataFrame  # (storage, time, eta_c, eta_d, loss, cs_lb, cs_ub)
    flow_map: pl.DataFrame  # (storage, charge_flow, discharge_flow)

    @classmethod
    def from_elements(cls, storages: list[Storage], timesteps: pl.Series) -> StoragesTable:
        index = pl.DataFrame({'storage': [s.label for s in storages]})

        params_rows: list[dict[str, Any]] = []
        time_params_rows: list[dict[str, Any]] = []
        flow_map_rows: list[dict[str, Any]] = []

        for s in storages:
            cyclic = s.initial_charge_state == 'cyclic'
            initial = 0.0 if cyclic else (float(s.initial_charge_state) if s.initial_charge_state is not None else 0.0)

            params_rows.append(
                {
                    'storage': s.label,
                    'capacity': s.capacity,
                    'initial_charge': initial,
                    'cyclic': cyclic,
                }
            )

            flow_map_rows.append(
                {
                    'storage': s.label,
                    'charge_flow': s.charging.label,
                    'discharge_flow': s.discharging.label,
                }
            )

            eta_c = to_polars_series(s.eta_charge, timesteps, 'eta_c')
            eta_d = to_polars_series(s.eta_discharge, timesteps, 'eta_d')
            loss = to_polars_series(s.relative_loss_per_hour, timesteps, 'loss')
            cs_lb = to_polars_series(s.relative_minimum_charge_state, timesteps, 'cs_lb')
            cs_ub = to_polars_series(s.relative_maximum_charge_state, timesteps, 'cs_ub')

            for i, t in enumerate(timesteps):
                time_params_rows.append(
                    {
                        'storage': s.label,
                        'time': t,
                        'eta_c': float(eta_c[i]),
                        'eta_d': float(eta_d[i]),
                        'loss': float(loss[i]),
                        'cs_lb': float(cs_lb[i]),
                        'cs_ub': float(cs_ub[i]),
                    }
                )

        params = pl.DataFrame(
            params_rows,
            schema={'storage': pl.String, 'capacity': pl.Float64, 'initial_charge': pl.Float64, 'cyclic': pl.Boolean},
        )
        time_params = pl.DataFrame(
            time_params_rows,
            schema={
                'storage': pl.String,
                'time': pl.String,
                'eta_c': pl.Float64,
                'eta_d': pl.Float64,
                'loss': pl.Float64,
                'cs_lb': pl.Float64,
                'cs_ub': pl.Float64,
            },
        )
        flow_map = pl.DataFrame(
            flow_map_rows,
            schema={'storage': pl.String, 'charge_flow': pl.String, 'discharge_flow': pl.String},
        )
        return cls(index=index, params=params, time_params=time_params, flow_map=flow_map)

    @classmethod
    def from_dataframe(
        cls, params_df: pl.DataFrame, time_params_df: pl.DataFrame, flow_map_df: pl.DataFrame
    ) -> StoragesTable:
        index = pl.DataFrame({'storage': params_df['storage']})
        return cls(index=index, params=params_df, time_params=time_params_df, flow_map=flow_map_df)


@dataclass
class ModelData:
    flows: FlowsTable
    buses: BusesTable
    converters: ConvertersTable
    effects: EffectsTable
    storages: StoragesTable
    timesteps: pl.DataFrame  # (time: str)
    dt: pl.DataFrame  # (time, dt: f64)


def _collect_flows(
    components: list[Source | Sink | LinearConverter],
    storages: list[Storage] | None,
) -> list[Flow]:
    from fluxopt.components import LinearConverter, Sink, Source

    flows: list[Flow] = []
    for comp in components:
        if isinstance(comp, Source):
            flows.extend(comp.outputs)
        elif isinstance(comp, Sink):
            flows.extend(comp.inputs)
        elif isinstance(comp, LinearConverter):
            flows.extend(comp.inputs)
            flows.extend(comp.outputs)
    if storages:
        for s in storages:
            flows.append(s.charging)
            flows.append(s.discharging)
    return flows


def build_model_data(
    timesteps: list[str] | pl.Series,
    buses: list[Bus],
    effects: list[Effect],
    components: list[Source | Sink | LinearConverter],
    storages: list[Storage] | None = None,
    dt: float | list[float] = 1.0,
) -> ModelData:
    """Build ModelData from element objects."""
    from fluxopt.components import LinearConverter
    from fluxopt.validation import validate_system

    ts_series = pl.Series('time', timesteps) if isinstance(timesteps, list) else timesteps.alias('time')

    flows = _collect_flows(components, storages)
    validate_system(buses, effects, components, storages, flows)

    converters = [c for c in components if isinstance(c, LinearConverter)]

    flows_table = FlowsTable.from_elements(flows, ts_series)
    buses_table = BusesTable.from_elements(buses, flows)
    converters_table = ConvertersTable.from_elements(converters, ts_series)
    effects_table = EffectsTable.from_elements(effects, flows, ts_series)
    storages_table = StoragesTable.from_elements(storages or [], ts_series)

    timesteps_df = pl.DataFrame({'time': ts_series})

    # Build dt DataFrame
    n = len(ts_series)
    if isinstance(dt, (int, float)):
        dt_values = [float(dt)] * n
    else:
        dt_values = [float(v) for v in dt]
        if len(dt_values) != n:
            msg = f'Length of dt ({len(dt_values)}) does not match length of ts_series ({n})'
            raise ValueError(msg)
    dt_df = pl.DataFrame({'time': ts_series, 'dt': dt_values})

    return ModelData(
        flows=flows_table,
        buses=buses_table,
        converters=converters_table,
        effects=effects_table,
        storages=storages_table,
        timesteps=timesteps_df,
        dt=dt_df,
    )
