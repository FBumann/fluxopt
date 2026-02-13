from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from fluxopt.types import Timesteps, compute_dt, compute_end_time, normalize_timesteps, to_polars_series

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Bus, Effect, Flow, Storage


@dataclass
class FlowsTable:
    index: pl.DataFrame  # (flow: str)
    sizes: pl.DataFrame  # (flow, size) — size is null for unsized flows
    relative_bounds: pl.DataFrame  # (flow, time, rel_lb, rel_ub)
    fixed: pl.DataFrame  # (flow, time, value) — relative profile values for fixed flows
    effect_coefficients: pl.DataFrame  # (flow, effect, time, coeff)

    def validate(self) -> None:
        from fluxopt.validation import validate_flow_bounds

        validate_flow_bounds(self.relative_bounds)

    @classmethod
    def from_elements(cls, flows: list[Flow], timesteps: pl.Series) -> FlowsTable:
        flow_ids = [f.id for f in flows]
        index = pl.DataFrame({'flow': flow_ids})
        time_dtype = timesteps.dtype

        sizes_rows: list[dict[str, Any]] = []
        bounds_rows: list[dict[str, Any]] = []
        fixed_rows: list[dict[str, Any]] = []
        coeff_rows: list[dict[str, Any]] = []

        for f in flows:
            sizes_rows.append({'flow': f.id, 'size': f.size})

            lb_series = to_polars_series(f.relative_minimum, timesteps, 'rel_lb')
            ub_series = to_polars_series(f.relative_maximum, timesteps, 'rel_ub')

            for i, t in enumerate(timesteps):
                bounds_rows.append(
                    {
                        'flow': f.id,
                        'time': t,
                        'rel_lb': float(lb_series[i]),
                        'rel_ub': float(ub_series[i]),
                    }
                )

            if f.fixed_relative_profile is not None:
                profile = to_polars_series(f.fixed_relative_profile, timesteps, 'value')
                for i, t in enumerate(timesteps):
                    fixed_rows.append({'flow': f.id, 'time': t, 'value': float(profile[i])})

            for effect_label, factor in f.effects_per_flow_hour.items():
                factor_series = to_polars_series(factor, timesteps, 'coeff')
                for i, t in enumerate(timesteps):
                    coeff_rows.append(
                        {
                            'flow': f.id,
                            'effect': effect_label,
                            'time': t,
                            'coeff': float(factor_series[i]),
                        }
                    )

        sizes = pl.DataFrame(sizes_rows, schema={'flow': pl.String, 'size': pl.Float64})
        relative_bounds = pl.DataFrame(
            bounds_rows,
            schema={'flow': pl.String, 'time': time_dtype, 'rel_lb': pl.Float64, 'rel_ub': pl.Float64},
        )
        fixed = pl.DataFrame(fixed_rows, schema={'flow': pl.String, 'time': time_dtype, 'value': pl.Float64})
        effect_coefficients = pl.DataFrame(
            coeff_rows,
            schema={'flow': pl.String, 'effect': pl.String, 'time': time_dtype, 'coeff': pl.Float64},
        )
        return cls(
            index=index,
            sizes=sizes,
            relative_bounds=relative_bounds,
            fixed=fixed,
            effect_coefficients=effect_coefficients,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        timesteps: pl.Series,
        effect_coefficients: pl.DataFrame | None = None,
    ) -> FlowsTable:
        """df schema: (flow, bus, size, rel_min, rel_max, [fixed_profile columns])"""
        flow_labels = df['flow'].to_list()
        index = pl.DataFrame({'flow': flow_labels})
        time_dtype = timesteps.dtype

        sizes_rows: list[dict[str, Any]] = []
        bounds_rows: list[dict[str, Any]] = []
        fixed_rows: list[dict[str, Any]] = []
        for row in df.iter_rows(named=True):
            size = row.get('size')
            rel_min = row.get('rel_min', 0.0)
            rel_max = row.get('rel_max', 1.0)
            if size is None and rel_max != 1.0:
                msg = f"Flow '{row['flow']}': rel_max={rel_max} has no effect without a size"
                raise ValueError(msg)
            sizes_rows.append({'flow': row['flow'], 'size': size})
            bounds_rows.extend(
                {
                    'flow': row['flow'],
                    'time': t,
                    'rel_lb': float(rel_min),
                    'rel_ub': float(rel_max),
                }
                for t in timesteps
            )

        sizes = pl.DataFrame(sizes_rows, schema={'flow': pl.String, 'size': pl.Float64})
        relative_bounds = pl.DataFrame(
            bounds_rows,
            schema={'flow': pl.String, 'time': time_dtype, 'rel_lb': pl.Float64, 'rel_ub': pl.Float64},
        )
        fixed = pl.DataFrame(fixed_rows, schema={'flow': pl.String, 'time': time_dtype, 'value': pl.Float64})
        if effect_coefficients is None:
            effect_coefficients = pl.DataFrame(
                schema={'flow': pl.String, 'effect': pl.String, 'time': time_dtype, 'coeff': pl.Float64}
            )
        return cls(
            index=index,
            sizes=sizes,
            relative_bounds=relative_bounds,
            fixed=fixed,
            effect_coefficients=effect_coefficients,
        )


@dataclass
class BusesTable:
    index: pl.DataFrame  # (bus: str)
    flow_coefficients: pl.DataFrame  # (bus, flow, coeff)

    @classmethod
    def from_elements(cls, buses: list[Bus], flows: list[Flow]) -> BusesTable:
        index = pl.DataFrame({'bus': [b.id for b in buses]})

        rows: list[dict[str, Any]] = []
        for flow in flows:
            # Output to bus: +1, Input from bus: -1
            coeff = -1.0 if flow._is_input else 1.0
            rows.append({'bus': flow.bus, 'flow': flow.id, 'coeff': coeff})

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
    def from_elements(cls, converters: list[Converter], timesteps: pl.Series) -> ConvertersTable:
        index = pl.DataFrame({'converter': [c.id for c in converters]})
        time_dtype = timesteps.dtype

        rows: list[dict[str, Any]] = []
        for conv in converters:
            for eq_idx, equation in enumerate(conv.conversion_factors):
                # Check if any factor is time-varying
                is_time_varying = any(not isinstance(v, (int, float)) for v in equation.values())

                if is_time_varying:
                    for flow_obj, factor in equation.items():
                        factor_series = to_polars_series(factor, timesteps, 'coeff')
                        for i, t in enumerate(timesteps):
                            rows.append(
                                {
                                    'converter': conv.id,
                                    'eq_idx': eq_idx,
                                    'flow': flow_obj.id,
                                    'time': t,
                                    'coeff': float(factor_series[i]),
                                }
                            )
                else:
                    for flow_obj, factor in equation.items():
                        scalar = float(factor)  # type: ignore[arg-type]  # guarded by is_time_varying check
                        rows.extend(
                            {
                                'converter': conv.id,
                                'eq_idx': eq_idx,
                                'flow': flow_obj.id,
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
                'time': time_dtype,
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
    objective_effect: str
    bounds: pl.DataFrame  # (effect, min_total, max_total)
    time_bounds_lb: pl.DataFrame  # (effect, time, value) — only rows where min_per_hour is set
    time_bounds_ub: pl.DataFrame  # (effect, time, value) — only rows where max_per_hour is set

    @classmethod
    def from_elements(cls, effects: list[Effect], timesteps: pl.Series) -> EffectsTable:
        index = pl.DataFrame({'effect': [e.id for e in effects]})
        time_dtype = timesteps.dtype

        objective_effects = [e for e in effects if e.is_objective]
        if not objective_effects:
            msg = 'No objective effect found. Include an Effect with is_objective=True.'
            raise ValueError(msg)
        objective_effect = objective_effects[0].id

        # Bounds
        bounds_rows: list[dict[str, Any]] = []
        tb_lb_rows: list[dict[str, Any]] = []
        tb_ub_rows: list[dict[str, Any]] = []
        for e in effects:
            bounds_rows.append(
                {
                    'effect': e.id,
                    'min_total': e.minimum_total,
                    'max_total': e.maximum_total,
                }
            )
            if e.minimum_per_hour is not None:
                min_ph = to_polars_series(e.minimum_per_hour, timesteps, 'value')
                for i, t in enumerate(timesteps):
                    tb_lb_rows.append({'effect': e.id, 'time': t, 'value': float(min_ph[i])})
            if e.maximum_per_hour is not None:
                max_ph = to_polars_series(e.maximum_per_hour, timesteps, 'value')
                for i, t in enumerate(timesteps):
                    tb_ub_rows.append({'effect': e.id, 'time': t, 'value': float(max_ph[i])})

        bounds = pl.DataFrame(
            bounds_rows, schema={'effect': pl.String, 'min_total': pl.Float64, 'max_total': pl.Float64}
        )
        time_bounds_lb = pl.DataFrame(
            tb_lb_rows,
            schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64},
        )
        time_bounds_ub = pl.DataFrame(
            tb_ub_rows,
            schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64},
        )
        return cls(
            index=index,
            objective_effect=objective_effect,
            bounds=bounds,
            time_bounds_lb=time_bounds_lb,
            time_bounds_ub=time_bounds_ub,
        )

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, time_dtype: pl.DataType | None = None) -> EffectsTable:
        """df: (effect, is_objective, min_total, max_total)"""
        index = pl.DataFrame({'effect': df['effect']})
        obj_row = df.filter(pl.col('is_objective') == True)  # noqa: E712
        objective_effect = obj_row['effect'][0]
        bounds = df.select('effect', 'min_total', 'max_total')
        if time_dtype is None:
            time_dtype = pl.Datetime()
        time_bounds_lb = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64})
        time_bounds_ub = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64})
        return cls(
            index=index,
            objective_effect=objective_effect,
            bounds=bounds,
            time_bounds_lb=time_bounds_lb,
            time_bounds_ub=time_bounds_ub,
        )


@dataclass
class StoragesTable:
    index: pl.DataFrame  # (storage: str)
    params: pl.DataFrame  # (storage, capacity, initial_charge, cyclic)
    time_params: pl.DataFrame  # (storage, time, eta_c, eta_d, loss)
    flow_map: pl.DataFrame  # (storage, charge_flow, discharge_flow)
    cs_bounds: pl.DataFrame  # (storage, time, cs_lb, cs_ub) — absolute values

    def validate(self) -> None:
        from fluxopt.validation import validate_storage_params, validate_storage_time_params

        validate_storage_params(self.params)
        validate_storage_time_params(self.time_params)

    @classmethod
    def from_elements(cls, storages: list[Storage], timesteps: pl.Series) -> StoragesTable:
        index = pl.DataFrame({'storage': [s.id for s in storages]})
        time_dtype = timesteps.dtype

        params_rows: list[dict[str, Any]] = []
        time_params_rows: list[dict[str, Any]] = []
        flow_map_rows: list[dict[str, Any]] = []
        cs_bounds_rows: list[dict[str, Any]] = []

        for s in storages:
            cyclic = s.initial_charge_state == 'cyclic'
            initial = 0.0 if cyclic else (float(s.initial_charge_state) if s.initial_charge_state is not None else 0.0)

            params_rows.append(
                {
                    'storage': s.id,
                    'capacity': s.capacity,
                    'initial_charge': initial,
                    'cyclic': cyclic,
                }
            )

            flow_map_rows.append(
                {
                    'storage': s.id,
                    'charge_flow': s.charging.id,
                    'discharge_flow': s.discharging.id,
                }
            )

            eta_c = to_polars_series(s.eta_charge, timesteps, 'eta_c')
            eta_d = to_polars_series(s.eta_discharge, timesteps, 'eta_d')
            loss = to_polars_series(s.relative_loss_per_hour, timesteps, 'loss')
            cs_lb_rel = to_polars_series(s.relative_minimum_charge_state, timesteps, 'cs_lb')
            cs_ub_rel = to_polars_series(s.relative_maximum_charge_state, timesteps, 'cs_ub')

            cap = s.capacity or 1e9

            for i, t in enumerate(timesteps):
                time_params_rows.append(
                    {
                        'storage': s.id,
                        'time': t,
                        'eta_c': float(eta_c[i]),
                        'eta_d': float(eta_d[i]),
                        'loss': float(loss[i]),
                    }
                )
                cs_bounds_rows.append(
                    {
                        'storage': s.id,
                        'time': t,
                        'cs_lb': float(cs_lb_rel[i]) * cap,
                        'cs_ub': float(cs_ub_rel[i]) * cap,
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
                'time': time_dtype,
                'eta_c': pl.Float64,
                'eta_d': pl.Float64,
                'loss': pl.Float64,
            },
        )
        flow_map = pl.DataFrame(
            flow_map_rows,
            schema={'storage': pl.String, 'charge_flow': pl.String, 'discharge_flow': pl.String},
        )
        cs_bounds = pl.DataFrame(
            cs_bounds_rows,
            schema={'storage': pl.String, 'time': time_dtype, 'cs_lb': pl.Float64, 'cs_ub': pl.Float64},
        )
        return cls(index=index, params=params, time_params=time_params, flow_map=flow_map, cs_bounds=cs_bounds)

    @classmethod
    def from_dataframe(
        cls, params_df: pl.DataFrame, time_params_df: pl.DataFrame, flow_map_df: pl.DataFrame
    ) -> StoragesTable:
        index = pl.DataFrame({'storage': params_df['storage']})
        # Infer time dtype from time_params_df
        time_dtype = time_params_df.schema.get('time', pl.Datetime())
        cs_bounds = pl.DataFrame(
            schema={'storage': pl.String, 'time': time_dtype, 'cs_lb': pl.Float64, 'cs_ub': pl.Float64}
        )
        return cls(index=index, params=params_df, time_params=time_params_df, flow_map=flow_map_df, cs_bounds=cs_bounds)


@dataclass
class ModelData:
    flows: FlowsTable
    buses: BusesTable
    converters: ConvertersTable
    effects: EffectsTable
    storages: StoragesTable
    timesteps: pl.DataFrame  # (time)
    dt: pl.DataFrame  # (time, dt: f64)
    weights: pl.DataFrame  # (time, weight: f64)
    charge_state_times: pl.DataFrame  # (time) — N+1 entries for storage


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> list[Flow]:
    flows: list[Flow] = []
    for port in ports:
        flows.extend(port.imports)
        flows.extend(port.exports)
    for conv in converters:
        flows.extend(conv.inputs)
        flows.extend(conv.outputs)
    for s in storages or []:
        flows.append(s.charging)
        flows.append(s.discharging)
    return flows


def build_model_data(
    timesteps: Timesteps,
    buses: list[Bus],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | pl.Series | None = None,
) -> ModelData:
    """Build ModelData from element objects."""
    from fluxopt.validation import validate_system

    converters = converters or []
    ts_series = normalize_timesteps(timesteps)
    dt_series = compute_dt(ts_series, dt)

    flows = _collect_flows(ports, converters, storages)
    validate_system(buses, effects, ports, converters, storages, flows)

    flows_table = FlowsTable.from_elements(flows, ts_series)
    buses_table = BusesTable.from_elements(buses, flows)
    converters_table = ConvertersTable.from_elements(converters, ts_series)
    effects_table = EffectsTable.from_elements(effects, ts_series)
    storages_table = StoragesTable.from_elements(storages or [], ts_series)

    flows_table.validate()
    storages_table.validate()

    timesteps_df = pl.DataFrame({'time': ts_series})
    dt_df = pl.DataFrame({'time': ts_series, 'dt': dt_series})
    weights_df = pl.DataFrame({'time': ts_series, 'weight': [1.0] * len(ts_series)})

    # Build charge_state_times: N+1 entries
    end_time = compute_end_time(ts_series, dt_series)
    cs_times = [*ts_series.to_list(), end_time]
    charge_state_times = pl.DataFrame({'time': pl.Series('time', cs_times, dtype=ts_series.dtype)})

    return ModelData(
        flows=flows_table,
        buses=buses_table,
        converters=converters_table,
        effects=effects_table,
        storages=storages_table,
        timesteps=timesteps_df,
        dt=dt_df,
        weights=weights_df,
        charge_state_times=charge_state_times,
    )
