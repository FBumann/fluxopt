"""NetCDF IO for fluxopt — serialize SolvedModel to/from NetCDF via xarray.

Requires the ``io`` extra: ``pip install fluxopt[io]``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import xarray as xr

    from fluxopt.results import SolvedModel
    from fluxopt.tables import ModelData


def _require_xarray() -> Any:
    """Import and return xarray, raising a helpful error if missing. Also checks netCDF4."""
    try:
        import xarray
    except ModuleNotFoundError:
        msg = "xarray is required for NetCDF IO. Install it with: pip install 'fluxopt[io]'"
        raise ModuleNotFoundError(msg) from None
    try:
        import netCDF4
    except ModuleNotFoundError:
        msg = "netCDF4 is required for NetCDF IO. Install it with: pip install 'fluxopt[io]'"
        raise ModuleNotFoundError(msg) from None
    _ = netCDF4  # ensure it's not flagged as unused
    return xarray


def _time_dtype_label(dtype: pl.DataType) -> str:
    """Return 'datetime' or 'int64' based on a Polars dtype."""
    if dtype.is_integer():
        return 'int64'
    return 'datetime'


# ---------------------------------------------------------------------------
# Polars <-> xarray helpers
# ---------------------------------------------------------------------------


def _polars_long_to_dataarray(
    df: pl.DataFrame,
    dims: list[str],
    value_col: str,
    name: str,
) -> xr.DataArray:
    """Convert a Polars long-format DataFrame to a sparse xarray DataArray (NaN fill)."""
    _xr = _require_xarray()
    if len(df) == 0:
        return _xr.DataArray(name=name)
    pdf = df.to_pandas()
    series = pdf.set_index(dims)[value_col]
    da: xr.DataArray = series.to_xarray()
    da.name = name
    return da


def _dataarray_to_polars_long(
    da: xr.DataArray,
    value_col: str,
    time_dtype: pl.DataType | None = None,
) -> pl.DataFrame:
    """Convert an xarray DataArray back to a Polars long-format DataFrame."""
    if da.size == 0 or da.dims == ():
        return pl.DataFrame()
    pdf = da.to_dataframe(name=value_col).reset_index()
    pdf = pdf.dropna(subset=[value_col])
    result: pl.DataFrame = pl.from_pandas(pdf)
    # Cast time columns back to the correct dtype
    if time_dtype is not None:
        for col in result.columns:
            if col == 'time':
                result = result.with_columns(pl.col(col).cast(time_dtype))
    return result


# ---------------------------------------------------------------------------
# Solution -> xarray Dataset
# ---------------------------------------------------------------------------


def solved_model_to_xarray(result: SolvedModel) -> xr.Dataset:
    """Convert solution data to an xarray Dataset (root group contents)."""
    _xr = _require_xarray()

    data_vars: dict[str, xr.DataArray] = {}

    # Flow rates (flow, time)
    if len(result.flow_rates) > 0:
        data_vars['flow_rates'] = _polars_long_to_dataarray(
            result.flow_rates, ['flow', 'time'], 'solution', 'flow_rates'
        )

    # Charge states (storage, time)
    if len(result.charge_states) > 0:
        da = _polars_long_to_dataarray(result.charge_states, ['storage', 'time'], 'solution', 'charge_states')
        # Rename time -> cs_time to distinguish from flow timesteps
        da = da.rename({'time': 'cs_time'})
        data_vars['charge_states'] = da

    # Effects (effect,)
    if len(result.effects) > 0:
        data_vars['effects'] = _polars_long_to_dataarray(result.effects, ['effect'], 'solution', 'effects')

    # Effects per timestep (effect, time)
    if len(result.effects_per_timestep) > 0:
        data_vars['effects_per_timestep'] = _polars_long_to_dataarray(
            result.effects_per_timestep, ['effect', 'time'], 'solution', 'effects_per_timestep'
        )

    # Contributions (source, contributor, effect, time)
    if len(result.contributions) > 0:
        data_vars['contributions'] = _polars_long_to_dataarray(
            result.contributions, ['source', 'contributor', 'effect', 'time'], 'solution', 'contributions'
        )

    ds = _xr.Dataset(data_vars)

    # Attrs
    ds.attrs['objective_value'] = result.objective_value
    if result.data is not None:
        ds.attrs['objective_effect'] = result.data.effects.objective_effect
        ds.attrs['time_dtype'] = _time_dtype_label(result.data.timesteps.schema['time'])
    else:
        # Infer from flow_rates time column
        time_dtype = result.flow_rates.schema.get('time', pl.Datetime())
        ds.attrs['time_dtype'] = _time_dtype_label(time_dtype)

    return ds


# ---------------------------------------------------------------------------
# Model data -> NetCDF group
# ---------------------------------------------------------------------------


def _write_model_data(data: ModelData, path: str | Path) -> None:
    """Append a 'model' group to an existing NetCDF file with model input data."""
    _xr = _require_xarray()

    data_vars: dict[str, xr.DataArray] = {}

    # dt and weight (time,)
    data_vars['dt'] = _polars_long_to_dataarray(data.dt, ['time'], 'dt', 'dt')
    data_vars['weight'] = _polars_long_to_dataarray(data.weights, ['time'], 'weight', 'weight')

    # charge_state_times -- store as a coordinate-only variable
    cs_times = data.charge_state_times['time'].to_list()
    data_vars['charge_state_times'] = _xr.DataArray(cs_times, dims=['cs_time'], name='charge_state_times')

    # Flow bounds (flow, time)
    if len(data.flows.relative_bounds) > 0:
        bounds = data.flows.relative_bounds
        lb_df = bounds.select('flow', 'time', pl.col('rel_lb').alias('value'))
        ub_df = bounds.select('flow', 'time', pl.col('rel_ub').alias('value'))
        data_vars['flow_lb'] = _polars_long_to_dataarray(lb_df, ['flow', 'time'], 'value', 'flow_lb')
        data_vars['flow_ub'] = _polars_long_to_dataarray(ub_df, ['flow', 'time'], 'value', 'flow_ub')

    # Flow fixed (flow, time)
    if len(data.flows.fixed) > 0:
        data_vars['flow_fixed'] = _polars_long_to_dataarray(data.flows.fixed, ['flow', 'time'], 'value', 'flow_fixed')

    # Flow sizes (flow,)
    if len(data.flows.sizes) > 0:
        data_vars['flow_size'] = _polars_long_to_dataarray(data.flows.sizes, ['flow'], 'size', 'flow_size')

    # Effect coefficients (flow, effect, time)
    if len(data.flows.effect_coefficients) > 0:
        data_vars['effect_coeff'] = _polars_long_to_dataarray(
            data.flows.effect_coefficients, ['flow', 'effect', 'time'], 'coeff', 'effect_coeff'
        )

    # Bus flow coefficients (bus, flow)
    if len(data.buses.flow_coefficients) > 0:
        data_vars['bus_flow_coeff'] = _polars_long_to_dataarray(
            data.buses.flow_coefficients, ['bus', 'flow'], 'coeff', 'bus_flow_coeff'
        )

    # Converter flow coefficients (converter, eq_idx, flow, time)
    if len(data.converters.flow_coefficients) > 0:
        data_vars['converter_flow_coeff'] = _polars_long_to_dataarray(
            data.converters.flow_coefficients,
            ['converter', 'eq_idx', 'flow', 'time'],
            'coeff',
            'converter_flow_coeff',
        )

    # Effect bounds
    if len(data.effects.bounds) > 0:
        bounds_e = data.effects.bounds
        min_df = bounds_e.select('effect', pl.col('min_total').alias('value'))
        max_df = bounds_e.select('effect', pl.col('max_total').alias('value'))
        data_vars['effect_min_total'] = _polars_long_to_dataarray(min_df, ['effect'], 'value', 'effect_min_total')
        data_vars['effect_max_total'] = _polars_long_to_dataarray(max_df, ['effect'], 'value', 'effect_max_total')

    # Effect time bounds
    if len(data.effects.time_bounds_lb) > 0:
        data_vars['effect_time_lb'] = _polars_long_to_dataarray(
            data.effects.time_bounds_lb, ['effect', 'time'], 'value', 'effect_time_lb'
        )
    if len(data.effects.time_bounds_ub) > 0:
        data_vars['effect_time_ub'] = _polars_long_to_dataarray(
            data.effects.time_bounds_ub, ['effect', 'time'], 'value', 'effect_time_ub'
        )

    # Storage params
    if len(data.storages.params) > 0:
        params = data.storages.params
        data_vars['storage_capacity'] = _polars_long_to_dataarray(
            params.select('storage', pl.col('capacity').alias('value')), ['storage'], 'value', 'storage_capacity'
        )
        data_vars['storage_initial_charge'] = _polars_long_to_dataarray(
            params.select('storage', pl.col('initial_charge').alias('value')),
            ['storage'],
            'value',
            'storage_initial_charge',
        )
        # cyclic is boolean -> store as int
        data_vars['storage_cyclic'] = _polars_long_to_dataarray(
            params.select('storage', pl.col('cyclic').cast(pl.Int8).alias('value')),
            ['storage'],
            'value',
            'storage_cyclic',
        )

    # Storage time params (storage, time)
    if len(data.storages.time_params) > 0:
        tp = data.storages.time_params
        for col in ['eta_c', 'eta_d', 'loss']:
            data_vars[f'storage_{col}'] = _polars_long_to_dataarray(
                tp.select('storage', 'time', pl.col(col).alias('value')),
                ['storage', 'time'],
                'value',
                f'storage_{col}',
            )

    # Storage charge state bounds (storage, time)
    if len(data.storages.cs_bounds) > 0:
        csb = data.storages.cs_bounds
        data_vars['storage_cs_lb'] = _polars_long_to_dataarray(
            csb.select('storage', 'time', pl.col('cs_lb').alias('value')),
            ['storage', 'time'],
            'value',
            'storage_cs_lb',
        )
        data_vars['storage_cs_ub'] = _polars_long_to_dataarray(
            csb.select('storage', 'time', pl.col('cs_ub').alias('value')),
            ['storage', 'time'],
            'value',
            'storage_cs_ub',
        )

    # Storage flow map (storage,)
    if len(data.storages.flow_map) > 0:
        fm = data.storages.flow_map
        data_vars['storage_charge_flow'] = _polars_long_to_dataarray(
            fm.select('storage', pl.col('charge_flow').alias('value')),
            ['storage'],
            'value',
            'storage_charge_flow',
        )
        data_vars['storage_discharge_flow'] = _polars_long_to_dataarray(
            fm.select('storage', pl.col('discharge_flow').alias('value')),
            ['storage'],
            'value',
            'storage_discharge_flow',
        )

    # Objective effect
    ds = _xr.Dataset(data_vars)
    ds.attrs['objective_effect'] = data.effects.objective_effect

    # Write to the 'model' group
    ds.to_netcdf(path, mode='a', group='model')


def _read_model_data(path: str | Path, time_dtype: pl.DataType) -> ModelData:
    """Read ModelData from the 'model' group of a NetCDF file."""
    _xr = _require_xarray()
    from fluxopt.tables import (
        BusesTable,
        ConvertersTable,
        EffectsTable,
        FlowsTable,
        ModelData,
        StoragesTable,
    )

    ds = _xr.open_dataset(path, group='model')
    objective_effect: str = ds.attrs['objective_effect']

    # Helper to read a DataArray back to Polars
    def _read_da(name: str, dims: list[str], value_col: str) -> pl.DataFrame:
        if name not in ds:
            return pl.DataFrame()
        return _dataarray_to_polars_long(ds[name], value_col, time_dtype)

    # Timesteps from dt
    dt_df = _read_da('dt', ['time'], 'dt')
    timesteps_df = dt_df.select('time')
    weights_df = _read_da('weight', ['time'], 'weight')

    # Charge state times
    if 'charge_state_times' in ds:
        cs_vals = ds['charge_state_times'].values.tolist()
        cs_times = pl.DataFrame({'time': pl.Series('time', cs_vals, dtype=time_dtype)})
    else:
        cs_times = pl.DataFrame({'time': pl.Series('time', [], dtype=time_dtype)})

    # Flows
    flow_lb_df = _read_da('flow_lb', ['flow', 'time'], 'value')
    flow_ub_df = _read_da('flow_ub', ['flow', 'time'], 'value')

    if len(flow_lb_df) > 0 and len(flow_ub_df) > 0:
        relative_bounds = flow_lb_df.rename({'value': 'rel_lb'}).join(
            flow_ub_df.rename({'value': 'rel_ub'}), on=['flow', 'time']
        )
    else:
        relative_bounds = pl.DataFrame(
            schema={'flow': pl.String, 'time': time_dtype, 'rel_lb': pl.Float64, 'rel_ub': pl.Float64}
        )

    if len(relative_bounds) > 0:
        flow_index = pl.DataFrame({'flow': relative_bounds['flow'].unique().sort()})
    else:
        flow_index = pl.DataFrame({'flow': pl.Series([], dtype=pl.String)})

    sizes_df = _read_da('flow_size', ['flow'], 'size')
    if len(sizes_df) == 0:
        sizes_df = pl.DataFrame(schema={'flow': pl.String, 'size': pl.Float64})

    fixed_df = _read_da('flow_fixed', ['flow', 'time'], 'value')
    if len(fixed_df) == 0:
        fixed_df = pl.DataFrame(schema={'flow': pl.String, 'time': time_dtype, 'value': pl.Float64})

    effect_coefficients = _read_da('effect_coeff', ['flow', 'effect', 'time'], 'coeff')
    if len(effect_coefficients) == 0:
        effect_coefficients = pl.DataFrame(
            schema={'flow': pl.String, 'effect': pl.String, 'time': time_dtype, 'coeff': pl.Float64}
        )

    flows_table = FlowsTable(
        index=flow_index,
        sizes=sizes_df,
        relative_bounds=relative_bounds,
        fixed=fixed_df,
        effect_coefficients=effect_coefficients,
    )

    # Buses
    bus_coeffs = _read_da('bus_flow_coeff', ['bus', 'flow'], 'coeff')
    if len(bus_coeffs) == 0:
        bus_coeffs = pl.DataFrame(schema={'bus': pl.String, 'flow': pl.String, 'coeff': pl.Float64})
    if len(bus_coeffs) > 0:
        bus_index = pl.DataFrame({'bus': bus_coeffs['bus'].unique().sort()})
    else:
        bus_index = pl.DataFrame({'bus': pl.Series([], dtype=pl.String)})
    buses_table = BusesTable(index=bus_index, flow_coefficients=bus_coeffs)

    # Converters
    conv_coeffs = _read_da('converter_flow_coeff', ['converter', 'eq_idx', 'flow', 'time'], 'coeff')
    if len(conv_coeffs) == 0:
        conv_coeffs = pl.DataFrame(
            schema={
                'converter': pl.String,
                'eq_idx': pl.Int64,
                'flow': pl.String,
                'time': time_dtype,
                'coeff': pl.Float64,
            }
        )
    if len(conv_coeffs) > 0:
        conv_index = pl.DataFrame({'converter': conv_coeffs['converter'].unique().sort()})
    else:
        conv_index = pl.DataFrame({'converter': pl.Series([], dtype=pl.String)})
    converters_table = ConvertersTable(index=conv_index, flow_coefficients=conv_coeffs)

    # Effects
    effect_min_df = _read_da('effect_min_total', ['effect'], 'value')
    effect_max_df = _read_da('effect_max_total', ['effect'], 'value')

    if len(effect_min_df) > 0 and len(effect_max_df) > 0:
        effects_bounds = effect_min_df.rename({'value': 'min_total'}).join(
            effect_max_df.rename({'value': 'max_total'}), on='effect'
        )
    elif len(effect_min_df) > 0:
        effects_bounds = effect_min_df.rename({'value': 'min_total'}).with_columns(
            pl.lit(None).cast(pl.Float64).alias('max_total')
        )
    elif len(effect_max_df) > 0:
        effects_bounds = (
            effect_max_df.rename({'value': 'max_total'})
            .with_columns(pl.lit(None).cast(pl.Float64).alias('min_total'))
            .select('effect', 'min_total', 'max_total')
        )
    else:
        effects_bounds = pl.DataFrame(schema={'effect': pl.String, 'min_total': pl.Float64, 'max_total': pl.Float64})

    if len(effects_bounds) > 0:
        effects_index = pl.DataFrame({'effect': effects_bounds['effect']})
    else:
        effects_index = pl.DataFrame({'effect': pl.Series([], dtype=pl.String)})

    effect_time_lb = _read_da('effect_time_lb', ['effect', 'time'], 'value')
    if len(effect_time_lb) == 0:
        effect_time_lb = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64})

    effect_time_ub = _read_da('effect_time_ub', ['effect', 'time'], 'value')
    if len(effect_time_ub) == 0:
        effect_time_ub = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'value': pl.Float64})

    effects_table = EffectsTable(
        index=effects_index,
        objective_effect=objective_effect,
        bounds=effects_bounds,
        time_bounds_lb=effect_time_lb,
        time_bounds_ub=effect_time_ub,
    )

    # Storages
    stor_capacity = _read_da('storage_capacity', ['storage'], 'value')
    if len(stor_capacity) > 0:
        stor_initial = _read_da('storage_initial_charge', ['storage'], 'value')
        stor_cyclic = _read_da('storage_cyclic', ['storage'], 'value')

        storage_params = (
            stor_capacity.rename({'value': 'capacity'})
            .join(stor_initial.rename({'value': 'initial_charge'}), on='storage')
            .join(stor_cyclic.rename({'value': 'cyclic'}), on='storage')
            .with_columns(pl.col('cyclic').cast(pl.Boolean))
        )
        storage_index = pl.DataFrame({'storage': storage_params['storage']})

        # Time params
        eta_c = _read_da('storage_eta_c', ['storage', 'time'], 'value').rename({'value': 'eta_c'})
        eta_d = _read_da('storage_eta_d', ['storage', 'time'], 'value').rename({'value': 'eta_d'})
        loss = _read_da('storage_loss', ['storage', 'time'], 'value').rename({'value': 'loss'})

        storage_time_params = eta_c.join(eta_d, on=['storage', 'time']).join(loss, on=['storage', 'time'])

        # Flow map
        charge_flow = _read_da('storage_charge_flow', ['storage'], 'value').rename({'value': 'charge_flow'})
        discharge_flow = _read_da('storage_discharge_flow', ['storage'], 'value').rename({'value': 'discharge_flow'})
        storage_flow_map = charge_flow.join(discharge_flow, on='storage')

        # CS bounds
        cs_lb = _read_da('storage_cs_lb', ['storage', 'time'], 'value')
        cs_ub = _read_da('storage_cs_ub', ['storage', 'time'], 'value')
        if len(cs_lb) > 0 and len(cs_ub) > 0:
            cs_bounds = cs_lb.rename({'value': 'cs_lb'}).join(cs_ub.rename({'value': 'cs_ub'}), on=['storage', 'time'])
        else:
            cs_bounds = pl.DataFrame(
                schema={'storage': pl.String, 'time': time_dtype, 'cs_lb': pl.Float64, 'cs_ub': pl.Float64}
            )
    else:
        storage_index = pl.DataFrame({'storage': pl.Series([], dtype=pl.String)})
        storage_params = pl.DataFrame(
            schema={
                'storage': pl.String,
                'capacity': pl.Float64,
                'initial_charge': pl.Float64,
                'cyclic': pl.Boolean,
            }
        )
        storage_time_params = pl.DataFrame(
            schema={
                'storage': pl.String,
                'time': time_dtype,
                'eta_c': pl.Float64,
                'eta_d': pl.Float64,
                'loss': pl.Float64,
            }
        )
        storage_flow_map = pl.DataFrame(
            schema={'storage': pl.String, 'charge_flow': pl.String, 'discharge_flow': pl.String}
        )
        cs_bounds = pl.DataFrame(
            schema={'storage': pl.String, 'time': time_dtype, 'cs_lb': pl.Float64, 'cs_ub': pl.Float64}
        )

    storages_table = StoragesTable(
        index=storage_index,
        params=storage_params,
        time_params=storage_time_params,
        flow_map=storage_flow_map,
        cs_bounds=cs_bounds,
    )

    return ModelData(
        flows=flows_table,
        buses=buses_table,
        converters=converters_table,
        effects=effects_table,
        storages=storages_table,
        timesteps=timesteps_df,
        dt=dt_df,
        weights=weights_df,
        charge_state_times=cs_times,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solved_model_to_netcdf(result: SolvedModel, path: str | Path) -> None:
    """Write a SolvedModel to a NetCDF file (solution in root group, model data in 'model' group)."""
    path = Path(path)

    # Write solution to root group
    ds = solved_model_to_xarray(result)
    ds.to_netcdf(path, mode='w')

    # Write model data to 'model' group
    if result.data is not None:
        _write_model_data(result.data, path)


def solved_model_from_netcdf(path: str | Path) -> SolvedModel:
    """Read a SolvedModel from a NetCDF file."""
    _xr = _require_xarray()
    from fluxopt.results import SolvedModel

    path = Path(path)

    # Read solution from root group
    ds = _xr.open_dataset(path)

    objective_value: float = float(ds.attrs['objective_value'])
    time_dtype_label: str = ds.attrs.get('time_dtype', 'datetime')
    time_dtype: pl.DataType = pl.Int64() if time_dtype_label == 'int64' else pl.Datetime()

    # Flow rates
    if 'flow_rates' in ds:
        flow_rates = _dataarray_to_polars_long(ds['flow_rates'], 'solution', time_dtype)
    else:
        flow_rates = pl.DataFrame(schema={'flow': pl.String, 'time': time_dtype, 'solution': pl.Float64})

    # Charge states
    if 'charge_states' in ds:
        cs_df = _dataarray_to_polars_long(ds['charge_states'], 'solution')
        # Rename cs_time -> time
        if 'cs_time' in cs_df.columns:
            cs_df = cs_df.rename({'cs_time': 'time'})
        if 'time' in cs_df.columns:
            cs_df = cs_df.with_columns(pl.col('time').cast(time_dtype))
        charge_states = cs_df
    else:
        charge_states = pl.DataFrame(schema={'storage': pl.String, 'time': time_dtype, 'solution': pl.Float64})

    # Effects
    if 'effects' in ds:
        effects = _dataarray_to_polars_long(ds['effects'], 'solution')
    else:
        effects = pl.DataFrame(schema={'effect': pl.String, 'solution': pl.Float64})

    # Effects per timestep
    if 'effects_per_timestep' in ds:
        effects_per_timestep = _dataarray_to_polars_long(ds['effects_per_timestep'], 'solution', time_dtype)
    else:
        effects_per_timestep = pl.DataFrame(schema={'effect': pl.String, 'time': time_dtype, 'solution': pl.Float64})

    # Contributions
    if 'contributions' in ds:
        contributions = _dataarray_to_polars_long(ds['contributions'], 'solution', time_dtype)
    else:
        contributions = pl.DataFrame(
            schema={
                'source': pl.String,
                'contributor': pl.String,
                'effect': pl.String,
                'time': time_dtype,
                'solution': pl.Float64,
            }
        )

    ds.close()

    # Read model data if 'model' group exists
    data: ModelData | None = None
    with contextlib.suppress(Exception):
        data = _read_model_data(path, time_dtype)

    return SolvedModel(
        objective_value=objective_value,
        flow_rates=flow_rates,
        charge_states=charge_states,
        effects=effects,
        effects_per_timestep=effects_per_timestep,
        contributions=contributions,
        data=data,
    )
