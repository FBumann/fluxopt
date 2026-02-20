"""Tests for the YAML+CSV loader."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fluxopt.yaml_loader import (
    YamlLoadError,
    _build_flow,
    _build_sizing,
    _build_status,
    _evaluate_expression,
    load_yaml,
    solve_yaml,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Expression parser
# ---------------------------------------------------------------------------


class TestExpressionParser:
    def test_literal(self):
        assert _evaluate_expression('42', {}) == 42.0

    def test_float_literal(self):
        assert _evaluate_expression('3.14', {}) == pytest.approx(3.14)

    def test_scientific_notation(self):
        assert _evaluate_expression('1.5e-2', {}) == pytest.approx(0.015)

    def test_addition(self):
        assert _evaluate_expression('2 + 3', {}) == 5.0

    def test_subtraction(self):
        assert _evaluate_expression('10 - 4', {}) == 6.0

    def test_multiplication(self):
        assert _evaluate_expression('3 * 4', {}) == 12.0

    def test_division(self):
        assert _evaluate_expression('10 / 4', {}) == 2.5

    def test_precedence(self):
        assert _evaluate_expression('2 + 3 * 4', {}) == 14.0

    def test_parentheses(self):
        assert _evaluate_expression('(2 + 3) * 4', {}) == 20.0

    def test_unary_minus(self):
        assert _evaluate_expression('-5', {}) == -5.0

    def test_unary_minus_in_expr(self):
        assert _evaluate_expression('3 + -2', {}) == 1.0

    def test_column_name(self):
        ns = {'price': np.array([1.0, 2.0, 3.0])}
        result = _evaluate_expression('price', ns)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_column_arithmetic(self):
        ns = {'price': np.array([10.0, 20.0])}
        result = _evaluate_expression('price * 1.19', ns)
        np.testing.assert_allclose(result, [11.9, 23.8])

    def test_two_columns(self):
        ns = {'a': np.array([1.0, 2.0]), 'b': np.array([3.0, 4.0])}
        result = _evaluate_expression('a + b', ns)
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_unknown_name_raises(self):
        with pytest.raises(YamlLoadError, match='Unknown name'):
            _evaluate_expression('unknown_col', {})

    def test_nested_parentheses(self):
        assert _evaluate_expression('((2 + 3))', {}) == 5.0

    def test_complex_expression(self):
        ns = {'x': 2.0}
        assert _evaluate_expression('(x + 3) * (x - 1) / 2', ns) == pytest.approx(2.5)

    def test_quoted_name(self):
        ns = {'gas-price': np.array([10.0, 20.0])}
        result = _evaluate_expression("'gas-price' * 1.19", ns)
        np.testing.assert_allclose(result, [11.9, 23.8])

    def test_quoted_name_with_spaces(self):
        ns = {'price [EUR/MWh]': np.array([5.0])}
        result = _evaluate_expression("'price [EUR/MWh]' / 100", ns)
        np.testing.assert_allclose(result, [0.05])

    def test_unterminated_quote_raises(self):
        with pytest.raises(YamlLoadError, match='Unterminated quoted name'):
            _evaluate_expression("'oops", {})


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------


class TestBuildSizing:
    def test_basic(self):
        s = _build_sizing({'min': 10, 'max': 100})
        assert s.min_size == 10.0
        assert s.max_size == 100.0
        assert s.mandatory is False

    def test_mandatory(self):
        s = _build_sizing({'min': 5, 'max': 50, 'mandatory': True})
        assert s.mandatory is True

    def test_with_effects(self):
        s = _build_sizing({'min': 0, 'max': 100, 'effects_per_size': {'cost': 1000}})
        assert s.effects_per_size == {'cost': 1000}


class TestBuildStatus:
    def test_true_gives_defaults(self):
        s = _build_status(True, {})
        assert isinstance(s, type(s))
        assert s.min_uptime is None

    def test_dict(self):
        s = _build_status({'min_uptime': 2, 'min_downtime': 1}, {})
        assert s.min_uptime == 2
        assert s.min_downtime == 1

    def test_invalid_type_raises(self):
        with pytest.raises(YamlLoadError, match='status must be true or a mapping'):
            _build_status('invalid', {})  # type: ignore[arg-type]


class TestBuildFlow:
    def test_minimal(self):
        f = _build_flow({'bus': 'elec'}, {}, 'test')
        assert f.bus == 'elec'
        assert f.id == 'elec'  # defaults to bus

    def test_with_size(self):
        f = _build_flow({'bus': 'gas', 'size': 200}, {}, 'test')
        assert f.size == 200.0

    def test_with_sizing(self):
        f = _build_flow({'bus': 'gas', 'size': {'min': 10, 'max': 100}}, {}, 'test')
        assert isinstance(f.size, type(f.size))
        assert f.size.min_size == 10.0  # type: ignore[union-attr]

    def test_with_profile(self):
        f = _build_flow({'bus': 'heat', 'size': 100, 'fixed_relative_profile': [0.4, 0.7]}, {}, 'test')
        assert f.fixed_relative_profile == [0.4, 0.7]

    def test_with_expression(self):
        ns = {'demand': np.array([40.0, 70.0])}
        f = _build_flow(
            {'bus': 'heat', 'size': 100, 'fixed_relative_profile': 'demand / 100'},
            ns,
            'test',
        )
        assert f.fixed_relative_profile == pytest.approx([0.4, 0.7])

    def test_missing_bus_raises(self):
        with pytest.raises(YamlLoadError, match="missing required key 'bus'"):
            _build_flow({'size': 100}, {}, 'test flow')

    def test_with_effects(self):
        f = _build_flow(
            {'bus': 'gas', 'effects_per_flow_hour': {'cost': 0.04}},
            {},
            'test',
        )
        assert f.effects_per_flow_hour == {'cost': 0.04}


# ---------------------------------------------------------------------------
# Integration: YAML roundtrip
# ---------------------------------------------------------------------------


class TestLoadYaml:
    def test_simple_system(self, tmp_path: Path):
        """Load a minimal YAML system (no CSV)."""
        yaml_content = dedent("""\
            timesteps:
              - "2024-01-01 00:00"
              - "2024-01-01 01:00"
              - "2024-01-01 02:00"
              - "2024-01-01 03:00"

            buses:
              - id: gas
              - id: heat

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: 0.04
              - id: demand
                exports:
                  - bus: heat
                    size: 100
                    fixed_relative_profile: [0.4, 0.7, 0.5, 0.6]

            converters:
              - id: boiler
                type: boiler
                thermal_efficiency: 0.9
                fuel:
                  bus: gas
                  size: 300
                thermal:
                  bus: heat
                  size: 200
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        result = load_yaml(tmp_path / 'model.yaml')

        assert len(result['timesteps']) == 4
        assert len(result['buses']) == 2
        assert len(result['effects']) == 1
        assert len(result['ports']) == 2
        assert len(result['converters']) == 1

    def test_with_csv(self, tmp_path: Path):
        """Load YAML with CSV time series."""
        csv_content = dedent("""\
            time,gas_price,demand_profile
            2024-01-01 00:00,0.03,0.4
            2024-01-01 01:00,0.05,0.7
            2024-01-01 02:00,0.04,0.5
            2024-01-01 03:00,0.06,0.6
        """)
        yaml_content = dedent("""\
            timeseries: data.csv

            buses:
              - id: gas
              - id: heat

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: "gas_price"
              - id: demand
                exports:
                  - bus: heat
                    size: 100
                    fixed_relative_profile: "demand_profile"

            converters:
              - id: boiler
                type: boiler
                thermal_efficiency: 0.9
                fuel:
                  bus: gas
                  size: 300
                thermal:
                  bus: heat
                  size: 200
        """)
        (tmp_path / 'data.csv').write_text(csv_content)
        (tmp_path / 'model.yaml').write_text(yaml_content)

        result = load_yaml(tmp_path / 'model.yaml')
        assert len(result['timesteps']) == 4

        # Verify expression resolution: gas_price column was loaded
        grid_port = result['ports'][0]
        gas_flow = grid_port.imports[0]
        assert gas_flow.effects_per_flow_hour['cost'] == pytest.approx([0.03, 0.05, 0.04, 0.06])

    def test_expression_in_csv(self, tmp_path: Path):
        """Expressions referencing CSV columns work."""
        csv_content = dedent("""\
            time,base_price
            2024-01-01 00:00,0.03
            2024-01-01 01:00,0.05
        """)
        yaml_content = dedent("""\
            timeseries: data.csv

            buses:
              - id: gas

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: "base_price * 1.19"
        """)
        (tmp_path / 'data.csv').write_text(csv_content)
        (tmp_path / 'model.yaml').write_text(yaml_content)

        result = load_yaml(tmp_path / 'model.yaml')
        gas_flow = result['ports'][0].imports[0]
        assert gas_flow.effects_per_flow_hour['cost'] == pytest.approx([0.03 * 1.19, 0.05 * 1.19])

    def test_explicit_converter(self, tmp_path: Path):
        """Explicit converter form with conversion_factors."""
        yaml_content = dedent("""\
            timesteps:
              - "2024-01-01 00:00"
              - "2024-01-01 01:00"

            buses:
              - id: gas
              - id: elec
              - id: heat

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
              - id: demand_e
                exports:
                  - bus: elec
                    size: 100
                    fixed_relative_profile: [0.5, 0.5]
              - id: demand_h
                exports:
                  - bus: heat
                    size: 100
                    fixed_relative_profile: [0.5, 0.5]

            converters:
              - id: chp
                inputs:
                  - bus: gas
                    size: 300
                outputs:
                  - bus: elec
                    size: 100
                  - bus: heat
                    size: 200
                conversion_factors:
                  - gas: 0.4
                    elec: -1
                  - gas: 0.5
                    heat: -1
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        result = load_yaml(tmp_path / 'model.yaml')

        conv = result['converters'][0]
        assert conv.id == 'chp'
        assert len(conv.inputs) == 1
        assert len(conv.outputs) == 2
        assert len(conv.conversion_factors) == 2

    def test_storage(self, tmp_path: Path):
        """Storage elements load correctly."""
        yaml_content = dedent("""\
            timesteps:
              - "2024-01-01 00:00"
              - "2024-01-01 01:00"
              - "2024-01-01 02:00"

            buses:
              - id: elec

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: elec
                    size: 200
                    effects_per_flow_hour:
                      cost: [0.02, 0.08, 0.02]
              - id: demand
                exports:
                  - bus: elec
                    size: 100
                    fixed_relative_profile: [0.5, 0.5, 0.5]

            storages:
              - id: battery
                charging:
                  bus: elec
                  size: 50
                discharging:
                  bus: elec
                  size: 50
                capacity: 100
                eta_charge: 0.95
                eta_discharge: 0.95
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        result = load_yaml(tmp_path / 'model.yaml')

        assert len(result['storages']) == 1
        s = result['storages'][0]
        assert s.id == 'battery'
        assert s.capacity == 100.0
        assert s.eta_charge == 0.95

    def test_multi_csv(self, tmp_path: Path):
        """Multiple CSVs merge into one namespace."""
        prices_csv = dedent("""\
            time,gas_price,elec_price
            2024-01-01 00:00,0.03,0.10
            2024-01-01 01:00,0.05,0.12
        """)
        weather_csv = dedent("""\
            time,solar_cf
            2024-01-01 00:00,0.0
            2024-01-01 01:00,0.8
        """)
        yaml_content = dedent("""\
            timeseries:
              prices: prices.csv
              weather: weather.csv

            buses:
              - id: gas
              - id: elec

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: "gas_price"
                  - bus: elec
                    size: 200
                    effects_per_flow_hour:
                      cost: "elec_price"
              - id: pv
                imports:
                  - bus: elec
                    size: 100
                    fixed_relative_profile: "solar_cf"
        """)
        (tmp_path / 'prices.csv').write_text(prices_csv)
        (tmp_path / 'weather.csv').write_text(weather_csv)
        (tmp_path / 'model.yaml').write_text(yaml_content)

        result = load_yaml(tmp_path / 'model.yaml')
        assert len(result['timesteps']) == 2

        # Columns from both CSVs are accessible
        grid = result['ports'][0]
        assert grid.imports[0].effects_per_flow_hour['cost'] == pytest.approx([0.03, 0.05])
        assert grid.imports[1].effects_per_flow_hour['cost'] == pytest.approx([0.10, 0.12])

        pv = result['ports'][1]
        assert pv.imports[0].fixed_relative_profile == pytest.approx([0.0, 0.8])

    def test_multi_csv_length_mismatch_raises(self, tmp_path: Path):
        """CSVs with different row counts raise an error."""
        (tmp_path / 'a.csv').write_text('time,x\n2024-01-01,1\n2024-01-02,2\n')
        (tmp_path / 'b.csv').write_text('time,y\n2024-01-01,1\n')
        yaml_content = dedent("""\
            timeseries:
              a: a.csv
              b: b.csv
            buses: []
            effects: []
            ports: []
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        with pytest.raises(YamlLoadError, match='rows, expected'):
            load_yaml(tmp_path / 'model.yaml')

    def test_multi_csv_duplicate_column_raises(self, tmp_path: Path):
        """Duplicate column names across CSVs raise an error."""
        (tmp_path / 'a.csv').write_text('time,price\n2024-01-01,1\n')
        (tmp_path / 'b.csv').write_text('time,price\n2024-01-01,2\n')
        yaml_content = dedent("""\
            timeseries:
              a: a.csv
              b: b.csv
            buses: []
            effects: []
            ports: []
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        with pytest.raises(YamlLoadError, match="Duplicate column 'price'"):
            load_yaml(tmp_path / 'model.yaml')

    def test_missing_csv_raises(self, tmp_path: Path):
        yaml_content = dedent("""\
            timeseries: nonexistent.csv
            buses: []
            effects: []
            ports: []
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        with pytest.raises(YamlLoadError, match='CSV file not found'):
            load_yaml(tmp_path / 'model.yaml')

    def test_missing_timesteps_raises(self, tmp_path: Path):
        yaml_content = dedent("""\
            buses: []
            effects: []
            ports: []
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        with pytest.raises(YamlLoadError, match='No timesteps found'):
            load_yaml(tmp_path / 'model.yaml')

    def test_unknown_converter_type_raises(self, tmp_path: Path):
        yaml_content = dedent("""\
            timesteps:
              - "2024-01-01 00:00"
            buses:
              - id: gas
            effects: []
            ports: []
            converters:
              - id: thing
                type: unknown_machine
                some_param: 42
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        with pytest.raises(YamlLoadError, match="unknown converter type 'unknown_machine'"):
            load_yaml(tmp_path / 'model.yaml')


# ---------------------------------------------------------------------------
# End-to-end: solve via YAML
# ---------------------------------------------------------------------------


class TestSolveYaml:
    def test_full_system(self, tmp_path: Path):
        """Replicate test_full_system from test_end_to_end.py via YAML."""
        yaml_content = dedent("""\
            timesteps:
              - "2024-01-01 00:00"
              - "2024-01-01 01:00"
              - "2024-01-01 02:00"
              - "2024-01-01 03:00"

            buses:
              - id: gas
              - id: heat

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: 0.04
              - id: demand
                exports:
                  - bus: heat
                    size: 100
                    fixed_relative_profile: [0.4, 0.7, 0.5, 0.6]

            converters:
              - id: boiler
                type: boiler
                thermal_efficiency: 0.9
                fuel:
                  bus: gas
                  size: 300
                thermal:
                  bus: heat
                  size: 200
        """)
        (tmp_path / 'model.yaml').write_text(yaml_content)
        result = solve_yaml(tmp_path / 'model.yaml')

        eta = 0.9
        heat_demand = [40.0, 70.0, 50.0, 60.0]

        # Verify gas = heat / eta
        gas_rates = result.flow_rate('boiler(gas)').values
        for gas, hd in zip(gas_rates, heat_demand, strict=False):
            assert gas == pytest.approx(hd / eta, abs=1e-6)

        # Verify cost
        total_gas = sum(h / eta for h in heat_demand)
        expected_cost = total_gas * 0.04
        assert result.objective == pytest.approx(expected_cost, abs=1e-6)

    def test_storage_system(self, tmp_path: Path):
        """Boiler + storage: optimizer uses cheap hours."""
        csv_content = dedent("""\
            time,gas_price
            2024-01-01 00:00,0.02
            2024-01-01 01:00,0.08
            2024-01-01 02:00,0.02
            2024-01-01 03:00,0.08
        """)
        yaml_content = dedent("""\
            timeseries: data.csv

            buses:
              - id: gas
              - id: heat

            effects:
              - id: cost
                is_objective: true

            ports:
              - id: grid
                imports:
                  - bus: gas
                    size: 500
                    effects_per_flow_hour:
                      cost: "gas_price"
              - id: demand
                exports:
                  - bus: heat
                    size: 100
                    fixed_relative_profile: [0.5, 0.5, 0.5, 0.5]

            converters:
              - id: boiler
                type: boiler
                thermal_efficiency: 0.9
                fuel:
                  bus: gas
                  size: 300
                thermal:
                  bus: heat
                  size: 200

            storages:
              - id: heat_store
                charging:
                  bus: heat
                  size: 100
                discharging:
                  bus: heat
                  size: 100
                capacity: 200
        """)
        (tmp_path / 'data.csv').write_text(csv_content)
        (tmp_path / 'model.yaml').write_text(yaml_content)

        result = solve_yaml(tmp_path / 'model.yaml')

        # Optimizer buys more gas in cheap hours
        gas_rates = result.flow_rate('grid(gas)').values
        assert gas_rates[0] > gas_rates[1]
