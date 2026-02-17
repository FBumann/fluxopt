"""YAML + CSV loader for fluxopt models.

Loads a declarative YAML topology (with optional CSV time series) and
constructs all fluxopt elements ready for ``solve()``.

Supports inline arithmetic expressions referencing CSV columns::

    effects_per_flow_hour:
      cost: "gas_price * 1.19"

Public API:
    - ``load_yaml(path)`` — returns kwargs for ``solve()``
    - ``solve_yaml(path)`` — load + solve in one call
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from fluxopt.components import Converter, Port
from fluxopt.elements import Bus, Effect, Flow, Sizing, Status, Storage

if TYPE_CHECKING:
    from fluxopt.results import SolvedModel
    from fluxopt.types import TimeSeries


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class YamlLoadError(Exception):
    """Raised for any problem during YAML/CSV loading or element construction."""


# ---------------------------------------------------------------------------
# Expression parser — eager-evaluation recursive descent
# ---------------------------------------------------------------------------


class _TokenType(enum.Enum):
    NUMBER = 'NUMBER'
    NAME = 'NAME'
    PLUS = '+'
    MINUS = '-'
    STAR = '*'
    SLASH = '/'
    LPAREN = '('
    RPAREN = ')'
    EOF = 'EOF'


@dataclass
class _Token:
    type: _TokenType
    value: str


def _tokenize(text: str) -> list[_Token]:
    """Scan expression text into tokens."""
    tokens: list[_Token] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in {'+', '-', '*', '/', '(', ')'}:
            # Check for unambiguous operator tokens; unary minus handled in parser
            tok_map = {
                '+': _TokenType.PLUS,
                '-': _TokenType.MINUS,
                '*': _TokenType.STAR,
                '/': _TokenType.SLASH,
                '(': _TokenType.LPAREN,
                ')': _TokenType.RPAREN,
            }
            tokens.append(_Token(tok_map[ch], ch))
            i += 1
            continue
        if ch.isdigit() or ch == '.':
            start = i
            while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                i += 1
            # scientific notation: e.g. 1.5e-2
            if i < len(text) and text[i] in {'e', 'E'}:
                i += 1
                if i < len(text) and text[i] in {'+', '-'}:
                    i += 1
                while i < len(text) and text[i].isdigit():
                    i += 1
            tokens.append(_Token(_TokenType.NUMBER, text[start:i]))
            continue
        if ch.isalpha() or ch == '_':
            start = i
            while i < len(text) and (text[i].isalnum() or text[i] == '_'):
                i += 1
            tokens.append(_Token(_TokenType.NAME, text[start:i]))
            continue
        raise YamlLoadError(f'Unexpected character {ch!r} in expression: {text!r}')
    tokens.append(_Token(_TokenType.EOF, ''))
    return tokens


class _ExpressionParser:
    """Recursive-descent parser that eagerly evaluates to float or ndarray."""

    def __init__(self, tokens: list[_Token], namespace: dict[str, Any]) -> None:
        self._tokens = tokens
        self._ns = namespace
        self._pos = 0

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def parse(self) -> float | np.ndarray:
        result = self._expr()
        if self._peek().type != _TokenType.EOF:
            raise YamlLoadError(f'Unexpected token after expression: {self._peek().value!r}')
        return result

    def _expr(self) -> float | np.ndarray:
        """term ((+|-) term)*"""
        left = self._term()
        while self._peek().type in {_TokenType.PLUS, _TokenType.MINUS}:
            op = self._advance()
            right = self._term()
            left = left + right if op.type == _TokenType.PLUS else left - right
        return left

    def _term(self) -> float | np.ndarray:
        """atom ((*|/) atom)*"""
        left = self._atom()
        while self._peek().type in {_TokenType.STAR, _TokenType.SLASH}:
            op = self._advance()
            right = self._atom()
            left = left * right if op.type == _TokenType.STAR else left / right
        return left

    def _atom(self) -> float | np.ndarray:
        """NUMBER | NAME | '(' expr ')' | '-' atom"""
        tok = self._peek()
        if tok.type == _TokenType.NUMBER:
            self._advance()
            return float(tok.value)
        if tok.type == _TokenType.NAME:
            self._advance()
            if tok.value not in self._ns:
                raise YamlLoadError(f'Unknown name {tok.value!r} in expression')
            val = self._ns[tok.value]
            return float(val) if isinstance(val, (int, float)) else np.asarray(val, dtype=float)
        if tok.type == _TokenType.MINUS:
            self._advance()
            return -self._atom()
        if tok.type == _TokenType.LPAREN:
            self._advance()
            result = self._expr()
            if self._peek().type != _TokenType.RPAREN:
                raise YamlLoadError("Expected ')' in expression")
            self._advance()
            return result
        raise YamlLoadError(f'Unexpected token {tok.value!r} in expression')


def _evaluate_expression(text: str, namespace: dict[str, Any]) -> float | np.ndarray:
    """Evaluate an arithmetic expression with column references.

    Args:
        text: Expression string, e.g. ``"gas_price * 1.19"``.
        namespace: Mapping of names to scalars or arrays.
    """
    tokens = _tokenize(text)
    return _ExpressionParser(tokens, namespace).parse()


# ---------------------------------------------------------------------------
# TimeSeries resolution
# ---------------------------------------------------------------------------


def _resolve_timeseries(
    value: float | int | str | list[Any],
    namespace: dict[str, Any],
) -> float | list[float]:
    """Resolve a YAML value to a float or list[float].

    Args:
        value: Scalar, list, or expression string.
        namespace: CSV column namespace for expression evaluation.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        result = _evaluate_expression(value, namespace)
        if isinstance(result, np.ndarray):
            return list(result.astype(float))
        return float(result)
    raise YamlLoadError(f'Cannot resolve time series from {type(value).__name__}: {value!r}')


def _resolve_effects_dict(
    raw: dict[str, Any] | None,
    namespace: dict[str, Any],
) -> dict[str, TimeSeries]:
    """Resolve an effects dictionary where values may be expressions.

    Args:
        raw: Mapping of effect id to scalar/expression/list.
        namespace: CSV column namespace.
    """
    if not raw:
        return {}
    return {k: _resolve_timeseries(v, namespace) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------


def _build_sizing(raw: dict[str, Any]) -> Sizing:
    """Build a Sizing from a YAML dict.

    Args:
        raw: Dict with keys ``min``, ``max``, and optionally ``mandatory``,
             ``effects_per_size``, ``effects_fixed``.
    """
    return Sizing(
        min_size=float(raw['min']),
        max_size=float(raw['max']),
        mandatory=bool(raw.get('mandatory', False)),
        effects_per_size=raw.get('effects_per_size', {}),
        effects_fixed=raw.get('effects_fixed', {}),
    )


def _build_status(raw: dict[str, Any] | bool, namespace: dict[str, Any]) -> Status:
    """Build a Status from a YAML dict or ``true``.

    Args:
        raw: ``true`` for defaults, or dict with optional keys.
        namespace: CSV column namespace.
    """
    if raw is True:
        return Status()
    if not isinstance(raw, dict):
        raise YamlLoadError(f'status must be true or a mapping, got {type(raw).__name__}')
    return Status(
        min_uptime=raw.get('min_uptime'),
        max_uptime=raw.get('max_uptime'),
        min_downtime=raw.get('min_downtime'),
        max_downtime=raw.get('max_downtime'),
        effects_per_running_hour=_resolve_effects_dict(raw.get('effects_per_running_hour'), namespace),
        effects_per_startup=_resolve_effects_dict(raw.get('effects_per_startup'), namespace),
    )


def _build_flow(raw: dict[str, Any], namespace: dict[str, Any], context: str) -> Flow:
    """Build a Flow from a YAML dict.

    Args:
        raw: Flow specification dict.
        namespace: CSV column namespace.
        context: Human-readable location for error messages.
    """
    try:
        bus = raw['bus']
    except KeyError:
        raise YamlLoadError(f"In {context}: missing required key 'bus'") from None

    # size: scalar, Sizing dict, or absent
    size_raw = raw.get('size')
    size: float | Sizing | None = None
    if isinstance(size_raw, dict):
        size = _build_sizing(size_raw)
    elif size_raw is not None:
        size = float(size_raw)

    # status
    status_raw = raw.get('status')
    status = _build_status(status_raw, namespace) if status_raw is not None else None

    kwargs: dict[str, Any] = {'bus': bus, 'size': size, 'status': status}

    if 'id' in raw:
        kwargs['id'] = raw['id']
    if 'relative_minimum' in raw:
        kwargs['relative_minimum'] = _resolve_timeseries(raw['relative_minimum'], namespace)
    if 'relative_maximum' in raw:
        kwargs['relative_maximum'] = _resolve_timeseries(raw['relative_maximum'], namespace)
    if 'fixed_relative_profile' in raw:
        kwargs['fixed_relative_profile'] = _resolve_timeseries(raw['fixed_relative_profile'], namespace)
    if 'effects_per_flow_hour' in raw:
        kwargs['effects_per_flow_hour'] = _resolve_effects_dict(raw['effects_per_flow_hour'], namespace)
    if 'prior' in raw:
        kwargs['prior'] = [float(v) for v in raw['prior']]

    return Flow(**kwargs)


def _build_buses(raw: list[dict[str, Any]]) -> list[Bus]:
    """Build Bus list from YAML.

    Args:
        raw: List of bus dicts with ``id`` and optional ``carrier``.
    """
    buses: list[Bus] = []
    for i, item in enumerate(raw):
        try:
            buses.append(Bus(id=item['id'], carrier=item.get('carrier')))
        except KeyError:
            raise YamlLoadError(f"In bus {i}: missing required key 'id'") from None
    return buses


def _build_effects(raw: list[dict[str, Any]], namespace: dict[str, Any]) -> list[Effect]:
    """Build Effect list from YAML.

    Args:
        raw: List of effect dicts.
        namespace: CSV column namespace.
    """
    effects: list[Effect] = []
    for i, item in enumerate(raw):
        try:
            eid = item['id']
        except KeyError:
            raise YamlLoadError(f"In effect {i}: missing required key 'id'") from None

        kwargs: dict[str, Any] = {
            'id': eid,
            'unit': item.get('unit', ''),
            'is_objective': bool(item.get('is_objective', False)),
        }

        for key in ('maximum_total', 'minimum_total'):
            if key in item:
                kwargs[key] = float(item[key])
        for key in ('maximum_per_hour', 'minimum_per_hour'):
            if key in item:
                kwargs[key] = _resolve_timeseries(item[key], namespace)
        if 'contribution_from' in item:
            kwargs['contribution_from'] = {k: float(v) for k, v in item['contribution_from'].items()}
        if 'contribution_from_per_hour' in item:
            kwargs['contribution_from_per_hour'] = _resolve_effects_dict(item['contribution_from_per_hour'], namespace)

        effects.append(Effect(**kwargs))
    return effects


def _build_ports(raw: list[dict[str, Any]], namespace: dict[str, Any]) -> list[Port]:
    """Build Port list from YAML.

    Args:
        raw: List of port dicts.
        namespace: CSV column namespace.
    """
    ports: list[Port] = []
    for i, item in enumerate(raw):
        try:
            pid = item['id']
        except KeyError:
            raise YamlLoadError(f"In port {i}: missing required key 'id'") from None

        context = f"port '{pid}'"
        imports = [
            _build_flow(f, namespace, f'{context}, import flow {j}') for j, f in enumerate(item.get('imports', []))
        ]
        exports = [
            _build_flow(f, namespace, f'{context}, export flow {j}') for j, f in enumerate(item.get('exports', []))
        ]
        ports.append(Port(id=pid, imports=imports, exports=exports))
    return ports


def _build_storages(raw: list[dict[str, Any]], namespace: dict[str, Any]) -> list[Storage]:
    """Build Storage list from YAML.

    Args:
        raw: List of storage dicts.
        namespace: CSV column namespace.
    """
    storages: list[Storage] = []
    for i, item in enumerate(raw):
        try:
            sid = item['id']
        except KeyError:
            raise YamlLoadError(f"In storage {i}: missing required key 'id'") from None

        context = f"storage '{sid}'"
        charging = _build_flow(item.get('charging', {}), namespace, f'{context}, charging flow')
        discharging = _build_flow(item.get('discharging', {}), namespace, f'{context}, discharging flow')

        # capacity
        cap_raw = item.get('capacity')
        capacity: float | Sizing | None = None
        if isinstance(cap_raw, dict):
            capacity = _build_sizing(cap_raw)
        elif cap_raw is not None:
            capacity = float(cap_raw)

        kwargs: dict[str, Any] = {
            'id': sid,
            'charging': charging,
            'discharging': discharging,
            'capacity': capacity,
        }

        for key in ('eta_charge', 'eta_discharge', 'relative_loss_per_hour'):
            if key in item:
                kwargs[key] = _resolve_timeseries(item[key], namespace)
        if 'initial_charge_state' in item:
            val = item['initial_charge_state']
            kwargs['initial_charge_state'] = val if isinstance(val, str) and val == 'cyclic' else float(val)
        for key in ('relative_minimum_charge_state', 'relative_maximum_charge_state'):
            if key in item:
                kwargs[key] = _resolve_timeseries(item[key], namespace)

        storages.append(Storage(**kwargs))
    return storages


# ---------------------------------------------------------------------------
# Converter builders
# ---------------------------------------------------------------------------


def _build_converter_shorthand(raw: dict[str, Any], namespace: dict[str, Any]) -> Converter:
    """Build a converter from shorthand ``type:`` form.

    Args:
        raw: Converter dict with a ``type`` key.
        namespace: CSV column namespace.
    """
    cid = raw['id']
    ctype = raw['type']
    context = f"converter '{cid}'"

    if ctype == 'boiler':
        eta = _resolve_timeseries(raw['thermal_efficiency'], namespace)
        fuel_flow = _build_flow(raw['fuel'], namespace, f'{context}, fuel flow')
        thermal_flow = _build_flow(raw['thermal'], namespace, f'{context}, thermal flow')
        return Converter.boiler(cid, eta, fuel_flow, thermal_flow)

    if ctype == 'heat_pump':
        cop = _resolve_timeseries(raw['cop'], namespace)
        electrical_flow = _build_flow(raw['electrical'], namespace, f'{context}, electrical flow')
        thermal_flow = _build_flow(raw['thermal'], namespace, f'{context}, thermal flow')
        return Converter.heat_pump(cid, cop, electrical_flow, thermal_flow)

    if ctype == 'chp':
        eta_el = _resolve_timeseries(raw['eta_el'], namespace)
        eta_th = _resolve_timeseries(raw['eta_th'], namespace)
        fuel_flow = _build_flow(raw['fuel'], namespace, f'{context}, fuel flow')
        electrical_flow = _build_flow(raw['electrical'], namespace, f'{context}, electrical flow')
        thermal_flow = _build_flow(raw['thermal'], namespace, f'{context}, thermal flow')
        return Converter.chp(cid, eta_el, eta_th, fuel_flow, electrical_flow, thermal_flow)

    raise YamlLoadError(f"In {context}: unknown converter type '{ctype}'")


def _build_converter_explicit(raw: dict[str, Any], namespace: dict[str, Any]) -> Converter:
    """Build a converter from explicit ``inputs``/``outputs`` form.

    Args:
        raw: Converter dict with ``inputs`` and ``outputs`` keys.
        namespace: CSV column namespace.
    """
    cid = raw['id']
    context = f"converter '{cid}'"

    inputs = [_build_flow(f, namespace, f'{context}, input flow {j}') for j, f in enumerate(raw['inputs'])]
    outputs = [_build_flow(f, namespace, f'{context}, output flow {j}') for j, f in enumerate(raw['outputs'])]

    # Build a lookup from short flow id → Flow object (pre-qualification)
    flow_by_id: dict[str, Flow] = {}
    for f in [*inputs, *outputs]:
        flow_by_id[f.id] = f

    conversion_factors: list[dict[Flow, TimeSeries]] = []
    for eq_idx, eq_raw in enumerate(raw.get('conversion_factors', [])):
        eq: dict[Flow, TimeSeries] = {}
        for key, val in eq_raw.items():
            if key not in flow_by_id:
                raise YamlLoadError(
                    f"In {context}, conversion_factors[{eq_idx}]: unknown flow '{key}'. Available: {list(flow_by_id)}"
                )
            eq[flow_by_id[key]] = _resolve_timeseries(val, namespace)
        conversion_factors.append(eq)

    return Converter(id=cid, inputs=inputs, outputs=outputs, conversion_factors=conversion_factors)


def _build_converters(raw: list[dict[str, Any]], namespace: dict[str, Any]) -> list[Converter]:
    """Build Converter list from YAML (shorthand or explicit).

    Args:
        raw: List of converter dicts.
        namespace: CSV column namespace.
    """
    converters: list[Converter] = []
    for i, item in enumerate(raw):
        if 'id' not in item:
            raise YamlLoadError(f"In converter {i}: missing required key 'id'")
        if 'type' in item:
            converters.append(_build_converter_shorthand(item, namespace))
        elif 'inputs' in item or 'outputs' in item:
            converters.append(_build_converter_explicit(item, namespace))
        else:
            raise YamlLoadError(
                f"In converter '{item['id']}': must have either 'type' (shorthand) or 'inputs'/'outputs' (explicit)"
            )
    return converters


# ---------------------------------------------------------------------------
# CSV loading + timestep derivation
# ---------------------------------------------------------------------------


def _load_raw(yaml_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[Any] | None]:
    """Load YAML, CSV namespace, and timesteps.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Tuple of (raw YAML dict, namespace dict, timesteps or None).
    """
    with open(yaml_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise YamlLoadError(f'Expected YAML mapping at top level, got {type(raw).__name__}')

    namespace: dict[str, Any] = {}
    timesteps: list[Any] | None = None

    # Load CSV time series
    ts_spec = raw.get('timeseries')
    if ts_spec is not None:
        yaml_dir = yaml_path.parent
        if isinstance(ts_spec, str):
            csv_path = yaml_dir / ts_spec
        elif isinstance(ts_spec, dict):
            # Take the first (and for now only) file
            csv_path = yaml_dir / next(iter(ts_spec.values()))
        else:
            raise YamlLoadError(f"'timeseries' must be a string or mapping, got {type(ts_spec).__name__}")

        if not csv_path.exists():
            raise YamlLoadError(f'CSV file not found: {csv_path}')

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        namespace = {col: df[col].values for col in df.columns}

        # Derive timesteps from CSV index
        timesteps = list(df.index.to_pydatetime()) if isinstance(df.index, pd.DatetimeIndex) else df.index.tolist()

    # Explicit timesteps override
    if 'timesteps' in raw and timesteps is None:
        ts_raw = raw['timesteps']
        if isinstance(ts_raw, list):
            timesteps = [pd.Timestamp(t).to_pydatetime() for t in ts_raw]
        else:
            raise YamlLoadError(f"'timesteps' must be a list, got {type(ts_raw).__name__}")

    if timesteps is None:
        raise YamlLoadError("No timesteps found: provide 'timeseries' CSV or explicit 'timesteps' list")

    return raw, namespace, timesteps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML+CSV model definition and return ``solve()`` kwargs.

    Args:
        path: Path to the YAML file. CSV paths are resolved relative to it.

    Returns:
        Dict with keys: ``timesteps``, ``buses``, ``effects``, ``ports``,
        ``converters``, ``storages``, ``dt``.

    Raises:
        YamlLoadError: On missing keys, unknown types, or parse errors.
    """
    yaml_path = Path(path)
    raw, namespace, timesteps = _load_raw(yaml_path)

    try:
        buses = _build_buses(raw.get('buses', []))
    except YamlLoadError:
        raise
    except Exception as exc:
        raise YamlLoadError(f'Error building buses: {exc}') from exc

    try:
        effects = _build_effects(raw.get('effects', []), namespace)
    except YamlLoadError:
        raise
    except Exception as exc:
        raise YamlLoadError(f'Error building effects: {exc}') from exc

    try:
        ports = _build_ports(raw.get('ports', []), namespace)
    except YamlLoadError:
        raise
    except Exception as exc:
        raise YamlLoadError(f'Error building ports: {exc}') from exc

    try:
        converters = _build_converters(raw.get('converters', []), namespace)
    except YamlLoadError:
        raise
    except Exception as exc:
        raise YamlLoadError(f'Error building converters: {exc}') from exc

    try:
        storages = _build_storages(raw.get('storages', []), namespace)
    except YamlLoadError:
        raise
    except Exception as exc:
        raise YamlLoadError(f'Error building storages: {exc}') from exc

    result: dict[str, Any] = {
        'timesteps': timesteps,
        'buses': buses,
        'effects': effects,
        'ports': ports,
    }

    if converters:
        result['converters'] = converters
    if storages:
        result['storages'] = storages

    dt_raw = raw.get('dt')
    if dt_raw is not None:
        result['dt'] = dt_raw

    return result


def solve_yaml(path: str | Path, solver: str = 'highs', silent: bool = True) -> SolvedModel:
    """Load a YAML model and solve it.

    Args:
        path: Path to the YAML file.
        solver: Solver backend name.
        silent: Suppress solver output.
    """
    from fluxopt import solve

    kwargs = load_yaml(path)
    return solve(**kwargs, solver=solver, silent=silent)
