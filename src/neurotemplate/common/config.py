"""YAML + --set + dataclass loader machinery, shared by modeling and control."""
from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, get_args, get_origin, get_type_hints, Union

import yaml

MODEL_CONFIG_DIR = Path("model_configs")
CONTROL_CONFIG_DIR = Path("control_configs")

T = TypeVar("T")


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping, got {type(data).__name__}")
    return data


def _unwrap_optional(tp: Any) -> Any:
    """If tp is Optional[T] / Union[T, None], return T; else return tp."""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_optional(tp: Any) -> bool:
    """True if tp is Optional[T] / Union[..., None]."""
    return get_origin(tp) is Union and type(None) in get_args(tp)


_NULL_STRINGS = {"none", "null"}


def _coerce_null(value: Any, field_type: Any) -> Any:
    """For Optional fields, treat string 'none'/'null'/'None' as Python None."""
    if (
        _is_optional(field_type)
        and isinstance(value, str)
        and value.strip().lower() in _NULL_STRINGS
    ):
        return None
    return value


def _coerce_float(value: Any, field_type: Any) -> Any:
    """Normalize float-typed config values written in any float format.

    PyYAML's `safe_load` only recognizes floats with a decimal point and a
    signed exponent, so `1e5`, `1E5`, `1.0e5`, `1.0E5` all load as strings
    and `10000` loads as an int. For fields declared `float` (or
    `Optional[float]`) this coerces string/int inputs so every one of
    `10000`, `10000.0`, `1e5`, `1E5`, `1.0e5`, `1.0E5` resolves to the same
    float. Booleans are left alone (a `float` field never wants True/False).
    """
    if _unwrap_optional(field_type) is not float:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"Expected a float value, got {value!r}") from exc


def dict_to_dataclass(data: dict, cls: Type[T], path: str = "") -> T:
    if not is_dataclass(cls):
        raise ConfigError(f"{cls} is not a dataclass")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping")

    field_map = {f.name: f for f in fields(cls)}
    for key in data.keys():
        if key not in field_map:
            raise ConfigError(f"Unknown config key: {path}{key}")

    type_hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        key = f.name
        if key in data:
            field_type = type_hints.get(key, f.type)
            value = _coerce_null(data[key], field_type)
            kwargs[key] = _coerce_float(value, field_type)
        else:
            if f.default is not MISSING:
                kwargs[key] = f.default
            elif f.default_factory is not MISSING:  # type: ignore
                kwargs[key] = f.default_factory()  # type: ignore
            else:
                raise ConfigError(f"Missing required config key: {path}{key}")

    return cls(**kwargs)


def default_savedir(prefix: str = "temp") -> str:
    """Per-process-unique scratch dir so concurrent runs (e.g. separate
    terminals) never share/clobber the same savedir."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def paths_as_list(value: Union[str, list[str]]) -> list[str]:
    """Normalize a `Union[str, list[str]]` data-path field to `list[str]`.

    Single strings become one-element lists so downstream code only ever
    sees `list[df]`. Empty inputs are not silently accepted — the empty
    case is rejected by `_check_paths_field` at validate time.
    """
    if isinstance(value, str):
        return [value]
    return list(value)


def _check_paths_field(name: str, value: Any) -> None:
    """Validate a data-path field accepts only non-empty string(s)."""
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} is empty; provide a CSV path.")
        return
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} is an empty list; provide at least one path.")
        for i, p in enumerate(value):
            if not isinstance(p, str) or not p:
                raise ValueError(
                    f"{name}[{i}] must be a non-empty string, got {p!r}"
                )
        return
    raise ValueError(
        f"{name} must be a string or list of strings, got {type(value).__name__}"
    )


def check_positive(obj: Any, name: str) -> None:
    x = getattr(obj, name)
    if x <= 0:
        raise ValueError(f"{name} must be positive, got {x}")


def check_non_negative(obj: Any, name: str) -> None:
    x = getattr(obj, name)
    if x < 0:
        raise ValueError(f"{name} must be non-negative, got {x}")


def check_one_of(obj: Any, name: str, options: list) -> None:
    x = getattr(obj, name)
    if x not in options:
        raise ValueError(f"{name} must be one of {options}, got {x}")


def _parse_value(raw: str) -> Any:
    """Parse a `--set key=value` CLI value into a Python object.

    Precedence (first match wins): bool literal -> null literal -> quoted
    string -> JSON/Python list/dict -> float (if contains . / e / E) ->
    int -> fall-through to raw string. Each check is narrow enough to be
    unambiguous against the next, so order rather than the individual
    checks does the disambiguating.
    """
    text = raw.strip()
    lower = text.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none"):
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]

    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                import ast
                return ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise ConfigError(
                    f"Invalid list/dict override value: {text}"
                ) from exc

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Override must be in key=value form: {override}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Override key is empty in '{override}'")
        if key not in config_dict:
            raise ConfigError(f"Override target does not exist: {key}")
        config_dict[key] = _parse_value(raw_value)

    return config_dict


def list_available_configs(config_dir: Path) -> list[str]:
    if not config_dir.exists():
        return []
    return sorted([p.name for p in config_dir.glob("*.yaml")])


def resolve_config_path(
    config_arg: Optional[str],
    default_config_path: Optional[Path],
    config_dir: Path,
) -> Path:
    if config_arg:
        path = Path(config_arg).expanduser()
        if path.exists():
            return path
        name = path.name
        if not name.endswith(".yaml"):
            name += ".yaml"
        candidate = config_dir / name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Config not found: {config_arg} (looked for {candidate})"
        )

    if default_config_path is None:
        raise ValueError("--config is required")
    return default_config_path


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run with a YAML configuration file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config path or name in the config directory.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with key=value (repeatable).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config as YAML and exit.",
    )
    parser.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help="Write the resolved config to a YAML file and exit.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configs in the config directory and exit.",
    )
    return parser


def load_config_from_cli(
    argv: list[str],
    cls: Type,
    config_dir: Path,
    default_config_path: Optional[Path] = None,
) -> tuple[Any, argparse.Namespace]:
    """Load a config dataclass from a YAML file + CLI --set overrides."""

    parser = build_cli_parser()
    cli_args = parser.parse_args(argv)

    if cli_args.list_configs:
        for name in list_available_configs(config_dir):
            print(name)
        raise SystemExit(0)

    config_path = resolve_config_path(cli_args.config, default_config_path, config_dir)
    # Write the resolved path back, so callers reporting which config ran
    # name the file that was loaded rather than the bare name (or None) the
    # user typed.
    cli_args.config = str(config_path)
    cfg = dict_to_dataclass(load_yaml(config_path), cls)

    if cli_args.set:
        cfg = dict_to_dataclass(apply_overrides(cfg.asdict(), cli_args.set), cls)

    cfg.validate()

    if cli_args.print_config:
        print(yaml.safe_dump(cfg.asdict(), sort_keys=False))
        raise SystemExit(0)

    if cli_args.dump_config:
        dump_path = Path(cli_args.dump_config).expanduser()
        dump_path.write_text(yaml.safe_dump(cfg.asdict(), sort_keys=False),
                             encoding="utf-8")
        raise SystemExit(0)

    return cfg, cli_args
