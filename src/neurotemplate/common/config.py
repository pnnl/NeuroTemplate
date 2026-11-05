"""YAML + --set + dataclass loader machinery, shared by modeling and control."""
from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import MISSING, fields, is_dataclass
from fractions import Fraction
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


def _to_float(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"Expected a float value, got {value!r}") from exc


def _coerce_float(value: Any, field_type: Any) -> Any:
    """Normalize float-typed config values written in any float format.

    PyYAML's `safe_load` only recognizes floats with a decimal point and a
    signed exponent, so `1e5`, `1E5`, `1.0e5`, `1.0E5` all load as strings
    and `10000` loads as an int. Wherever a field admits a `float` — itself,
    inside a Union, or as the elements of a `list[float]` or the values of a
    `dict[str, float]` — this coerces string/int inputs so every one of
    `10000`, `10000.0`, `1e5`, `1E5`, `1.0e5`, `1.0E5` resolves to the same
    float. Booleans are left alone (a `float` field never wants True/False).
    """
    members = get_args(field_type) if get_origin(field_type) is Union else (field_type,)
    for tp in members:
        if tp is float and isinstance(value, (int, str)):
            return _to_float(value)
        if get_origin(tp) is list and get_args(tp) == (float,) and isinstance(value, list):
            return [_to_float(v) for v in value]
        if get_origin(tp) is dict and get_args(tp)[1] is float and isinstance(value, dict):
            return {k: _to_float(v) for k, v in value.items()}
    return value


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


#: A window, written as a time: `5h`, `20m`, `6s`, `100ms`, `10us`. Zero needs no unit.
_TIME = re.compile(r"(\d+(?:\.\d+)?)(h|m|s|ms|us)")
_MICROSECONDS = {"h": 3_600_000_000, "m": 60_000_000, "s": 1_000_000, "ms": 1_000, "us": 1}
#: A value for any window field, in `Union` so the loader leaves it alone.
Time = Union[str, int]


def time_microseconds(value: Time, name: str) -> Fraction:
    """`value` in microseconds, exactly; a bare number other than zero is refused,
    because a count of steps means nothing until you know the timestep."""
    if value in (0, "0"):
        return Fraction(0)
    match = _TIME.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ConfigError(f"{name} is a time, like 5h, 20m, 6s, 100ms or 10us; got {value!r}.")
    return Fraction(match.group(1)) * _MICROSECONDS[match.group(2)]


def time_steps(value: Time, steps_per_hour: Optional[float], name: str) -> int:
    """`value` in steps of the data, or a refusal when it is not a whole number of them."""
    if steps_per_hour is None:
        raise ConfigError(f"{name} ({value}) can only be read as steps once the data has "
                          "set the timestep (`steps_per_hour`).")
    timestep = Fraction(3_600_000_000) / Fraction(steps_per_hour).limit_denominator(10**9)
    steps = time_microseconds(value, name) / timestep
    if steps.denominator != 1:
        raise ConfigError(f"{name} ({value}) is not a whole number of the data's "
                          f"{float(timestep) / 1e6:g} s steps.")
    return int(steps)


def steps_as_time(steps: int, steps_per_hour: float) -> Time:
    """`steps` of the data as a config writes them: whole hours or minutes, or seconds."""
    if steps == 0:
        return 0
    seconds = Fraction(3600) * steps / Fraction(steps_per_hour).limit_denominator(10**9)
    for unit, size in (("h", 3600), ("m", 60)):
        if (seconds / size).denominator == 1:
            return f"{seconds / size}{unit}"
    return f"{float(seconds)}s"


#: What a window left unset comes to, in steps of the data so it is always a whole number
#: of them. One place, printed by every run (`settings_report`).
DEFAULT_PAST_STEPS = 3
DEFAULT_HORIZON_STEPS = 48
DEFAULT_TEST_HORIZON_STEPS = 288
#: Training windows start a quarter of a horizon apart: enough windows without
#: near-identical ones. Dev and test windows do not overlap.
TRAIN_STRIDES_PER_HORIZON = 4


class TimedWindows:
    """What both configs share: their windows are times, read as steps once the data has
    set `steps_per_hour`."""

    steps_per_hour: Optional[float]
    horizon: Optional[Time]
    test_horizon: Optional[Time]
    train_stride: Optional[Time]
    dev_stride: Optional[Time]
    test_stride: Optional[Time]

    def _steps(self, name: str) -> int:
        return time_steps(getattr(self, name), self.steps_per_hour, name)

    def _time(self, steps: int) -> Time:
        return steps_as_time(steps, self.steps_per_hour)

    def fill_common_windows(self, longest_test_horizon: int) -> None:
        """What each unset window comes to, written back so the run logs it.
        `longest_test_horizon` is what the shortest test file can hold, in steps."""
        if self.horizon is None:
            self.horizon = self._time(DEFAULT_HORIZON_STEPS)
        if self.test_horizon is None:
            if longest_test_horizon < 1:
                raise ValueError(
                    "The shortest test file is too short to hold a test rollout after its "
                    "history; use a longer file, or a shorter `past` or `context`.")
            self.test_horizon = self._time(min(DEFAULT_TEST_HORIZON_STEPS, longest_test_horizon))
        if self.train_stride is None:
            self.train_stride = self._time(max(1, self.horizon_steps // TRAIN_STRIDES_PER_HORIZON))
        if self.dev_stride is None:
            self.dev_stride = self.horizon
        if self.test_stride is None:
            self.test_stride = self.test_horizon

    @property
    def horizon_steps(self) -> int:
        return self._steps("horizon")

    @property
    def test_horizon_steps(self) -> int:
        return self._steps("test_horizon")

    @property
    def train_stride_steps(self) -> int:
        return self._steps("train_stride")

    @property
    def dev_stride_steps(self) -> int:
        return self._steps("dev_stride")

    @property
    def test_stride_steps(self) -> int:
        return self._steps("test_stride")

    def _check_common_windows(self) -> None:
        """Every window a whole number of steps, and positive: run once the data has set
        the timestep and the defaults have been filled."""
        for name in ("horizon", "test_horizon", "train_stride", "dev_stride", "test_stride"):
            if self._steps(name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    def _check_common_times(self) -> None:
        """Every set window spelled as a time: before any data has loaded."""
        for name in ("horizon", "test_horizon", "train_stride", "dev_stride", "test_stride"):
            if getattr(self, name) is not None:
                time_microseconds(getattr(self, name), name)


def fields_set(cfg) -> set[str]:
    """The fields of `cfg` that differ from the dataclass's default: what the caller
    chose, read before a run fills in what it did not."""
    fresh = type(cfg)()
    return {f.name for f in fields(cfg)
            if f.name != "savedir" and getattr(cfg, f.name) != getattr(fresh, f.name)}


def _setting(cfg, name: str) -> str:
    value = getattr(cfg, name)
    if isinstance(value, dict) and len(value) > 3:
        return f"{name}: {len(value)} entries"
    return f"{name}: {value}"


def settings_report(cfg, given: set[str], common: tuple[str, ...]) -> str:
    """Two lines: what the caller set, and what the library chose. `common` is what is
    worth naming when left alone; a setting nobody set outside it stays out of the way,
    and `--print-config` has them all."""
    you_set = [_setting(cfg, name) for name in sorted(given)]
    defaults = [_setting(cfg, name) for name in common if name not in given]
    return ("You set: " + (" · ".join(you_set) or "nothing") + "\n"
            + "Defaults used: " + (" · ".join(defaults) or "none"))


def default_savedir(prefix: str = "temp") -> str:
    """Per-process-unique scratch dir so concurrent runs (e.g. separate
    terminals) never share/clobber the same savedir."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def check_positive(obj: Any, name: str) -> None:
    x = getattr(obj, name)
    if x <= 0:
        raise ValueError(f"{name} must be positive, got {x}")


#: `cfg.progress`: what a training run prints (see `common.progress`).
PROGRESS_LEVELS = ("silent", "summary", "live", "log")


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


def parse_set_args(overrides: list[str]) -> dict[str, Any]:
    """`--set key=value` strings as {key: parsed value}; a repeated key keeps
    its last value."""
    parsed = {}
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Override must be in key=value form: {override}")
        key, raw_value = override.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Override key is empty in '{override}'")
        parsed[key] = _parse_value(raw_value)
    return parsed


def apply_overrides(config_dict: dict, overrides: dict[str, Any]) -> dict:
    unknown = sorted(set(overrides) - set(config_dict))
    if unknown:
        raise ConfigError(f"Override target does not exist: {', '.join(unknown)}")
    return {**config_dict, **overrides}


def load_config(path: Union[str, Path], cls: Type[T],
                overrides: Optional[dict[str, Any]] = None) -> T:
    """A YAML config file as a validated `cls`, with `overrides` (field name
    -> value) applied on top — what `--config` and `--set` do on the CLI."""
    cfg = dict_to_dataclass(load_yaml(Path(path).expanduser()), cls)
    if overrides:
        cfg = dict_to_dataclass(apply_overrides(cfg.asdict(), overrides), cls)
    cfg.validate()
    return cfg


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


def build_cli_parser(
    command_flags: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """The standard config flags, plus a command's own (`command_flags`, built
    with `add_help=False`) so `--help` lists both."""
    parser = argparse.ArgumentParser(
        description="Run with a YAML configuration file.",
        parents=[] if command_flags is None else [command_flags],
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
    command_flags: Optional[argparse.ArgumentParser] = None,
) -> tuple[Any, argparse.Namespace]:
    """Load a config dataclass from a YAML file + CLI --set overrides.

    A command with flags of its own passes them as `command_flags`; they are
    parsed with the standard ones and returned on the same namespace.
    """

    parser = build_cli_parser(command_flags)
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
    cfg = load_config(config_path, cls, parse_set_args(cli_args.set))

    if cli_args.print_config:
        print(yaml.safe_dump(cfg.asdict(), sort_keys=False))
        raise SystemExit(0)

    if cli_args.dump_config:
        dump_path = Path(cli_args.dump_config).expanduser()
        dump_path.write_text(yaml.safe_dump(cfg.asdict(), sort_keys=False),
                             encoding="utf-8")
        raise SystemExit(0)

    return cfg, cli_args
