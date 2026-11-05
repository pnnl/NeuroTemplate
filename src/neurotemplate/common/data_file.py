"""
`data.yaml`: where the record is and which column plays which role, once, for both
the model and the controller.

    train: data/train.csv        # one path, or a list: no window spans two files
    dev: data/dev.csv
    test: data/test.csv
    columns:
      Y: [...]                   # what the model predicts
      U: [...]                   # what a controller writes
      D: [...]                   # what is known without the plant

A column no role claims is allowed: it is how a band or a tariff sits in the record.
The model never reads it, a controller's objective or window may name it, and each run
lists what it leaves unread.

Relative paths are read from the directory the file is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

from neurotemplate.common.config import ConfigError

ROLES = ("Y", "U", "D")
SPLITS = ("train", "dev", "test")
#: What the CSV loader adds to every record: elapsed seconds, `t` in every window.
TIME_COLUMN = "sec_elapsed"


@dataclass(frozen=True)
class DataFile:
    """A parsed `data.yaml`. Every split is a list of one or more paths."""
    path: Path
    train: list[str]
    dev: list[str]
    test: list[str]
    columns: dict[str, list[str]]

    def splits(self) -> dict[str, list[str]]:
        return {"train": self.train, "dev": self.dev, "test": self.test}


def _paths(name: str, value, base: Path, source: Path) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values or not all(
            isinstance(v, str) and v for v in values):
        raise ConfigError(f"{source}: `{name}` must be a path or a list of paths; got {value!r}.")
    return [str(base / v) for v in values]


def load_data_file(path: Union[str, Path]) -> DataFile:
    """`path` parsed and its shape checked; whether the files and columns exist is for
    whoever reads them."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"{source} not found, in {Path.cwd()}. It says where the record is and which "
            "column plays which role: `nt init` writes one.")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    unknown = sorted(set(raw) - {*SPLITS, "columns"})
    missing = [name for name in SPLITS if name not in raw]
    if unknown or missing:
        raise ConfigError(f"{source}: it holds {list(SPLITS)} and `columns`; unknown keys "
                          f"{unknown}, missing {missing}.")
    columns = raw.get("columns") or {}
    stray = sorted(set(columns) - set(ROLES))
    if stray:
        raise ConfigError(f"{source}: `columns` holds the roles {list(ROLES)}; it also has {stray}.")
    return DataFile(source, *(_paths(name, raw[name], source.parent, source) for name in SPLITS),
                    {role: list(columns.get(role) or []) for role in ROLES})


def _listing(columns: list[str]) -> str:
    """Every column as `data.yaml`'s `columns`, ready to paste and sort into roles."""
    lines = ["columns:", "  Y:"]
    lines += [f'    - "{c}"' for c in columns]
    return "\n".join(lines + ["  U: []", "  D: []"])


def nm_mapping(data_file: DataFile, record_columns: list[str]) -> dict[str, list[str]]:
    """`{"Y", "U", "D", "t"}` from `data_file`, every name checked against the record's
    columns and every problem listed at once. With no roles assigned at all this is the
    first thing a project hits, so it stops and prints what to paste."""
    available = [c for c in record_columns if c != TIME_COLUMN]
    named = [c for role in ROLES for c in data_file.columns[role]]
    if not named:
        raise SystemExit(
            f"{data_file.path} assigns no columns yet. The record has {len(available)}; "
            "paste this over `columns` and move each to its role. Y is what the model "
            "predicts, U what a controller writes, D what is known without the plant, and "
            f"a column you leave out is simply not read.\n\n{_listing(available)}")
    problems = []
    unknown = [c for c in named if c not in record_columns]
    if unknown:
        problems.append(f"columns not in the record: {unknown}")
    repeated = sorted({c for c in named if named.count(c) > 1})
    if repeated:
        problems.append(f"columns in more than one role (or twice in one): {repeated}")
    if not data_file.columns["Y"]:
        problems.append("Y is empty, so the model has nothing to predict")
    if problems:
        raise ValueError(f"{data_file.path}:\n  - " + "\n  - ".join(problems)
                         + f"\nThe record has: {available}")
    return {**data_file.columns, "t": [TIME_COLUMN]}


def unclaimed(nm: dict[str, list[str]], record_columns: list[str]) -> list[str]:
    """The record's columns no role claims, in file order."""
    claimed = {c for role in ROLES for c in nm[role]} | {TIME_COLUMN}
    return [c for c in record_columns if c not in claimed]
