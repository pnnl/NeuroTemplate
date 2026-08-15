"""
Splitting a multi-output model into independent per-zone sub-models.

A *zone* is a group of Y channels that gets its own sub-model: its own
parameters and, for stateful families like `2r2c`, its own hidden state.
Zones are independent by construction — no term in one zone's update reads
another zone's state. Inter-zone conduction is deliberately out of scope.

What a zone owns
================

  Y  its own channels, and only its own. Every Y channel belongs to exactly
     one zone.
  U  its own actuation, plus any U channel no zone claimed. Unclaimed U
     channels are treated as *shared* and handed to every zone, which is
     what you want for a single air handler serving several zones.
  D  all of it. Weather is shared by definition.

Where the split comes from
==========================

An optional `zone_mapping` hook in the application's modeling hooks file:

    def zone_mapping(nm_mapping, cfg):
        return {
            "north": {"Y": ["North Zone Air Temperature [°C]"],
                      "U": ["North Cooling Setpoint [°C]"]},
            "south": {"Y": ["South Zone Air Temperature [°C]"],
                      "U": ["South Cooling Setpoint [°C]"]},
        }

With no hook, `build_zones` returns a single zone holding every channel —
which for `2r2c` still means one independent RC network per Y channel, since
that block is per-channel internally. The hook is what restricts *which
setpoint drives which zone*; you only need it when that restriction matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from neurotemplate._typing import beartype


@dataclass(frozen=True)
class Zone:
    """One independent sub-model's slice of the Y and U channel lists.

    `y_idx` and `u_idx` index into `nm_mapping["Y"]` and `nm_mapping["U"]`.
    D is not stored: every zone reads every D channel.
    """

    name: str
    y_idx: tuple[int, ...]
    u_idx: tuple[int, ...]

    @property
    def ny(self) -> int:
        return len(self.y_idx)

    @property
    def nu(self) -> int:
        return len(self.u_idx)


def _resolve(columns: list[str], names: list[str], zone: str, key: str) -> list[int]:
    """Map column names to their positions in `columns`, or raise."""
    missing = [n for n in names if n not in columns]
    if missing:
        raise ValueError(
            f"zone_mapping: zone {zone!r} lists {key} column(s) {missing} that "
            f"are not in nm_mapping[{key!r}] ({columns})."
        )
    return [columns.index(n) for n in names]


@beartype
def build_zones(
    nm_mapping: dict[str, list[str]],
    zone_spec: dict[str, dict[str, list[str]]] | None = None,
) -> list[Zone]:
    """Turn a `zone_mapping` hook result into validated index slices.

    `zone_spec` is `{zone_name: {"Y": [...], "U": [...]}}`. Passing None (no
    hook) gives one zone holding every Y and U channel.

    Raises ValueError if the Y channels are not partitioned exactly: every Y
    channel in one and only one zone. U channels may be left unclaimed, in
    which case every zone gets them.
    """
    y_cols, u_cols = nm_mapping["Y"], nm_mapping["U"]

    if zone_spec is None:
        return [Zone("all", tuple(range(len(y_cols))), tuple(range(len(u_cols))))]

    if not zone_spec:
        raise ValueError("zone_mapping returned an empty mapping; return None instead.")

    claimed_y: list[int] = []
    claimed_u: list[int] = []
    per_zone: list[tuple[str, list[int], list[int]]] = []
    for name, spec in zone_spec.items():
        y_idx = _resolve(y_cols, spec.get("Y", []), name, "Y")
        u_idx = _resolve(u_cols, spec.get("U", []), name, "U")
        if not y_idx:
            raise ValueError(f"zone_mapping: zone {name!r} claims no Y channels.")
        claimed_y += y_idx
        claimed_u += u_idx
        per_zone.append((name, y_idx, u_idx))

    duplicated = sorted({y_cols[i] for i in claimed_y if claimed_y.count(i) > 1})
    if duplicated:
        raise ValueError(
            f"zone_mapping: Y column(s) {duplicated} are claimed by more than "
            "one zone. Each Y channel belongs to exactly one zone."
        )
    unclaimed_y = [c for i, c in enumerate(y_cols) if i not in claimed_y]
    if unclaimed_y:
        raise ValueError(
            f"zone_mapping: Y column(s) {unclaimed_y} are not claimed by any "
            "zone. Every Y channel belongs to exactly one zone."
        )
    duplicated_u = sorted({u_cols[i] for i in claimed_u if claimed_u.count(i) > 1})
    if duplicated_u:
        raise ValueError(
            f"zone_mapping: U column(s) {duplicated_u} are claimed by more than "
            "one zone. Leave a U channel out entirely to share it with all zones."
        )

    # Whatever no zone asked for is shared equipment: hand it to everyone.
    shared_u = [i for i in range(len(u_cols)) if i not in claimed_u]
    return [
        Zone(name, tuple(y_idx), tuple(sorted(u_idx + shared_u)))
        for name, y_idx, u_idx in per_zone
    ]
