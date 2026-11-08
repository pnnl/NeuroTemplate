"""
Self-contained fixtures for control_src tests.

No MLflow needed: build a real (untrained) sys-id SystemPreview via
src/models.get_dynamics and freeze it — this exercises the actual Node /
input_map integration the closed loop depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import yaml

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.data_utils import compute_norm_stats
from neurotemplate.modeling.dynamics import _rollout_system, get_dynamics
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.losses import extend_norm_stats, resolve_objective
from model_fixtures import model_cfg, steps, uniform_past

# Single-zone Modelica RTU strip-mall dataset. Y is one zone temp plus the
# RTU thermal/flow state plus the two metered RTU power channels; U is a
# single cooling setpoint.
ZONE = "Zone Air Temperature [°C]"
SUPPLY_T = "RTU Supply Air Temperature [°C]"
SUPPLY_F = "RTU Supply Air Flow [m³/s]"
STAGE = "Cooling Stage Signal [-]"
FAN_P = "RTU Fan Power [W]"
DX_P = "RTU DX Power [W]"
SETPT = "Zone Cooling Setpoint [°C]"
DIST = [
    "Outdoor Dry-Bulb Temperature [°C]",
    "Global Horizontal Irradiance [W/m²]",
    "Occupancy Signal [-]",
]

Y_COLS = [ZONE, SUPPLY_T, SUPPLY_F, STAGE, FAN_P, DX_P]
U_COLS = [SETPT]


def make_nm_mapping() -> dict[str, list[str]]:
    return {"Y": list(Y_COLS), "U": list(U_COLS), "D": list(DIST),
            "t": ["sec_elapsed"]}


# Where the fixture plant record starts, in wall-clock time.
PLANT_ORIGIN = "2024-08-02 06:35"


def make_raw_df(n_rows: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        ZONE:     24.0 + rng.normal(0, 1.0, n_rows),
        SUPPLY_T: 18.0 + rng.normal(0, 3.0, n_rows),
        SUPPLY_F: np.clip(0.5 + rng.normal(0, 0.3, n_rows), 0.0, None),
        STAGE:    rng.integers(0, 3, n_rows).astype(float),
        FAN_P:    np.clip(rng.normal(300, 200, n_rows), 0.0, None),
        DX_P:     np.clip(rng.normal(1100, 1500, n_rows), 0.0, None),
        SETPT:    26.0 + rng.normal(0, 1.5, n_rows),
        DIST[0]:  25.0 + rng.normal(0, 3.0, n_rows),
        DIST[1]:  np.clip(rng.normal(230, 290, n_rows), 0, None),
        DIST[2]:  rng.integers(0, 2, n_rows).astype(float),
        "sec_elapsed": np.arange(n_rows, dtype=float) * 300.0,
    }
    # A DatetimeIndex like the one `get_data` sets, so anything reading the
    # plant's wall clock (closed-loop plots, price_align: day) sees what it
    # would see in a real run. Deliberately not midnight: a frame that starts
    # at 00:00 hides every off-by-an-origin bug.
    return pd.DataFrame(data, index=pd.date_range(
        PLANT_ORIGIN, periods=n_rows, freq="5min", name="timestamp",
    ))


def make_norm_stats(raw_df: pd.DataFrame) -> dict:
    return compute_norm_stats(raw_df, norm_type="z_score")


def make_sysid_cfg(history_steps: int = 4, model: str = "mlp") -> Config:
    return model_cfg(model=model, past=uniform_past(make_nm_mapping(), history_steps))


# The reference columns: a comfort band on the zone, and a tariff.
LOW = "Comfort Low [°C]"
HIGH = "Comfort High [°C]"
TARIFF = "Tariff [$/kWh]"
TARIFF_SCALE = 0.30
CHEAP, EXPENSIVE = 0.12, 0.48

#: The comfort band and setpoint smoothing — what every control test trains
#: against unless it says otherwise.
BAND_LOSSES = {
    "comfort_lower": {"column": LOW, "target": ZONE, "map": "relu", "norm": 2},
    "comfort_upper": {"column": HIGH, "target": ZONE, "map": "nrelu", "norm": 2},
    "du": {"column": SETPT, "map": "first_difference", "norm": 1},
}
BAND_GAINS = {"comfort_lower": 50.0, "comfort_upper": 50.0, "du": 10.0}

#: The three reductions of power, for the grid-response tests.
POWER_LOSSES = {
    "energy": {"column": "power", "norm": 1},
    "cost": {"column": "power", "norm": 1, "weight": TARIFF},
    "peak": {"column": "power", "norm": 4},
}


def make_reference_df(plant: pd.DataFrame, segment: int = 6,
                      seed: int = 0) -> pd.DataFrame:
    """A reference record for `plant`: same timestamps, a piecewise-constant
    comfort band (a new width and floor every `segment` steps, inside
    [23, 27] °C), and a tariff alternating every step, so a one-step
    misalignment between price and power changes the cost."""
    rng = np.random.default_rng(seed)
    n = len(plant)
    n_seg = -(-n // segment)
    widths = rng.uniform(0.5, 2.0, n_seg)
    lows = 23.0 + (4.0 - widths) * rng.uniform(0.0, 1.0, n_seg)
    low = np.repeat(lows, segment)[:n]
    high = np.repeat(lows + widths, segment)[:n]
    tariff = np.where(np.arange(n) % 2 == 0, CHEAP, EXPENSIVE)
    return pd.DataFrame({LOW: low, HIGH: high, TARIFF: tariff}, index=plant.index)


def make_policy_window(cfg: ControlConfig, past: int = 2, u_past: int = 0,
                       future: dict[str, int] | None = None) -> dict:
    """A `policy_window` over the fixture's columns and `cfg`'s reference
    columns: every Y and D column `past` steps back, every U column `u_past`,
    and each D and reference column `future[column]` steps ahead (0 if unlisted)."""
    nm = make_nm_mapping()
    future = future or {}
    table = {c: {"past": steps(past)} for c in nm["Y"]}
    table.update({c: {"past": steps(u_past)} for c in nm["U"]})
    table.update({c: {"past": steps(past), "future": steps(future.get(c, 0))} for c in nm["D"]})
    table.update({c: {"past": 0, "future": steps(future.get(c, 0))}
                  for c in resolve_objective(cfg, nm).r_columns})
    return table


def make_control_cfg(future: dict[str, int] | None = None, u_past: int = 0,
                     **overrides) -> ControlConfig:
    """The fixture's control config. Unless `overrides` sets `policy_window`,
    it is `make_policy_window` with `future` and `u_past`, built after the
    overrides so it follows the objective they set."""
    cfg = ControlConfig()
    cfg.sysid_run_id = "test-run"
    cfg.losses = {k: dict(v) for k, v in BAND_LOSSES.items()}
    cfg.gains = dict(BAND_GAINS)
    cfg.steps_per_hour = 12.0
    cfg.horizon = steps(6)
    # hmax = max(sysid_cfg.history_steps=4, cfg.history_steps=3) = 4, so a 12-row window.
    cfg.test_horizon = steps(8)
    cfg.test_stride = steps(5)
    cfg.dev_stride = steps(3)
    cfg.batch_size = 4
    cfg.iters = 2
    cfg.start_box = {}   # starts as recorded; the box has its own test
    cfg.solver["max_iters"] = 2  # the plan runs; its convergence is test_plan's
    cfg.policy = "mlp"      # the family with more to get wrong; linear is tested on its own
    cfg.seed = 0
    cfg.device = torch.device("cpu")
    for k, v in overrides.items():
        if k == "solver":
            cfg.solver.update(v)
        else:
            setattr(cfg, k, v)
    if "policy_window" not in overrides:
        cfg.policy_window = make_policy_window(cfg, u_past=u_past, future=future)
    return cfg


def make_frozen_dynamics(sysid_cfg: Config, nm_mapping: dict, norm_stats: dict):
    """Real SystemPreview from src, frozen — stand-in for a loaded model,
    normalising with the plant's `norm_stats` as a trained one would."""
    dyn_sys = get_dynamics(sysid_cfg, nm_mapping, norm_stats)
    for p in dyn_sys.parameters():
        p.requires_grad_(False)
    dyn_sys.eval()
    return dyn_sys, dyn_sys.nodes[0]


class TrackingZone(torch.nn.Module):
    """A plant of the user's own (`modeling.contract`): the zone closes half
    its gap to the cooling setpoint every step, every other output holds.
    Stable and responsive, so a planner has something to plan."""

    def forward(self, y, u, d):
        nxt = y[:, -1].clone()
        nxt[:, 0] = nxt[:, 0] + 0.5 * (u[:, -1, 0] - nxt[:, 0])
        return nxt


def make_tracking_plant(sysid_cfg: Config, nm_mapping: dict):
    """`TrackingZone` as a frozen surrogate: (dyn_sys, dyn_node)."""
    plant = TrackingZone()
    dyn_sys = _rollout_system(plant, sysid_cfg, nm_mapping)
    return dyn_sys.eval(), dyn_sys.nodes[0]


def write_data_file(tmp_path, record_path, nm: dict | None = None):
    """A `data.yaml` in `tmp_path` reading `record_path` for every split, with the
    fixture's column roles; returns its path."""
    nm = nm or make_nm_mapping()
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({
        "train": str(record_path), "dev": str(record_path), "test": str(record_path),
        "columns": {role: nm[role] for role in ("Y", "U", "D")},
    }, allow_unicode=True), encoding="utf-8")
    return path


def write_splits(cfg, tmp_path, plant: pd.DataFrame,
                 references: list[pd.DataFrame]) -> None:
    """The record with the first reference's columns in it, read by every split, and
    the rest of the references as draws."""
    record = plant.drop(columns="sec_elapsed").join(references[0]).rename_axis("timestamp")
    path = tmp_path / "plant.csv"
    record.to_csv(path)
    cfg.data_file = str(write_data_file(tmp_path, path))
    cfg.draws = []
    for i, ref in enumerate(references[1:], start=1):
        draw = tmp_path / f"draw_{i}.csv"
        ref.rename_axis("timestamp").to_csv(draw)
        cfg.draws.append(str(draw))


def full_setup(reference: pd.DataFrame | None = None, **control_overrides):
    """One-call bundle used by most tests.

    `norm_stats` carries the reference columns and `frame` is the plant
    frame with its reference columns joined — what `load_control_splits`
    hands the loaders.
    """
    nm = make_nm_mapping()
    plant = make_raw_df()
    sysid_cfg = make_sysid_cfg()
    cfg = make_control_cfg(**control_overrides)
    norm_stats = extend_norm_stats(resolve_objective(cfg, nm), cfg,
                                   make_norm_stats(plant))
    raw = plant.join(make_reference_df(plant) if reference is None else reference)
    dyn_sys, dyn_node = make_frozen_dynamics(sysid_cfg, nm, make_norm_stats(plant))
    return dict(
        nm=nm, plant=plant, raw=raw, norm_stats=norm_stats, frame=raw,
        sysid_cfg=sysid_cfg, cfg=cfg, dyn_sys=dyn_sys, dyn_node=dyn_node,
    )
