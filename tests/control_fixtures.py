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

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.common.data_utils import compute_norm_stats, normalize_df
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.control.config import ControlConfig

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
    return pd.DataFrame(data)


def make_norm_stats(raw_df: pd.DataFrame) -> dict:
    return compute_norm_stats(raw_df, norm_type="z_score")


def make_sysid_cfg(in_steps: int = 4, model: str = "mlp") -> Config:
    cfg = Config()
    cfg.model = model
    cfg.in_steps = in_steps
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.residual = True
    cfg.device = torch.device("cpu")
    return cfg


def make_control_cfg(**overrides) -> ControlConfig:
    cfg = ControlConfig()
    cfg.sysid_run_id = "test-run"
    cfg.in_steps = 3
    cfg.train_pred_steps = 6
    # hmax = max(sysid_cfg.in_steps=4, cfg.in_steps=3) = 4; test_window includes warmup.
    cfg.test_window = 12
    cfg.test_stride = 5
    cfg.batch_size = 4
    cfg.epochs = 1
    cfg.log2_policy_hidden = 3
    cfg.policy_layers = 1
    cfg.seed = 0
    cfg.device = torch.device("cpu")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_frozen_dynamics(sysid_cfg: Config, nm_mapping: dict):
    """Real SystemPreview from src, frozen — stand-in for a loaded model."""
    dyn_sys = get_dynamics(sysid_cfg, nm_mapping)
    for p in dyn_sys.parameters():
        p.requires_grad_(False)
    dyn_sys.eval()
    return dyn_sys, dyn_sys.nodes[0]


def full_setup(**control_overrides):
    """One-call bundle used by most tests."""
    nm = make_nm_mapping()
    raw = make_raw_df()
    norm_stats = make_norm_stats(raw)
    norm_df = normalize_df(raw, norm_stats)
    sysid_cfg = make_sysid_cfg()
    cfg = make_control_cfg(**control_overrides)
    dyn_sys, dyn_node = make_frozen_dynamics(sysid_cfg, nm)
    return dict(
        nm=nm, raw=raw, norm_stats=norm_stats, norm_df=norm_df,
        sysid_cfg=sysid_cfg, cfg=cfg, dyn_sys=dyn_sys, dyn_node=dyn_node,
    )
