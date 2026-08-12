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


# Tariff used by the price/cost/peak tests. A stand-in for a real historical
# record: an evening peak whose height varies day to day, so a window's price
# is NOT a function of its position in the day alone — which is the property
# that stops a policy from learning the clock instead of the price.
PRICE_SCALE = 0.30
CHEAP, EXPENSIVE = 0.12, 0.48
PRICE_STEPS_PER_HOUR = 12
PRICE_DAYS = 30


def _price_series() -> torch.Tensor:
    """[T, 1] deterministic $/kWh series, upsampled to a 5-minute grid."""
    hours = torch.arange(PRICE_DAYS * 24, dtype=torch.float32)
    on_peak = ((hours % 24) >= 16) & ((hours % 24) < 21)
    # Peak height cycles across days with an irrational-ish period, so no two
    # consecutive days share a price profile.
    height = CHEAP + (EXPENSIVE - CHEAP) * (
        0.5 + 0.5 * torch.cos(hours // 24 * 1.1)
    )
    dollars = torch.where(on_peak, height, torch.full_like(hours, CHEAP))
    return dollars.repeat_interleave(PRICE_STEPS_PER_HOUR)[:, None]


PRICE_SERIES = _price_series()


def sampled_price(n_samples, nsteps, generator=None):
    """[n_samples, nsteps, 1] $/kWh — random windows out of PRICE_SERIES.

    Offsets are drawn independently of the weather windows, which is the
    contract `build_price` documents.
    """
    starts = torch.randint(0, PRICE_SERIES.shape[0] - nsteps + 1, (n_samples,),
                           generator=generator)
    idx = starts[:, None] + torch.arange(nsteps)[None, :]
    return PRICE_SERIES[idx]


def flat_price(value):
    """A constant tariff, for tests that want a hand-computable cost."""
    def signal(n_samples, nsteps, generator=None):
        return torch.full((n_samples, nsteps, 1), float(value))
    return signal


def alternating_price(n_samples, nsteps, generator=None):
    """A tariff that changes every step, so a one-step shift is detectable."""
    steps = (torch.arange(nsteps, dtype=torch.float32) % 2) * EXPENSIVE + CHEAP
    return steps.reshape(1, nsteps, 1).repeat(n_samples, 1, 1)


def install_price_hook(monkeypatch, signal=sampled_price, scale=PRICE_SCALE):
    """Give the fixture hooks module a tariff for the duration of one test.

    Price plumbing is off by default (no `price_signal` attribute), which is
    what keeps every other test a regression guard on "a run without a tariff
    is unchanged".
    """
    import control_user_defined as cud
    monkeypatch.setattr(cud, "PRICE_SCALE", scale, raising=False)
    monkeypatch.setattr(cud, "price_signal", signal, raising=False)


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
