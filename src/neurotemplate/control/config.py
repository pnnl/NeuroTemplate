"""Config for the DPC control pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (  # noqa: F401  re-exported
    _check_paths_field,
    check_non_negative,
    check_one_of,
    check_positive,
    default_savedir,
    load_config_from_cli,
    paths_as_list,
    CONTROL_CONFIG_DIR,
)

from neurotemplate._typing import beartype


@dataclass
class ControlConfig:
    """Knobs for loading a frozen surrogate and training a DPC policy."""

    # ── Sys-id model to control against ──────────────────────────
    # MLflow run id of the trained dynamics surrogate (required).
    sysid_run_id: str = ""
    # Optional override for the run's artifacts directory.
    run_dir: Optional[str] = None
    # Optional explicit weights path (else best_model_state_dict.pth).
    weights: Optional[str] = None

    # ── Data (all three required; never inherited from the sys-id run) ───
    # Each may be a single path string or a list of path strings. With a
    # list, files are loaded independently and their windows are pooled
    # into one DataLoader; no rollout window ever spans two source files.
    # Normalize via `paths_as_list(...)` at the entry point so downstream
    # code sees a uniform `list[df]` shape.
    train_data_path: Union[str, list[str]] = ""
    dev_data_path: Union[str, list[str]] = ""
    test_data_path: Union[str, list[str]] = ""
    batch_size: int = 256

    # Optional path or dotted module name for the control hooks file.
    # If unset, the library looks for `control_user_defined.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ── Control horizon / policy history ─────────────────────────
    # Closed-loop training rollout length (pure prediction; window = hmax + train_pred_steps).
    train_pred_steps: int = 48
    # Controller input history window (H_ctrl).
    in_steps: int = 12
    # Future-D steps the policy previews (0 = current only).
    d_preview: int = 0

    # ── Comfort band (sampled time-varying signal, °C) ───────────
    # Per segment, width ~ U[band_width_min, band_width_max] and lower bound
    # ~ U[band_lo, band_hi - width]. `*_band_sub_samples` controls how many
    # equal-ish contiguous segments make up a window: 1 = constant band,
    # k > 1 = (k-1) step-jumps inside the window. test_band_* (lo/hi/widths)
    # default to the train values; set them to override at evaluation time.
    band_lo: float = 23.0
    band_hi: float = 27.0
    band_width_min: float = 0.5
    band_width_max: float = 2.0
    train_band_sub_samples: int = 1
    test_band_sub_samples: int = 4
    test_band_lo: Optional[float] = None
    test_band_hi: Optional[float] = None
    test_band_width_min: Optional[float] = None
    test_band_width_max: Optional[float] = None

    # ── Objective weights ────────────────────────────────────────
    comfort_weight: float = 50.0
    du_weight: float = 10
    # Effort/energy term weight. > 0 requires control_user_defined.power_metric
    # to be implemented (fail-fast at problem build).
    effort_weight: float = 0.1

    # ── Policy network ───────────────────────────────────────────
    # Network family (see control.policy.policy_catalog). "linear" ignores
    # policy_layers / log2_policy_hidden / policy_nonlin and produces a single
    # saturated affine feedback law.
    policy: str = "mlp"
    policy_nonlin: str = "gelu"
    # How the network output is squashed into [umin, umax]: "sigmoid_scale"
    # (smooth, never exactly reaches the limits) or "relu_clamp" (hard clip).
    policy_bound_method: str = "sigmoid_scale"
    log2_policy_hidden: int = 6
    policy_layers: int = 2
    # Snap policy output to a fixed setpoint grid (°C). E.g. 0.5 = half-degree
    # thermostat steps. Straight-through estimator on backward, so gradients
    # still flow as if the output were continuous. 0 disables quantization.
    setpoint_step: float = 0.5

    # ── Training ─────────────────────────────────────────────────
    epochs: int = 300
    epoch_verbose: int = 1
    warmup: int = 5
    patience: int = 25
    lr: float = 0.001

    # ── Evaluation (long closed-loop rollout) ────────────────────
    # Total window length for eval (hmax warmup IS counted in this; actual
    # rollout on plots is test_window - hmax). Matches sys-id semantics.
    test_window: int = 287
    test_stride: int = 288

    # ── Runtime ──────────────────────────────────────────────────
    seed: int = -1
    device: Optional[str] = None
    verbose: bool = True

    # ── Logging ──────────────────────────────────────────────────
    savedir: str = field(default_factory=default_savedir)
    location: str = "sqlite:///mlflow.db"
    exp: str = "DPC Training"
    run: str = ""
    plots_dir: str = "plots"
    plot: bool = True

    def validate(self) -> None:
        # Deferred import: policy.py pulls in Neuromancer, and config is
        # imported by the CLI before anything else. Same pattern as
        # train_band()/test_band() below.
        from neurotemplate.control.policy import (
            POLICY_BOUND_METHODS, POLICY_NONLIN, policy_catalog,
        )

        if not self.sysid_run_id:
            raise ValueError("sysid_run_id must be set (MLflow run id of the "
                             "trained dynamics surrogate)")

        check_one_of(self, "policy", list(policy_catalog))
        check_one_of(self, "policy_nonlin", list(POLICY_NONLIN))
        check_one_of(self, "policy_bound_method", POLICY_BOUND_METHODS)

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

        for name in [
            "batch_size", "train_pred_steps", "in_steps",
            "log2_policy_hidden", "policy_layers",
            "epochs", "epoch_verbose", "lr",
            "test_window", "test_stride", "band_width_max",
            "train_band_sub_samples", "test_band_sub_samples",
        ]:
            check_positive(self, name)

        for name in [
            "d_preview", "warmup", "patience",
            "comfort_weight", "du_weight", "effort_weight",
            "band_width_min", "setpoint_step",
        ]:
            check_non_negative(self, name)

        if self.band_width_min > self.band_width_max:
            raise ValueError(
                f"band_width_min ({self.band_width_min}) must be <= "
                f"band_width_max ({self.band_width_max})"
            )
        if self.band_lo > self.band_hi:
            raise ValueError(
                f"band_lo ({self.band_lo}) must be <= "
                f"band_hi ({self.band_hi})"
            )
        tb = self.test_band()
        if tb.width_min > tb.width_max:
            raise ValueError(
                f"resolved test band width_min ({tb.width_min}) must be <= "
                f"width_max ({tb.width_max})"
            )
        if tb.lo > tb.hi:
            raise ValueError(
                f"resolved test band lo ({tb.lo}) must be <= hi ({tb.hi})"
            )

    def train_band(self):
        from neurotemplate.control.scenarios import BandSpec
        return BandSpec(
            lo=self.band_lo, hi=self.band_hi,
            width_min=self.band_width_min, width_max=self.band_width_max,
            sub_samples=self.train_band_sub_samples,
        )

    def test_band(self):
        from neurotemplate.control.scenarios import BandSpec
        lo = self.band_lo if self.test_band_lo is None else self.test_band_lo
        hi = self.band_hi if self.test_band_hi is None else self.test_band_hi
        wmin = (self.band_width_min if self.test_band_width_min is None
                else self.test_band_width_min)
        wmax = (self.band_width_max if self.test_band_width_max is None
                else self.test_band_width_max)
        return BandSpec(
            lo=lo, hi=hi, width_min=wmin, width_max=wmax,
            sub_samples=self.test_band_sub_samples,
        )

    def asdict(self) -> dict:
        return asdict(self)


__all__ = [
    "ControlConfig",
    "load_config_from_cli",
]
