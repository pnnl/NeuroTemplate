"""Config for the DPC control pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (  # noqa: F401  re-exported
    _check_paths_field,
    check_max,
    check_non_negative,
    check_one_of,
    check_positive,
    default_savedir,
    load_config_from_cli,
    paths_as_list,
    CONTROL_CONFIG_DIR,
)
from neurotemplate.modeling.config import MAX_IN_STEPS

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

    # ── Energy tariff (all four required together, or none) ───────
    # One CSV per split — a timestamp column plus one price column ($/kWh).
    # Sampled independently of the weather windows (see control.scenarios),
    # so a price record does not need to span the plant record, and the test
    # file is where to put the price swings the eval should be judged on.
    train_price_path: Optional[str] = None
    dev_price_path: Optional[str] = None
    test_price_path: Optional[str] = None
    # Which CSV column carries price. Unset infers the sole non-timestamp
    # column; set it when a file has more than one.
    price_column: Optional[str] = None
    # How many plant timesteps one price-file row spans (12 = hourly price
    # over 5-minute plant data). 1 = price file is already at the plant's
    # resolution.
    price_repeat: int = 1
    # The tariff arrives in physical units ($/kWh); the dict carries
    # price / price_scale so the policy sees an O(1) input, and the cost
    # term multiplies the scale back in. Deliberately a pure scaling with no
    # shift, so a zero price stays a zero cost.
    price_scale: Optional[float] = None

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
    # Setpoint-smoothness weight. A scalar applies to every actuator; a list
    # (one entry per U column, in nm_mapping order) or a dict keyed by column
    # name weights them separately — worth doing as soon as two actuators
    # differ in authority, since one pooled penalty is then necessarily
    # mis-weighted for one of them.
    du_weight: Union[float, list[float], dict[str, float]] = 10
    # Effort/energy term weight. > 0 requires control_user_defined.power_metric
    # to be implemented (fail-fast at problem build).
    energy_weight: float = 0.1
    # Energy-cost weight: multiplies mean_t(price * power), in $/h. > 0
    # additionally requires the control_user_defined.price_signal hook.
    cost_weight: float = 0.0
    # Peak-demand weight: multiplies the per-window power mean M_q(power),
    # averaged over the batch. Same units as the energy term.
    peak_weight: float = 0.0
    # Exponent of that power mean. 1 = plain mean (no peak preference),
    # 2 = RMS, larger = closer to a hard max. Must be >= 1.
    peak_q: float = 4.0

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
    # Half-width of the residual box, in physical units (°C). None disables the
    # residual entirely (a plain policy over the actuation box). Set it and the
    # policy commands `reference_control(...) + δ` with |δ| <= residual_delta,
    # which caps how far a surrogate error can move the deployed setpoint.
    # 0.0 emits the reference rule exactly — a useful floor to train nothing
    # and score. Requires the control_user_defined.reference_control hook.
    residual_delta: Optional[float] = None

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
        check_max(self, "in_steps", MAX_IN_STEPS)

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

        price_paths = {
            "train_price_path": self.train_price_path,
            "dev_price_path": self.dev_price_path,
            "test_price_path": self.test_price_path,
        }
        n_set = sum(v is not None for v in price_paths.values())
        if 0 < n_set < 3:
            missing = [k for k, v in price_paths.items() if v is None]
            raise ValueError(
                f"train_price_path/dev_price_path/test_price_path must be set "
                f"together or not at all (missing: {missing}). A tariff is "
                "sampled per split, so a split with no price file would "
                "silently see no cost term."
            )
        if n_set == 3 and self.price_scale is None:
            raise ValueError(
                "A price_*_path is set but price_scale is not. The tariff "
                "arrives in physical $/kWh; price_scale is required so the "
                "dict can carry price / price_scale as an O(1) policy input."
            )
        if n_set == 0 and self.price_scale is not None:
            raise ValueError(
                "price_scale is set but no train_price_path/dev_price_path/"
                "test_price_path is. Set all three price paths to enable a "
                "tariff, or unset price_scale."
            )
        if self.price_scale is not None and self.price_scale <= 0:
            raise ValueError(
                f"price_scale must be positive, got {self.price_scale}"
            )

        for name in [
            "batch_size", "train_pred_steps", "in_steps",
            "log2_policy_hidden", "policy_layers",
            "epochs", "epoch_verbose", "lr",
            "test_window", "test_stride", "band_width_max",
            "train_band_sub_samples", "test_band_sub_samples",
            "price_repeat",
        ]:
            check_positive(self, name)

        for name in [
            "d_preview", "warmup", "patience",
            "comfort_weight", "energy_weight", "cost_weight", "peak_weight",
            "band_width_min", "setpoint_step",
        ]:
            check_non_negative(self, name)

        # q < 1 is not a peak preference at all: M_q below 1 rewards spiky
        # profiles, so it would quietly optimise the opposite of the request.
        if self.peak_q < 1:
            raise ValueError(
                f"peak_q must be >= 1, got {self.peak_q}. 1 is the plain mean; "
                "larger values approach a hard max."
            )

        if self.residual_delta is not None:
            if self.residual_delta < 0:
                raise ValueError(
                    f"residual_delta must be >= 0, got {self.residual_delta}. "
                    "0 emits the reference rule exactly; null disables the "
                    "residual."
                )
            # The single most likely way to waste a campaign on this feature:
            # every δ snaps back to zero and you get a rule that trained for an
            # hour, with nothing on any report to say so.
            if 0 < self.residual_delta < self.setpoint_step:
                raise ValueError(
                    f"residual_delta ({self.residual_delta}) is smaller than "
                    f"setpoint_step ({self.setpoint_step}), so every residual "
                    "quantizes back to zero and the policy can only ever emit "
                    "the reference rule. Raise it to at least one grid step, "
                    "or set setpoint_step: 0."
                )

        # du_weight is per-channel-capable, so it is checked by value rather
        # than by field. Column names can only be resolved against the
        # surrogate's U, which happens in closed_loop.resolve_du_weights.
        weights = (self.du_weight.values() if isinstance(self.du_weight, dict)
                   else self.du_weight if isinstance(self.du_weight, list)
                   else [self.du_weight])
        if any(w < 0 for w in weights):
            raise ValueError(f"du_weight must be >= 0, got {self.du_weight}")

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
