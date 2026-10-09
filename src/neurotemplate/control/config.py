"""Config for the DPC control pipeline."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (
    _check_paths_field,
    check_non_negative,
    check_one_of,
    check_positive,
    default_savedir,
)
from neurotemplate.modeling.config import IN_STEPS_WARN_THRESHOLD


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

    # ── Reference files (all three together, or none) ─────────────
    # The columns the objective names that are not the surrogate's — a comfort
    # band, a setpoint schedule, a tariff. One list per split, parallel to its
    # data paths: the i-th reference file is row-aligned with the i-th plant
    # file and must carry exactly its timestamps. Listing one plant file
    # several times against several reference files ("draws") multiplies the
    # training windows. The reference columns reach the policy and the loss,
    # never the surrogate.
    train_reference_path: Union[str, list[str], None] = None
    dev_reference_path: Union[str, list[str], None] = None
    test_reference_path: Union[str, list[str], None] = None
    # Divisor per reference column that bounds no Y column (a tariff, a power
    # cap), so the policy sees an O(1) input. Pure scaling, no shift, so a
    # zero stays a zero. A column that bounds a Y column uses that column's
    # normalisation instead, and must not be listed here.
    reference_scale: dict[str, float] = field(default_factory=dict)
    # Future steps of each reference column the policy sees, beyond the
    # current one. Shifting load means acting before the expensive hours,
    # which a policy can only do if it sees them coming. A per-run knob, so it
    # lives here rather than in the hooks' `preview_config`.
    reference_preview: dict[str, int] = field(default_factory=dict)

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

    # ── Objective ────────────────────────────────────────────────
    # A table of loss terms over named columns, and one gain per term — see
    # control.losses for how an entry is read. Two flat dicts, so a gain can be
    # swept with --set without restating the table.
    losses: dict[str, dict] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)

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
    # Snap policy output to a fixed setpoint grid, in the plant's physical
    # units (e.g. 0.5 for half-degree HVAC thermostat steps). Straight-through
    # estimator on backward, so gradients still flow as if the output were
    # continuous. 0 disables quantization — the default, since a nonzero grid
    # is one plant's actuator resolution, not a library-wide default.
    setpoint_step: float = 0.0
    # The residual box, in the plant's physical units. None disables the
    # residual entirely (a plain policy over the actuation box). Set it and
    # the policy commands `reference_control(...) + δ`, which caps how far a
    # surrogate error can move the deployed setpoint: a number Δ bounds
    # -Δ <= δ <= Δ; a pair [below, above] bounds -below <= δ <= above, so
    # [0, 0.5] lets a setpoint move up off the rule and never down. One box
    # for every U column. 0 (or [0, 0]) emits the reference rule exactly — a
    # useful floor to train nothing and score. Requires the
    # control_user_defined.reference_control hook.
    residual_delta: Union[float, list[float], None] = None

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
        # imported by the CLI before anything else.
        from neurotemplate.control.losses import parse_losses
        from neurotemplate.control.policy import (
            POLICY_BOUND_METHODS, POLICY_NONLIN, policy_catalog,
        )

        if not self.sysid_run_id:
            raise ValueError("sysid_run_id must be set (MLflow run id of the "
                             "trained dynamics surrogate)")

        check_one_of(self, "policy", list(policy_catalog))
        check_one_of(self, "policy_nonlin", list(POLICY_NONLIN))
        check_one_of(self, "policy_bound_method", POLICY_BOUND_METHODS)

        # Advisory, not fatal — same threshold and reasoning as
        # ModelingConfig.in_steps: this used to be a hard ValueError ceiling,
        # which made the same value fatal here and merely advisory on the
        # modeling side for no good reason. The real risks (memory, a window
        # longer than the training horizon) don't justify refusing a value
        # the user set on purpose; the right window length is dataset-specific.
        if self.in_steps > IN_STEPS_WARN_THRESHOLD:
            warnings.warn(
                f"in_steps={self.in_steps} is above the usual ceiling of "
                f"{IN_STEPS_WARN_THRESHOLD}. A longer context window costs "
                "more memory per batch, and once it exceeds train_pred_steps "
                "it hands the model, at rollout time, history it never saw "
                "during training. Not rejected outright — the right window "
                "length is dataset-specific.",
                stacklevel=2,
            )

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

        references = {name: getattr(self, name) for name in (
            "train_reference_path", "dev_reference_path", "test_reference_path")}
        unset = [name for name, value in references.items() if value is None]
        if 0 < len(unset) < 3:
            raise ValueError(
                f"{', '.join(references)} must be set together or not at all "
                f"(unset: {unset}). A split with no reference file would have "
                "no value for a reference column the objective names."
            )
        for name, value in references.items():
            if value is not None:
                _check_paths_field(name, value)
        for name, scale in self.reference_scale.items():
            if not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError(
                    f"reference_scale[{name!r}] must be a positive number, "
                    f"got {scale!r}")
        for name, steps in self.reference_preview.items():
            if not isinstance(steps, int) or steps < 0:
                raise ValueError(
                    f"reference_preview[{name!r}] must be a non-negative int, "
                    f"got {steps!r}")
        # Names are resolved against the surrogate's columns at load time
        # (losses.resolve_objective); the table's own shape is checked now.
        parse_losses(self.losses, self.gains)

        for name in [
            "batch_size", "train_pred_steps", "in_steps",
            "log2_policy_hidden", "policy_layers",
            "epochs", "epoch_verbose", "lr",
            "test_window", "test_stride",
        ]:
            check_positive(self, name)

        for name in ["d_preview", "warmup", "patience", "setpoint_step"]:
            check_non_negative(self, name)

        if self.residual_delta is not None:
            box = self.residual_delta
            pair = box if isinstance(box, list) else [box, box]
            if len(pair) != 2 or any(isinstance(d, bool) or not isinstance(
                    d, (int, float)) or d < 0 for d in pair):
                raise ValueError(
                    f"residual_delta must be a number >= 0 or a pair [below, "
                    f"above] of numbers >= 0, got {box!r}. 0 emits the "
                    "reference rule exactly; null disables the residual."
                )
            # The single most likely way to waste a campaign on this feature:
            # every δ snaps back to zero and you get a rule that trained for an
            # hour, with nothing on any report to say so.
            for side in self.residual_box():
                if 0 < side < self.setpoint_step:
                    raise ValueError(
                        f"residual_delta ({box}) is smaller than setpoint_step "
                        f"({self.setpoint_step}) on one side, so every residual "
                        "that way quantizes back to zero. Raise it to at least "
                        "one grid step, make that side 0, or set "
                        "setpoint_step: 0."
                    )

    def residual_box(self) -> Optional[tuple[float, float]]:
        """(below, above) of the residual box in physical units, or None
        when there is no residual."""
        box = self.residual_delta
        if box is None:
            return None
        below, above = box if isinstance(box, list) else (box, box)
        return float(below), float(above)

    def asdict(self) -> dict:
        return asdict(self)

