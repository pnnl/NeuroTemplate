"""Config for the DPC control pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (
    PROGRESS_LEVELS,
    _check_paths_field,
    check_non_negative,
    check_one_of,
    check_positive,
    default_savedir,
)
from neurotemplate.modeling.config import warn_deep_window

#: The keys a `policy_window` entry may set.
WINDOW_KEYS = ("past", "future")


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

    # Optional path or dotted module name for the control hooks file.
    # If unset, the library looks for `nt_control.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ── Control horizon / policy window ──────────────────────────
    # Closed-loop training rollout length (pure prediction; window = hmax + train_pred_steps).
    train_pred_steps: int = 48
    # What the policy reads, per Y, U, D and reference column:
    # `{column: {past: n, future: m}}`, steps before and after the current
    # one. Every column is listed. A Y or U column takes no `future` (not
    # yet measured, or the policy's own output); a D or reference column
    # requires one. A U column's window ends at the previous step, the last
    # command applied, so `past: 0` means the policy does not read it. See
    # control.policy.build_policy.
    policy_window: dict[str, dict[str, int]] = field(default_factory=dict)

    # ── Objective ────────────────────────────────────────────────
    # A table of loss terms over named columns, and one gain per term — see
    # control.losses for how an entry is read. Two flat dicts, so a gain can be
    # swept with --set without restating the table.
    losses: dict[str, dict] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)

    # ── Policy network ───────────────────────────────────────────
    # Network family (see control.policy.policy_catalog). "linear" ignores
    # policy_layers / log2_policy_hidden / policy_nonlin and produces a single
    # saturated affine feedback law: the default, because it has little it
    # can learn from the surrogate that the plant does not share. "mpc" trains
    # nothing: it plans through the surrogate at every step (control.mpc).
    policy: str = "linear"
    policy_nonlin: str = "gelu"
    # How the network output is squashed into [umin, umax]: "sigmoid_scale"
    # (smooth, never exactly reaches the limits) or "relu_clamp" (hard clip).
    policy_bound_method: str = "sigmoid_scale"
    log2_policy_hidden: int = 3
    policy_layers: int = 1
    # `policy: mpc`: steps planned ahead (every D and reference column's
    # `future` must reach them), and each step's solve (`control.solver`):
    # Adam at `mpc_lr` (a share of each command's range per step), stopped
    # once `mpc_patience` steps pass without the objective improving by more
    # than `mpc_rtol` of it, capped at `mpc_max_iters`.
    mpc_horizon: int = 72
    mpc_lr: float = 0.02
    mpc_rtol: float = 1e-3
    mpc_patience: int = 10
    mpc_max_iters: int = 200
    # The residual, per U column: {column: [below, above]} in that column's
    # physical units. A column named here commands
    # clamp(reference_control(...)[column] + δ, umin, umax) with
    # -below <= δ <= above, which caps how far a surrogate error can move what
    # is deployed: [0, 0.5] lets a heating setpoint move up off the rule and
    # never down, [0, 0] is the rule exactly. A U column not named is a free
    # policy output over its actuation bounds. Empty: no residual, a plain
    # policy. Requires the nt_control.reference_control hook, which
    # returns exactly the columns named here.
    residual_delta: dict[str, list[float]] = field(default_factory=dict)

    # A per-step range for a U column, `{column: [low, high]}`: each side a
    # number in the column's units, the name of a column (a reference the
    # policy reads, or another command), or `[column, offset]`,
    # that column plus a signed offset in the U column's units. The policy
    # commands inside its actuation box as usual, and that box is mapped onto
    # [low, high] at each step. A side naming another command orders the two:
    # `{Cool: [[Heat, 1.0], 30.0]}` holds cooling at least 1 above heating at
    # every step. A column cannot be both here and in residual_delta.
    command_bounds: dict[str, list] = field(default_factory=dict)

    # ── Training ─────────────────────────────────────────────────
    epochs: int = 300
    # What training prints: silent | summary | live | log (`common.progress`).
    progress: str = "summary"
    # How often `live` and `log` update, in epochs.
    update_every: int = 10
    warmup: int = 5
    patience: int = 25
    lr: float = 0.001

    # ── Evaluation (long closed-loop rollout) ────────────────────
    # Total window length for eval (hmax warmup IS counted in this; actual
    # rollout on plots is test_window - hmax). Matches sys-id semantics.
    test_window: int = 287
    test_stride: int = 288
    # The best open-loop plan on the surrogate over the same windows, the
    # bar the policy is read against (`control.plan`): solved as MPC's is,
    # from `plan_starts` starts (the controller's commands and random ones),
    # each window keeping its best. 0 starts skips it.
    plan_starts: int = 4
    plan_lr: float = 0.02
    plan_rtol: float = 1e-3
    plan_patience: int = 10
    plan_max_iters: int = 200

    # ── Runtime ──────────────────────────────────────────────────
    seed: int = -1
    device: Optional[str] = None
    # Torch CPU threads for this run (`torch.set_num_threads`); None keeps
    # torch's own choice, one per core. These models are small enough that
    # one thread was fastest for every family and batch size measured.
    threads: Optional[int] = 1

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

        check_one_of(self, "policy", [*policy_catalog, "mpc"])
        if self.policy == "mpc" and self.residual_delta:
            raise ValueError("policy: mpc plans every command itself, so it has no "
                             "residual to box: leave residual_delta empty.")
        check_one_of(self, "progress", list(PROGRESS_LEVELS))
        check_one_of(self, "policy_nonlin", list(POLICY_NONLIN))
        check_one_of(self, "policy_bound_method", POLICY_BOUND_METHODS)

        # Which columns, and which may look ahead, is checked at policy
        # build, against the surrogate's columns and the objective's.
        check_policy_window(self.policy_window)
        deepest = max(e.get("future", 0) for e in self.policy_window.values())
        if deepest > self.train_pred_steps:
            raise ValueError(
                f"policy_window previews {deepest} steps but train_pred_steps is "
                f"{self.train_pred_steps}: nothing past a training rollout is scored, so "
                "the policy would read a preview it is never rewarded for using. Make "
                "train_pred_steps at least the deepest `future`.")
        warn_deep_window("the policy's history_steps", self.history_steps)

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
        # Names are resolved against the surrogate's columns at load time
        # (losses.resolve_objective); the table's own shape is checked now.
        parse_losses(self.losses, self.gains)

        for name in [
            "batch_size", "train_pred_steps",
            "log2_policy_hidden", "policy_layers",
            "epochs", "update_every", "lr",
            "test_window", "test_stride",
            "plan_lr", "plan_patience", "plan_max_iters",
            "mpc_horizon", "mpc_lr", "mpc_patience", "mpc_max_iters",
        ]:
            check_positive(self, name)
        if self.threads is not None:
            check_positive(self, "threads")

        for name in ["warmup", "patience", "plan_starts", "plan_rtol", "mpc_rtol"]:
            check_non_negative(self, name)

        # Column names are resolved against the surrogate's U at policy build.
        for column, box in self.residual_delta.items():
            if not (isinstance(box, list) and len(box) == 2 and all(
                    isinstance(side, (int, float)) and not isinstance(side, bool)
                    and side >= 0 for side in box)):
                raise ValueError(
                    f"residual_delta[{column!r}] must be a pair [below, above] "
                    f"of numbers >= 0 in the column's units, got {box!r}. "
                    "[0, 0] emits the reference rule exactly."
                )

        def side_ok(side) -> bool:
            if isinstance(side, list):
                return (len(side) == 2 and isinstance(side[0], str)
                        and isinstance(side[1], (int, float)) and not isinstance(side[1], bool))
            return isinstance(side, (int, float, str)) and not isinstance(side, bool)

        for column, pair in self.command_bounds.items():
            if not (isinstance(pair, list) and len(pair) == 2 and all(map(side_ok, pair))):
                raise ValueError(
                    f"command_bounds[{column!r}] must be a pair [low, high], each a "
                    "number in the column's units, a column name, or "
                    f"[column, offset] in the column's units, got {pair!r}.")
            if column in self.residual_delta:
                raise ValueError(
                    f"{column!r} is in both residual_delta and command_bounds; a "
                    "column is either a residual around the rule or bounded per step.")

    @property
    def history_steps(self) -> int:
        """The policy's window depth: its deepest `past`, plus the current
        step. The closed loop seeds at least this many steps, and training
        windows start this far apart."""
        return max((entry["past"] for entry in self.policy_window.values()),
                   default=-1) + 1

    def asdict(self) -> dict:
        return asdict(self)


def check_policy_window(table: dict) -> None:
    """Each entry is `{past: int >= 0[, future: int >= 0]}`, and there is at
    least one."""
    if not table:
        raise ValueError(
            "policy_window is empty. List every Y, U, D and reference column "
            "with its {past, future}; there is no default.")
    for column, entry in table.items():
        if not isinstance(entry, dict) or "past" not in entry or set(entry) - set(WINDOW_KEYS):
            raise ValueError(
                f"policy_window[{column!r}] must be a mapping with `past` and "
                f"optionally `future`, got {entry!r}")
        for key, steps in entry.items():
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                raise ValueError(
                    f"policy_window[{column!r}][{key!r}] must be an int >= 0, "
                    f"got {steps!r}")

