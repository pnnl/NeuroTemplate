"""Config for the DPC control pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from neurotemplate.common.config import (
    DEFAULT_HORIZON_STEPS,
    DEFAULT_PAST_STEPS,
    PROGRESS_LEVELS,
    Time,
    TimedWindows,
    check_one_of,
    check_positive,
    default_savedir,
    time_microseconds,
    time_steps,
)
from neurotemplate.modeling.config import warn_deep_window

#: The keys a `policy_window` entry may set.
WINDOW_KEYS = ("past", "future", "stride")

#: One solve, for `policy: mpc` and `nt control gap` alike (`control.solver`): Adam at
#: `lr`, a share of each command's range per step, stopped once `patience` steps pass
#: without the objective improving by more than `rtol` of it, capped at `max_iters`.
SOLVER_DEFAULTS = {"lr": 0.02, "rtol": 1e-3, "patience": 10, "max_iters": 200}


@dataclass
class ControlConfig(TimedWindows):
    """Knobs for loading a frozen surrogate and training a DPC policy."""

    # ── Sys-id model to control against ──────────────────────────
    # MLflow run id of the trained dynamics surrogate. Unset, the latest finished one.
    sysid_run_id: str = ""
    # Optional override for the run's artifacts directory.
    run_dir: Optional[str] = None
    # Optional explicit weights path (else best_model_state_dict.pth).
    weights: Optional[str] = None

    # ── Data ─────────────────────────────────────────────────────
    # The record (`common.data_file`). The columns the objective, a window or a bound
    # names that are not the surrogate's — a comfort band, a tariff — are columns no role
    # claims in that record: they reach the policy and the loss, never the surrogate.
    data_file: str = "data.yaml"
    batch_size: int = 256
    # The same record against another band or tariff. Each file replaces the columns
    # it shares with the record it matches by timestamp, and adds a pass over the
    # training windows. Dev and test take no draws: to score a held-out tariff, point
    # `test` at a file holding it. Relative paths are read from the working directory.
    draws: list[str] = field(default_factory=list)
    # Divisor per reference column that bounds no Y column (a tariff, a power
    # cap), so the policy sees an O(1) input. Pure scaling, no shift, so a
    # zero stays a zero. A column that bounds a Y column uses that column's
    # normalisation instead, and must not be listed here. Unset, its largest
    # magnitude in the training record.
    reference_scale: dict[str, float] = field(default_factory=dict)

    # Optional path or dotted module name for the control hooks file.
    # If unset, the library looks for `nt_control.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ── Control horizon / policy window ──────────────────────────
    # How far a training rollout predicts, past its history; for `policy: mpc`, how far
    # each step plans ahead (every D and reference column's `future` must reach it).
    # Unset, 48 steps.
    horizon: Optional[Time] = None
    # Time between the starts of dev windows; unset, a horizon. Training windows start at
    # any row of the records, drawn afresh at every step (`common.sampler`).
    dev_stride: Optional[Time] = None
    # What the policy reads, any column of the record, listed only if it is read:
    # `{column: {past: n, future: m}}`, steps before and after the current one. A
    # Y or U column takes no `future` (not yet measured, or the policy's own output);
    # any other column requires one. A U column's window ends at the previous step, the
    # last command applied, so `past: 0` means the policy does not read it. A forecast's
    # `stride` thins its future to every so many steps: `{past: 0, future: 6h, stride:
    # 45m}` reads now and 8 rows ahead. Unset, every row. See control.policy.build_policy.
    # A trained policy's package takes only what is listed.
    # Unset: the regulated outputs and the D columns 3 steps back, the last command, and
    # each D and reference column a horizon ahead.
    policy_window: dict[str, dict[str, Time]] = field(default_factory=dict)

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
    # Each solve of `policy: mpc` and of `nt control gap`; see SOLVER_DEFAULTS. Name
    # only what you change.
    solver: dict = field(default_factory=dict)

    # What a U column may command, `{column: [low, high]}`: each side a number in
    # the column's units, the name of a column (a reference the policy reads, or
    # another command), or `[column, offset]`, that column plus a signed offset in
    # the U column's units. A column not listed takes its recorded range, the region
    # the surrogate was identified on; a side that names a column moves at each step.
    # A side naming another command orders the two: `{Cool: [[Heat, 1.0], 30.0]}`
    # holds cooling at least 1 above heating at every step.
    control_bounds: dict[str, list] = field(default_factory=dict)

    # What training and dev rollouts start from: `{Y column: [low, high]}` draws that
    # column's last measured value uniform in the box, whatever the window's weather;
    # `{Y column: records}`, over its range in the training records. Columns not listed
    # start as recorded. Unset, every `rc_network` zone over its recorded range; `{}`,
    # every column as recorded. Test always starts as recorded (`control.windows`).
    start_box: Optional[dict] = None

    # ── Training ─────────────────────────────────────────────────
    # Gradient steps: a fixed budget, the dev set scored 20 times and the best kept.
    iters: int = 3000
    # The peak rate: it ramps up over the first 5% of the steps, then follows a cosine
    # down to a hundredth of it (`common.fit`).
    lr: float = 1e-3
    # What training prints: silent | summary | live | log (`common.progress`).
    progress: str = "summary"

    # ── Evaluation (long closed-loop rollout) ────────────────────
    # How far a test rollout runs, past its history, and how far apart they start.
    # Unset, 288 steps or what the shortest test file holds, and not overlapping.
    test_horizon: Optional[Time] = None
    test_stride: Optional[Time] = None

    # ── Runtime ──────────────────────────────────────────────────
    # Timesteps per hour: the surrogate's, set when it loads.
    steps_per_hour: Optional[float] = None
    seed: int = -1
    device: Optional[str] = None
    # Torch CPU threads for this run; None means one per core. These models are
    # small enough that one thread was fastest for every family and batch size
    # measured.
    threads: Optional[int] = 1

    # ── Logging ──────────────────────────────────────────────────
    savedir: str = field(default_factory=default_savedir)
    location: str = "sqlite:///mlflow.db"
    exp: str = "DPC Training"
    run: str = ""
    plots_dir: str = "plots"
    plot: bool = True

    def __post_init__(self) -> None:
        unknown = sorted(set(self.solver) - set(SOLVER_DEFAULTS))
        if unknown:
            raise ValueError(f"solver has unknown key(s) {unknown}; it takes "
                             f"{list(SOLVER_DEFAULTS)}.")
        solver = {**SOLVER_DEFAULTS, **self.solver}
        self.solver = {"lr": float(solver["lr"]), "rtol": float(solver["rtol"]),
                       "patience": int(solver["patience"]), "max_iters": int(solver["max_iters"])}

    def validate(self) -> None:
        # Deferred import: policy.py pulls in Neuromancer, and config is
        # imported by the CLI before anything else.
        from neurotemplate.control.losses import parse_losses
        from neurotemplate.control.policy import (
            POLICY_BOUND_METHODS, POLICY_NONLIN, policy_catalog,
        )

        check_one_of(self, "policy", [*policy_catalog, "mpc"])
        check_one_of(self, "progress", list(PROGRESS_LEVELS))
        check_one_of(self, "policy_nonlin", list(POLICY_NONLIN))
        check_one_of(self, "policy_bound_method", POLICY_BOUND_METHODS)

        # Which columns, and which may look ahead, is checked at policy
        # build, against the surrogate's columns and the objective's.
        check_policy_window(self.policy_window)
        self._check_common_times()

        for column, side in (self.start_box or {}).items():
            box = (isinstance(side, list) and len(side) == 2
                   and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in side)
                   and side[0] < side[1])
            if side != "records" and not box:
                raise ValueError(f"start_box[{column!r}] is `[low, high]` with low below high, "
                                 f"or `records`; got {side!r}.")
        for name, scale in self.reference_scale.items():
            if not isinstance(scale, (int, float)) or scale <= 0:
                raise ValueError(
                    f"reference_scale[{name!r}] must be a positive number, "
                    f"got {scale!r}")
        # Names are resolved against the surrogate's columns at load time
        # (losses.resolve_objective); the table's own shape is checked now.
        parse_losses(self.losses, self.gains)

        for name in [
            "batch_size",
            "log2_policy_hidden", "policy_layers",
            "lr",
        ]:
            check_positive(self, name)
        check_positive(self, "iters")
        if self.threads is not None:
            check_positive(self, "threads")

        for name in ("lr", "patience", "max_iters"):
            if self.solver[name] <= 0:
                raise ValueError(f"solver[{name!r}] must be positive, got {self.solver[name]}")
        if self.solver["rtol"] < 0:
            raise ValueError(f"solver['rtol'] must be >= 0, got {self.solver['rtol']}")

        # Column names are resolved against the surrogate's U at policy build.
        def side_ok(side) -> bool:
            if isinstance(side, list):
                return (len(side) == 2 and isinstance(side[0], str)
                        and isinstance(side[1], (int, float)) and not isinstance(side[1], bool))
            return isinstance(side, (int, float, str)) and not isinstance(side, bool)

        for column, pair in self.control_bounds.items():
            if not (isinstance(pair, list) and len(pair) == 2 and all(map(side_ok, pair))):
                raise ValueError(
                    f"control_bounds[{column!r}] must be a pair [low, high], each a "
                    "number in the column's units, a column name, or "
                    f"[column, offset] in the column's units, got {pair!r}.")

    def fill_window_table(self, regulated: list[str], nm_mapping: dict[str, list[str]],
                          r_columns: list[str]) -> None:
        """`policy_window`, when unset: see the field's comment. Written back so the run
        logs it. It needs `horizon`, so that is filled here if it was unset."""
        if self.horizon is None:
            self.horizon = self._time(DEFAULT_HORIZON_STEPS)
        if self.policy_window:
            return
        past = self._time(DEFAULT_PAST_STEPS)
        self.policy_window = {
            **{c: {"past": past} for c in regulated},
            **{c: {"past": self._time(1)} for c in nm_mapping["U"]},
            **{c: {"past": past, "future": self.horizon} for c in nm_mapping["D"]},
            **{c: {"past": 0, "future": self.horizon} for c in r_columns},
        }

    def check_windows(self) -> None:
        """Every window a whole number of steps, and nothing previewed past a rollout: run
        once the surrogate has set the timestep and the defaults have been filled."""
        self._check_common_windows()
        deepest = max((e.get("future", 0) for e in self.window_steps.values()), default=0)
        if deepest > self.horizon_steps:
            raise ValueError(
                f"policy_window previews {deepest} steps but `horizon` ({self.horizon}) is "
                f"{self.horizon_steps}: nothing past a training rollout is scored, so "
                "the policy would read a preview it is never rewarded for using. Make "
                "`horizon` at least the deepest `future`.")
        strides = {column: entry["stride"] for column, entry in self.window_steps.items()
                   if "stride" in entry}
        if any(stride < 1 for stride in strides.values()):
            raise ValueError(f"policy_window strides must be at least a step: {strides}.")
        if strides and self.policy == "mpc":
            raise ValueError(f"policy: mpc plans over every row of its forecasts, so a "
                             f"`stride` would be ignored; drop it from {sorted(strides)}.")
        warn_deep_window("the policy's history_steps", self.history_steps)

    @property
    def window_steps(self) -> dict[str, dict[str, int]]:
        """`policy_window` in steps."""
        return {column: {key: time_steps(value, self.steps_per_hour,
                                         f"policy_window[{column!r}][{key!r}]")
                         for key, value in entry.items()}
                for column, entry in self.policy_window.items()}

    @property
    def history_steps(self) -> int:
        """The policy's window depth: its deepest `past`, plus the current
        step. The closed loop seeds at least this many steps."""
        return max((entry["past"] for entry in self.window_steps.values()), default=-1) + 1

    def asdict(self) -> dict:
        return asdict(self)


def check_policy_window(table: dict) -> None:
    """Each entry is `{past: time[, future: time[, stride: time]]}`."""
    for column, entry in table.items():
        if not isinstance(entry, dict) or "past" not in entry or set(entry) - set(WINDOW_KEYS):
            raise ValueError(
                f"policy_window[{column!r}] must be a mapping with `past` and "
                f"optionally `future` and its `stride`, got {entry!r}")
        if "stride" in entry and "future" not in entry:
            raise ValueError(f"policy_window[{column!r}] has a `stride` but no `future`: a "
                             "stride thins a forecast's future rows.")
        for key, value in entry.items():
            time_microseconds(value, f"policy_window[{column!r}][{key!r}]")

