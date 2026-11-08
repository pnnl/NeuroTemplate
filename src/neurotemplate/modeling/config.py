from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (
    DEFAULT_PAST_STEPS,
    PROGRESS_LEVELS,
    Time,
    TimedWindows,
    check_non_negative,
    check_one_of,
    check_positive,
    check_training,
    default_savedir,
    time_microseconds,
    time_steps,
)

# History-window warning threshold for a window's depth in steps (the
# surrogate's `history_steps`, the policy's deepest `past + 1`). Past this
# many steps, a longer context window costs more memory per batch, and once
# it exceeds `horizon` it hands the model, at rollout time, context
# it never saw during training. Advisory only — the right window length is
# dataset-specific, so validate() warns rather than rejects. Twelve is a
# reasonable ceiling for buildings on 5-minute data, not a limit the library
# can defend for every system.
HISTORY_WARN_THRESHOLD = 12


def check_lag_table(name: str, table: dict) -> None:
    """A per-column table of times.

    Which columns it must name is checked where the model is built
    (`dynamics.column_lags`), so commands that build no model, `nt data`'s,
    run without one.
    """
    for column, value in table.items():
        time_microseconds(value, f"{name}[{column!r}]")


def warn_deep_window(name: str, steps: int) -> None:
    if steps > HISTORY_WARN_THRESHOLD:
        warnings.warn(
            f"{name}={steps} is above the usual ceiling of "
            f"{HISTORY_WARN_THRESHOLD}. A longer context window costs "
            "more memory per batch, and once it exceeds `horizon` "
            "it hands the model, at rollout time, history it never saw "
            "during training. Not rejected outright — the right window "
            "length is dataset-specific.",
            stacklevel=3,
        )


@dataclass
class ModelingConfig(TimedWindows):
    """Top-level config object for system-identification training and loading."""

    # ---- DATA ----------------------------------------------------------------
    # The record and its column roles (`common.data_file`). With several files in a
    # split, windows are pooled and no rollout window spans two files.
    data_file: str = "data.yaml"
    # Optional JSON mapping used by data_utils.get_grad_masks (None disables grads).
    grad_path: Optional[str] = None
    # DataLoader batch size used in data_utils.get_loader.
    batch_size: int = 256
    # Normalization strategy: min_max | abs_max | z_score | None.
    norm_type: Optional[str] = "z_score"
    # Optional path or dotted module name for the modeling hooks file.
    # If unset, the library looks for `nt_model.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ---- MODEL ---------------------------------------------------------------
    # A built-in family (mlp, nssm, lssm, gru, rc_network), or a name in the hooks
    # file's custom_models() (see modeling.contract).
    model: str = "nssm"
    # How far before the current step the model reads: one time for every column, or
    # `{column: time}`, a column not named reading the current row only. Each learned
    # map reads only its columns' own lags (see dynamics.column_lags). Unset, 3 steps.
    past: Union[Time, dict[str, Time], None] = None
    # History every model is handed, beyond what `past` asks: what a model that keeps
    # state starts from. An `rc_network`'s filter runs over this window to place its
    # hidden nodes. Unset, 24h when a block is an rc_network, and otherwise nothing.
    context: Optional[Time] = None
    log2_hidden_size: int = 6
    num_layers: int = 3
    # A model's output is the change from the last row (`increment`) or the next
    # `value`. A block may say otherwise for itself.
    predicts: str = "increment"
    # One block's family, from here: `{block name: family}`. It wins over the
    # block's own `model` in block_mapping, which wins over `model` above; every
    # run prints which family each block took and from where.
    block_models: dict[str, str] = field(default_factory=dict)
    # Timesteps per hour, resolved from the data's sampling frequency by the
    # CLI (like `device` and `seed`) rather than set in YAML. Only
    # `rc_network` reads it, to discretise its network in real hours.
    steps_per_hour: Optional[float] = None

    # ---- PHYSICAL PARAMETERS ---------------------------------------------------
    # A grey box's physical parameters (an rc_network's C, R, gains and
    # thermostat) are declared per block, as boxes in the block_mapping hook's
    # `params`. Here: their learning rate. Adam moves a raw parameter by
    # about `lr` per update, and a box position has the whole box to cover
    # where a network weight moves by thousandths; at `lr` they stall.
    physical_lr: float = 0.05

    # ---- TRAIN ---------------------------------------------------------------
    # Gradient steps: a fixed budget, with the best weights on dev kept. Provisional
    # until a plant bench sets it.
    iters: int = 3000
    # Steps between scorings of the dev set.
    eval_every: int = 100
    # The peak rate: it ramps up over the first 5% of the steps, then follows a cosine
    # down to `lr_final` (`common.fit`). `lr: 0` trains nothing and keeps the model as built.
    lr: float = 1e-3
    # Where the cosine ends.
    lr_final: float = 1e-5
    # What training prints: silent | summary | live | log (`common.progress`).
    progress: str = "summary"
    # How far a training or dev rollout predicts, past its history. Unset, 48 steps.
    horizon: Optional[Time] = None
    # Time between the starts of training windows, and of dev windows. Unset, a quarter
    # of a horizon, and a whole one.
    train_stride: Optional[Time] = None
    dev_stride: Optional[Time] = None

    # ---- TEST ----------------------------------------------------------------
    # How far a test rollout predicts, past its history, and how far apart they start.
    # Unset, 288 steps or what the shortest test file holds, and not overlapping.
    test_horizon: Optional[Time] = None
    test_stride: Optional[Time] = None

    # ---- LOSS ----------------------------------------------------------------
    ref_gain: float = 1.0

    # ---- RUNTIME -------------------------------------------------------------
    seed: int = -1
    device: Optional[str] = None
    # Torch CPU threads for this run; None means one per core. These models are
    # small enough that one thread was fastest for every family and batch size
    # measured.
    threads: Optional[int] = 1
    # How many independently-seeded models train_and_evaluate trains (each at
    # cfg.seed + i) and evaluates, reporting the min/median/max spread across
    # them instead of a single seed's draw. 1 reproduces today's single-model
    # behaviour exactly. Only the primary (offset-0) model's weights are
    # logged, plotted, and rendered on the reportcard — the ensemble adds
    # error bars around that one artifact, it does not change what you get.
    ensemble_seeds: int = 1

    # ---- LOGGING -------------------------------------------------------------
    savedir: str = field(default_factory=default_savedir)
    location: str = "sqlite:///mlflow.db"
    exp: str = "Model Training"
    run: str = ""
    plots_dir: str = "plots"
    plot: bool = True

    # ---- PHYSICS -------------------------------------------------------------
    # grad_* weights a training penalty; synth_* only sizes the post-training
    # physics assertions, which always run when synth_cfg returns a spec.
    grad_gain: float = 0.0
    grad_n_samples: int = 128
    synth_n_scenarios: int = 32
    # Scenario horizon in HOURS, converted to steps against the data's own
    # sampling frequency. A step count means nothing without knowing the
    # timestep, and for a reachability bound the horizon is load-bearing.
    synth_hours: float = 4.0
    synth_n_plot: int = 4

    def validate(self) -> None:
        for name in [
            "batch_size", "log2_hidden_size", "num_layers",
            "ensemble_seeds",
        ]:
            check_positive(self, name)
        check_training(self)
        if self.threads is not None:
            check_positive(self, "threads")

        for name in [
            # lr=0 is valid: it mints a checkpoint without any parameter updates.
            "lr",
            "ref_gain", "grad_gain",
            "grad_n_samples", "synth_n_scenarios",
            "synth_hours", "synth_n_plot",
        ]:
            check_non_negative(self, name)

        check_positive(self, "physical_lr")

        # `model` is checked where it is built (`get_dynamics`): a name that
        # is no built-in family may be one of the hooks file's own.
        check_one_of(self, "progress", list(PROGRESS_LEVELS))
        check_one_of(self, "norm_type", ["min_max", "abs_max", "z_score", None])

        check_one_of(self, "predicts", ["increment", "value"])
        if isinstance(self.past, dict):
            check_lag_table("past", self.past)
        elif self.past is not None:
            time_microseconds(self.past, "past")
        if self.context is not None:
            time_microseconds(self.context, "context")
        self._check_common_times()

        if (self.grad_gain > 0) and (self.grad_path is None):
            raise ValueError("grad_gain set but not grad_path")

    def fill_windows(self, shortest_test_rows: int) -> None:
        """What each unset window comes to, written back so the run logs it: once the
        data has set the timestep and `context` has been decided (`dynamics`)."""
        if self.past is None:
            self.past = self._time(DEFAULT_PAST_STEPS)
        self.context = 0 if self.context is None else self.context
        self.fill_common_windows(shortest_test_rows - self.history_steps)

    def check_windows(self) -> None:
        """Every window a whole number of steps: run once the data has set the timestep."""
        self._check_common_windows()
        # Only what a learned map reads: `context` is what a model starts from, not
        # inputs to a network.
        warn_deep_window("past", self.deepest_lag + 1)

    def lags_for(self, columns: list[str]) -> dict[str, int]:
        """`past` for `columns`, in steps."""
        table = self.past if isinstance(self.past, dict) else dict.fromkeys(columns, self.past)
        return {c: time_steps(table.get(c, 0), self.steps_per_hour, f"past[{c!r}]")
                for c in columns}

    @property
    def deepest_lag(self) -> int:
        """The deepest `past`, in steps."""
        values = self.past.values() if isinstance(self.past, dict) else [self.past]
        return max((time_steps(v, self.steps_per_hour, "past") for v in values), default=0)

    @property
    def context_steps(self) -> int:
        return self._steps("context") if self.context is not None else 0

    @property
    def history_steps(self) -> int:
        """Rows of every column the model is handed: the deepest lag plus the
        current step, or `context` if that is more. Data windows, the
        seeded history and the model export all run at this depth."""
        return max(self.deepest_lag + 1, self.context_steps)

    def asdict(self) -> dict:
        return asdict(self)
