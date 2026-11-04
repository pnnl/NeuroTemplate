from __future__ import annotations

import warnings
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

# History-window warning threshold for a window's depth in steps (the
# surrogate's `history_steps`, the policy's deepest `past + 1`). Past this
# many steps, a longer context window costs more memory per batch, and once
# it exceeds train_pred_steps it hands the model, at rollout time, context
# it never saw during training. Advisory only — the right window length is
# dataset-specific, so validate() warns rather than rejects. Twelve is a
# reasonable ceiling for buildings on 5-minute data, not a limit the library
# can defend for every system.
HISTORY_WARN_THRESHOLD = 12


def check_lag_table(name: str, table: dict) -> None:
    """A per-column table of steps, every value an int >= 0.

    Which columns it must name is checked where the model is built
    (`dynamics.column_lags`), so commands that build no model, `nt data`'s,
    run without one.
    """
    for column, steps in table.items():
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(
                f"{name}[{column!r}] must be an int >= 0, got {steps!r}")


def warn_deep_window(name: str, steps: int) -> None:
    if steps > HISTORY_WARN_THRESHOLD:
        warnings.warn(
            f"{name}={steps} is above the usual ceiling of "
            f"{HISTORY_WARN_THRESHOLD}. A longer context window costs "
            "more memory per batch, and once it exceeds train_pred_steps "
            "it hands the model, at rollout time, history it never saw "
            "during training. Not rejected outright — the right window "
            "length is dataset-specific.",
            stacklevel=3,
        )


@dataclass
class ModelingConfig:
    """Top-level config object for system-identification training, search, and loading."""

    # ---- DATA ----------------------------------------------------------------
    # Train/dev/test CSV paths (all three required). Each may be a single
    # path string or a list of path strings. When a list is given, files
    # are loaded independently and their windows are pooled into one
    # DataLoader; no rollout window ever spans two source files. Inside
    # the pipeline the value is always normalized to `list[str]` via
    # `paths_as_list(...)` before consumption.
    train_data_path: Union[str, list[str]] = ""
    dev_data_path: Union[str, list[str]] = ""
    test_data_path: Union[str, list[str]] = ""
    # Optional JSON mapping used by data_utils.get_grad_masks (None disables grads).
    grad_path: Optional[str] = None
    # DataLoader batch size used in data_utils.get_loader.
    batch_size: int = 256
    # Normalization strategy: min_max | abs_max | z_score | None.
    norm_type: Optional[str] = "z_score"
    # Exact column names to drop before Y/U/D mapping.
    exclude_columns: list[str] = field(default_factory=list)
    # Optional path or dotted module name for the modeling hooks file.
    # If unset, the library looks for `nt_model.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ---- MODEL ---------------------------------------------------------------
    # A built-in family (mlp, nssm, lssm, gru, rc_network), or a name in the hooks
    # file's custom_models() (see modeling.contract).
    model: str = "nssm"
    # Steps before the current one the model reads, per Y, U and D column:
    # `{column: lag}`. Every column is listed. The window every block shares
    # is `history_steps = max(lag) + 1` deep; each learned map then reads
    # only its columns' own lags (see dynamics.column_lags).
    history: dict[str, int] = field(default_factory=dict)
    # Rows of history every model is handed, beyond what `history` asks: what a
    # model that keeps state starts from. An `rc_network`'s filter runs over this
    # window to place its hidden nodes. 0 asks for nothing beyond `history`.
    context_steps: int = 0
    log2_hidden_size: int = 6
    num_layers: int = 3
    residual: bool = True
    # Give each block its own sub-model, with its own parameters and hidden
    # state and no term reading another block's state on the same step.
    # Which channels form a block comes from the optional `block_mapping`
    # hook; without one there is one block holding everything, and this
    # flag only costs a wrapper. `rc_network` is per-block by construction
    # and rejects this flag.
    per_block: bool = False
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
    epochs: int = 300
    # What training prints: silent | summary | live | log (`common.progress`).
    progress: str = "summary"
    # How often `live` and `log` update, in epochs.
    update_every: int = 10
    warmup: int = 5
    patience: int = 25
    train_pred_steps: int = 48
    # Rows between the starts of training and dev windows. None means one window
    # every `history_steps` rows; with a long `context_steps` that is few windows,
    # and with `batch_size` at or above their number, one update per epoch.
    train_stride: Optional[int] = None
    curriculum_steps: int = 0
    curriculum_ratio: float = 4.0
    lr: float = 0.001

    # ---- DEV -----------------------------------------------------------------
    dev_pred_steps: int = 48

    # ---- TEST ----------------------------------------------------------------
    test_window: int = 287
    test_stride: int = 288

    # ---- LOSS ----------------------------------------------------------------
    ref_gain: float = 1.0

    # ---- STAGED CHANNELS -----------------------------------------------------
    # A binary / ordinal channel is usually a rare-event channel (a furnace
    # that fires 2% of the time), and neither MSE nor a midpoint snap values
    # the rare level. Both settings come from the data, not the optimiser —
    # see modeling/calibration.py — and neither touches a regression head.
    #
    # Weight each levelled head's loss by the inverse square-root training
    # frequency of its levels, so a rare level counts more but not
    # proportionally more (a 1% level gets 10x, not 100x). Off, every sample
    # weighs the same (the pre-existing behaviour).
    balance_levels: bool = True
    # After training, move each ordinal head's decision thresholds so the
    # predicted level shares on dev rollouts match the true ones. Off, decode
    # snaps at the level midpoints — a p=0.5 rule the data does not justify.
    calibrate_thresholds: bool = True

    # ---- RUNTIME -------------------------------------------------------------
    seed: int = -1
    device: Optional[str] = None
    # Torch CPU threads for this run (`torch.set_num_threads`); None keeps
    # torch's own choice, one per core. These models are small enough that
    # one thread was fastest for every family and batch size measured.
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
    save_config: str = "last_step"

    # ---- OPTUNA --------------------------------------------------------------
    n_trials: int = 100
    study_name: str = "study"
    trial_epochs: int = 100
    optuna_dir: str = "optuna_dbs"
    trial_seeds: int = 3

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
            "epochs", "update_every", "train_pred_steps",
            "dev_pred_steps", "test_window", "test_stride",
            "curriculum_ratio", "ensemble_seeds", "trial_seeds",
        ]:
            check_positive(self, name)
        if self.threads is not None:
            check_positive(self, "threads")

        for name in [
            # lr=0 is valid: it mints a checkpoint without any parameter updates.
            "lr",
            "warmup", "patience", "curriculum_steps",
            "ref_gain",
            "n_trials", "grad_gain", "trial_epochs",
            "grad_n_samples", "synth_n_scenarios",
            "synth_hours", "synth_n_plot",
        ]:
            check_non_negative(self, name)

        check_positive(self, "physical_lr")

        # Imported here rather than at module scope: dynamics imports this
        # config module, so a top-level import would be circular.
        # `model` is checked where it is built (`get_dynamics`): a name that
        # is no built-in family may be one of the hooks file's own.
        from neurotemplate.modeling.dynamics import PER_BLOCK_FAMILIES

        check_one_of(self, "progress", list(PROGRESS_LEVELS))
        check_one_of(self, "norm_type", ["min_max", "abs_max", "z_score", None])
        check_one_of(self, "save_config", ["last_step", "all", "None", None])

        check_lag_table("history", self.history)
        # Only what a learned map reads: `context_steps` is what a model starts
        # from, not inputs to a network.
        warn_deep_window("history", max(self.history.values(), default=-1) + 1)
        if self.context_steps < 0:
            raise ValueError(f"context_steps must be >= 0; got {self.context_steps}.")
        if self.train_stride is not None:
            check_positive(self, "train_stride")

        if self.per_block and self.model not in PER_BLOCK_FAMILIES:
            extra = (" rc_network is already per-block and needs no flag."
                     if self.model == "rc_network" else "")
            raise ValueError(
                f"per_block is not supported for model '{self.model}'; "
                f"available: {list(PER_BLOCK_FAMILIES)}.{extra}"
            )

        # The first history_steps of a test window are the seeded history,
        # not predictions; a window no longer than that scores nothing and
        # the run would fail only after training, in the test metrics.
        if self.test_window <= self.history_steps:
            raise ValueError(
                f"test_window ({self.test_window}) must be > history_steps "
                f"({self.history_steps}); the first history_steps of each test "
                "window are seeded history, so nothing would be scored."
            )

        if (self.grad_gain > 0) and (self.grad_path is None):
            raise ValueError("grad_gain set but not grad_path")

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

    @property
    def history_steps(self) -> int:
        """Rows of every column the model is handed: the deepest lag plus the
        current step, or `context_steps` if that is more. Data windows, the
        seeded history and the model export all run at this depth."""
        return max(max(self.history.values(), default=-1) + 1, self.context_steps)

    def asdict(self) -> dict:
        return asdict(self)
