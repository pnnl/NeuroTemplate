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

# History-window warning threshold for ModelingConfig.in_steps. Past this
# many steps, a longer context window costs more memory per batch, and once
# it exceeds train_pred_steps it hands the model, at rollout time, context
# it never saw during training. Advisory only — the right window length is
# dataset-specific, so ModelingConfig.validate() warns rather than rejects
# (see below). Twelve is a reasonable default for buildings on 5-minute
# data, not a limit the library can defend for every system.
IN_STEPS_WARN_THRESHOLD = 12


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
    # If unset, the library looks for `modeling_user_defined.py` in CWD.
    user_hooks_path: Optional[str] = None

    # ---- MODEL ---------------------------------------------------------------
    model: str = "nssm"
    in_steps: int = 12
    log2_hidden_size: int = 8
    num_layers: int = 3
    residual: bool = True
    # Give each block its own sub-model, with its own parameters and hidden
    # state and no term reading another block's state on the same step.
    # Which channels form a block comes from the optional `block_mapping`
    # hook; without one there is one block holding everything, and this
    # flag only costs a wrapper. `2r2c` is per-block by construction and
    # rejects this flag.
    per_block: bool = False
    # Timesteps per hour, resolved from the data's sampling frequency by the
    # CLI (like `device` and `seed`) rather than set in YAML. Only `2r2c`
    # reads it, to place its time constants in real hours.
    steps_per_hour: Optional[float] = None

    # ---- 2R2C ----------------------------------------------------------------
    # The RC network's three component time constants, in hours: zone<->ambient,
    # zone<->mass from the zone's side, and mass<->zone from the mass's side.
    # These are the storage *prior*, and the reportcard prints the two modes
    # they produce (0.6 h and 13.4 h at these defaults on 5-minute data).
    #
    # They are a prior rather than a fitted parameter because a training
    # rollout cannot identify them: the mass temperature is unobserved, every
    # rollout has to guess it from an in_steps window that is far shorter than
    # the mass lags, and a wrong guess is cheaper to absorb by shortening the
    # slow mode than by fixing the forcing. Left trainable, the optimizer does
    # exactly that. Set these from what you know about the building — a
    # measured free-decay time constant is the number tau_slow should match.
    rc_tau_ambient_h: float = 2.0
    rc_tau_zone_mass_h: float = 1.0
    rc_tau_mass_zone_h: float = 4.0
    # Let training move the three above. Only worth it when the rollout is
    # long relative to the slow mode AND the data excites it; otherwise it
    # drags tau_slow down and the reportcard's number stops being checkable.
    rc_train_taus: bool = False

    # ---- TRAIN ---------------------------------------------------------------
    epochs: int = 300
    epoch_verbose: int = 1
    warmup: int = 5
    patience: int = 25
    train_pred_steps: int = 48
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
    vel_gain: float = 0.0
    sgn_gain: float = 0.0
    sgn_steepness: float = 2.0

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
    # After training, move each ordinal head's decision thresholds to the
    # balanced-accuracy optimum on dev rollouts. Off, decode snaps at the
    # level midpoints — a p=0.5 rule the data does not justify.
    calibrate_thresholds: bool = True

    # ---- RUNTIME -------------------------------------------------------------
    seed: int = -1
    device: Optional[str] = None
    verbose: bool = True
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
    # Deprecated: units are now per-column, parsed from each state column's
    # own "[unit]" suffix (see `common.data_utils.parse_column_unit`) and
    # carried on `norm_stats`. This field only still matters as a fallback
    # for a state column whose name carries no unit of its own — the
    # synthetic-scenario y-axis falls back to this label for such a column
    # rather than going unlabeled. Columns that name their own unit ignore it.
    synth_y_unit: str = "°C"

    def validate(self) -> None:
        for name in [
            "batch_size", "in_steps", "log2_hidden_size", "num_layers",
            "epochs", "epoch_verbose", "train_pred_steps",
            "dev_pred_steps", "test_window", "test_stride",
            "curriculum_ratio", "ensemble_seeds",
        ]:
            check_positive(self, name)

        for name in [
            # lr=0 is valid: it mints a fixed-model checkpoint without any
            # parameter updates (used by the `dummy` passthrough surrogate).
            "lr",
            "warmup", "patience", "curriculum_steps",
            "ref_gain", "vel_gain", "sgn_gain", "sgn_steepness",
            "n_trials", "grad_gain", "trial_epochs",
            "trial_seeds", "grad_n_samples", "synth_n_scenarios",
            "synth_hours", "synth_n_plot",
        ]:
            check_non_negative(self, name)

        for name in ["rc_tau_ambient_h", "rc_tau_zone_mass_h", "rc_tau_mass_zone_h"]:
            check_positive(self, name)

        # Imported here rather than at module scope: dynamics imports this
        # config module, so a top-level import would be circular.
        # VALID_MODEL_NAMES is the single whitelist of valid families (the
        # union of model_catalog and BLOCK_BUILDERS); check_one_of's list and
        # PER_BLOCK_FAMILIES both used to be hand-maintained copies of it.
        from neurotemplate.modeling.dynamics import VALID_MODEL_NAMES, PER_BLOCK_FAMILIES

        check_one_of(self, "model", list(VALID_MODEL_NAMES))
        check_one_of(self, "save_config", ["last_step", "all", "None", None])

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

        if self.per_block and self.model not in PER_BLOCK_FAMILIES:
            extra = (" 2r2c is already per-block and needs no flag."
                     if self.model == "2r2c" else "")
            raise ValueError(
                f"per_block is not supported for model '{self.model}'; "
                f"available: {list(PER_BLOCK_FAMILIES)}.{extra}"
            )

        if (self.grad_gain > 0) and (self.grad_path is None):
            raise ValueError("grad_gain set but not grad_path")

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

    def asdict(self) -> dict:
        return asdict(self)
