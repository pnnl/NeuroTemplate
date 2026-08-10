from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from neurotemplate.common.config import (
    _check_paths_field,
    check_max,
    check_non_negative,
    check_one_of,
    check_positive,
    default_savedir,
)

# History-window ceiling, enforced on both ModelingConfig and ControlConfig
# (the two places `in_steps` appears). Keeps the lookback window bounded
# across the library rather than per-project.
MAX_IN_STEPS = 12


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

    # ---- RUNTIME -------------------------------------------------------------
    seed: int = -1
    device: Optional[str] = None
    verbose: bool = True

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
    synth_n_steps: int = 24
    synth_n_plot: int = 4

    def validate(self) -> None:
        for name in [
            "batch_size", "in_steps", "log2_hidden_size", "num_layers",
            "epochs", "epoch_verbose", "train_pred_steps",
            "dev_pred_steps", "test_window", "test_stride",
            "curriculum_ratio",
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
            "synth_n_steps", "synth_n_plot",
        ]:
            check_non_negative(self, name)

        check_one_of(self, "model", ["mlp", "nssm", "lssm", "gru", "dummy"])
        check_one_of(self, "save_config", ["last_step", "all", "None", None])
        check_max(self, "in_steps", MAX_IN_STEPS)

        if (self.grad_gain > 0) and (self.grad_path is None):
            raise ValueError("grad_gain set but not grad_path")

        for name in ("train_data_path", "dev_data_path", "test_data_path"):
            _check_paths_field(name, getattr(self, name))

    def asdict(self) -> dict:
        return asdict(self)
