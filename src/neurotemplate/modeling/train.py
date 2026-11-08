"""
One tracked sys-id training run, start to finish — `nt model train` without
the command line:

    from neurotemplate.common.config import load_config
    from neurotemplate.modeling.config import ModelingConfig
    from neurotemplate.modeling.train import train_model

    cfg = load_config("model_configs/modeling.yaml", ModelingConfig,
                      {"iters": 4})
    run = train_model(cfg)
    run.run_id, run.test_results.metrics

Data prep is pipeline.prepare_data; training, eval, plotting and the
reportcard are pipeline.train_and_evaluate. This file owns the MLflow run
around them.
"""

from copy import deepcopy
from typing import Optional

from neurotemplate._typing import beartype

from neurotemplate.common.config import fields_set, settings_report
from neurotemplate.metrics.run_summary import write_run_summary
from neurotemplate.common.tracking import tracked_run
from neurotemplate.hooks import MODELING_SNAPSHOT, load_modeling_hooks, snapshot_hooks
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.dynamics import family_lines
from neurotemplate.modeling.experiment import resolve_device, set_seeds, set_threads
from neurotemplate.modeling.pipeline import (
    TrainedModel, prepare_data, run_summary_metrics, train_and_evaluate,
)

#: What each run names even when you left it alone. Everything else sits in
#: `nt docs config`, and appears on "You set" only when you set it.
COMMON_SETTINGS = ("model", "past", "context", "horizon", "test_horizon", "train_stride",
                   "dev_stride", "test_stride", "iters", "lr", "batch_size",
                   "log2_hidden_size", "num_layers", "seed", "steps_per_hour")

#: Loss terms echoed to stdout at each evaluation, and the substrings of the metric
#: keys logged to MLflow.
STDOUT_METRICS = ["train_loss", "train_ref_loss", "train_grad_penalty", "dev_loss"]
LOGGED_METRICS = ["train_loss", "dev_loss", "ref_loss", "grad_penalty", "time",
                  "rmse"]


@beartype
def train_model(cfg: ModelingConfig,
                config_path: Optional[str] = None) -> TrainedModel:
    """Train, evaluate and log one model as its own MLflow run.

    `cfg` comes from `load_config` or is built in code; it is validated
    here and not modified: the run resolves its device, seed and run name
    on a copy.
    `config_path` is the file `cfg` came from, recorded in the run summary;
    None for a config built in code.
    """
    given = fields_set(cfg)
    cfg = deepcopy(cfg)
    cfg.validate()
    cfg.device = resolve_device(cfg.device)
    set_threads(cfg.threads)
    cfg.seed = set_seeds(cfg.seed)
    hooks = load_modeling_hooks(cfg)

    data = prepare_data(cfg, hooks)
    if cfg.progress != "silent":
        print(settings_report(cfg, given, COMMON_SETTINGS))
    if data.block_spec and cfg.progress != "silent":
        print("Blocks:\n" + "\n".join(family_lines(cfg, data.nm_mapping, data.block_spec)))

    cfg.run = cfg.model.upper() + " " + cfg.run

    with tracked_run(cfg, stdout=STDOUT_METRICS, logout=LOGGED_METRICS) as logger:
        snapshot_hooks(hooks, MODELING_SNAPSHOT, logger)
        result = train_and_evaluate(cfg, logger=logger, **vars(data))
        write_run_summary(
            logger, stage="model", config_path=config_path, cfg=cfg,
            metrics=run_summary_metrics(result.test_results),
            mapping=data.nm_mapping,
        )
    return result
