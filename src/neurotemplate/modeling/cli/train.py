"""
Entry point for training with physics-informed losses.

Thin wrapper: this file owns CLI parsing and the closing lines. The run
itself is modeling.train.train_model, which Python callers use directly.

Usage:
    nt model train --config model_configs/modeling.yaml
    nt model train --config model_configs/modeling.yaml \
        --set grad_gain=1.0
"""

import sys

from neurotemplate._typing import beartype

from neurotemplate.common.config import config_line, load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.data_file import check_roles
from neurotemplate.modeling.config import ModelingConfig

# torch, neuromancer and mlflow are imported in `main`, after the column roles are
# checked: on a fresh install they take a minute to load, and an unassigned or
# misspelled column should not wait on them.


def print_finished(run_id: str) -> None:
    """The closing lines of a training run: where to look next. Whether the
    model is any good is the reportcard's to say, not a few lines here."""
    from neurotemplate.common.tracking import artifact_file
    # A path or a command on a line of its own, so it is never broken mid-way.
    print(f"Reportcard:\n  {artifact_file(run_id, 'reportcard.md')}")
    print("MLflow UI:\n  uv run mlflow ui")
    print(f"Load model:\n  uv run nt model load --data <test csv> \\\n    --run_id {run_id}")


@beartype
def main() -> None:
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ModelingConfig,
        config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )
    if cfg.progress != "silent":
        print(config_line(cli))
    check_roles(cfg.data_file)
    from neurotemplate.modeling.train import train_model
    result = train_model(cfg, config_path=cli.config)
    if cfg.progress != "silent":
        print_finished(result.run_id)


if __name__ == "__main__":
    main()
