"""
Entry point for training with physics-informed losses.

Thin wrapper: this file owns CLI parsing and the closing summary. The run
itself is modeling.train.train_model, which Python callers use directly.

Usage:
    nt model train --config model_configs/modeling.yaml
    nt model train --config model_configs/modeling.yaml \
        --set grad_gain=1.0
"""

import sys

from neurotemplate._typing import beartype

from neurotemplate.common.config import load_config_from_cli, MODEL_CONFIG_DIR
from neurotemplate.common.metrics import ChannelMetrics
from neurotemplate.common.tracking import artifact_file
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.train import train_model


def print_finished(run_id: str, metrics: list[ChannelMetrics]) -> None:
    """The closing lines of a training run: its id, each channel's test RMSE
    and how to look at it.

    Per channel and in physical units, because one pooled normalised number
    rounds two different surrogates to the same three digits.
    """
    print(f"Finished run: {run_id}")
    width = max(len(m.name) for m in metrics)
    for m in metrics:
        print(f"  test RMSE  {m.name:<{width}s}  {m.rmse:.4g}")
    print(f"Reportcard: {artifact_file(run_id, 'reportcard.md')}")
    print("MLflow UI:  uv run mlflow ui")
    print(f"Load model: uv run nt model load --run_id {run_id} --data <test csv>")


@beartype
def main() -> None:
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ModelingConfig,
        config_dir=MODEL_CONFIG_DIR,
        default_config_path=MODEL_CONFIG_DIR / "modeling.yaml",
    )
    result = train_model(cfg, config_path=cli.config)
    if cfg.progress != "silent":
        print_finished(result.run_id, result.test_results.metrics)


if __name__ == "__main__":
    main()
