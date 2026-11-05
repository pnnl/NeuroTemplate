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
from neurotemplate.common.tracking import artifact_file
from neurotemplate.modeling.config import ModelingConfig
from neurotemplate.modeling.pipeline import TestResults
from neurotemplate.metrics.model_card import MAX_ROWS, WORST_ROWS, ranked_skills
from neurotemplate.modeling.train import train_model


def print_finished(run_id: str, test_results: TestResults) -> None:
    """The closing lines of a training run: its id, test RMSE beside
    persistence per channel (the lowest-skill ones when there are many), and
    where to look next."""
    print(f"Finished run: {run_id}")
    rows = ranked_skills(test_results.metrics, test_results.persistence)
    shown = rows if len(rows) <= MAX_ROWS else rows[:WORST_ROWS]
    if len(shown) < len(rows):
        print(f"  the {len(shown)} lowest-skill of {len(rows)} channels:")
    width = max(len(m.name) for m, _, _ in shown)
    for m, reference, skill in shown:
        print(f"  test RMSE  {m.name:<{width}s}  {m.rmse:.4g} (persistence "
              f"{reference:.4g}, skill {skill:.2f})")
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
        print_finished(result.run_id, result.test_results)


if __name__ == "__main__":
    main()
