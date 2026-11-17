"""
DPC control pipeline entry point.

  nt control train --config control_configs/control.yaml \
      --set sysid_run_id=<mlflow run id>

Thin wrapper: this file owns CLI parsing and the closing lines. The run
itself is control.train.train_control, which Python callers use directly.
"""

from __future__ import annotations

import sys

from neurotemplate._typing import beartype

from neurotemplate.common.config import CONTROL_CONFIG_DIR, config_line, load_config_from_cli
from neurotemplate.common.tracking import artifact_file
from neurotemplate.control.config import ControlConfig
from neurotemplate.control.train import train_control


@beartype
def main() -> None:
    cfg, cli = load_config_from_cli(
        sys.argv[1:],
        cls=ControlConfig,
        config_dir=CONTROL_CONFIG_DIR,
        default_config_path=CONTROL_CONFIG_DIR / "control.yaml",
    )
    if cfg.progress != "silent":
        print(config_line(cli))
    run = train_control(cfg, config_path=cli.config)
    if cfg.progress == "silent":
        return
    # Its id is what `nt control gap` and `nt export policy` take.
    print(f"Finished control run: {run.run_id}")
    print(f"Reportcard:\n  {artifact_file(run.run_id, 'control_reportcard.md')}")
    print("MLflow UI:\n  uv run mlflow ui")


if __name__ == "__main__":
    main()
