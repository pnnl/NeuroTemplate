"""
DPC control pipeline entry point.

  nt control train --config control_configs/control.yaml \
      --set sysid_run_id=<mlflow run id>

Thin wrapper: this file owns CLI parsing and the closing summary. The run
itself is control.train.train_control, which Python callers use directly.
"""

from __future__ import annotations

import sys

from neurotemplate._typing import beartype

from neurotemplate.common.config import CONTROL_CONFIG_DIR, load_config_from_cli
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
    run = train_control(cfg, config_path=cli.config)

    print(f"Comfort violation rate: "
          f"{run.results.comfort_violation_rate * 100:.2f}%")
    print(f"Mean |ΔU| (all U columns, physical units): {run.results.mean_abs_du:.4g}")
    print(f"Finished DPC run: {run.run_id}")


if __name__ == "__main__":
    main()
