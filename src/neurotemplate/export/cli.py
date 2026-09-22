"""
Export a trained dynamics model as a standalone ONNX package.

  nt export model --run_id <mlflow run id> --data data/test.csv \
      --out delivery/my_model --package my_model

Loads the run, exports its two graphs, rolls both the trained system and the
exported package out over every row of `--data` (a held-out split), and
refuses to write unless they agree. See `neurotemplate.export.package` for
the layout written to `--out`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurotemplate.common.data_utils import get_data
from neurotemplate.export.package import export_package
from neurotemplate.export.trace import EXPORTERS
from neurotemplate.modeling.cli.load import load_trained_dynamics


def get_cli_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--run_id", required=True, help="MLflow run to export.")
    parser.add_argument("--data", required=True,
                        help="Held-out CSV: verified over every row, and shipped "
                             "as the reference fixture.")
    parser.add_argument("--out", required=True, help="Package directory to write.")
    parser.add_argument("--package", required=True,
                        help="Python package name for the runtime (an identifier).")
    parser.add_argument("--description", default=None,
                        help="One line describing the model, for README and describe().")
    parser.add_argument("--exporter", choices=EXPORTERS, default="dynamo",
                        help="torch.export (default) or legacy TorchScript tracing, "
                             "which writes smaller files.")
    parser.add_argument("--fixture_steps", type=int, default=None,
                        help="Ship only the first N steps as the fixture "
                             "(default: all of --data). Verification still "
                             "covers every row.")
    parser.add_argument("--version", default="0.1.0",
                        help="Version written to the package's pyproject.toml.")
    parser.add_argument("--run_dir", default=None,
                        help="Optional override for the run's artifacts directory.")
    return parser.parse_args(argv)


def main() -> None:
    args = get_cli_args(sys.argv[1:])
    dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(
        args.run_id, run_dir=args.run_dir)
    frame, timestep_seconds = get_data(args.data)

    result = export_package(
        dyn_sys=dyn_sys, cfg=cfg, nm_mapping=nm_mapping, norm_stats=norm_stats,
        frame=frame, timestep_seconds=timestep_seconds, run_id=args.run_id,
        out_dir=Path(args.out), package=args.package,
        description=args.description, exporter=args.exporter,
        fixture_steps=args.fixture_steps, version=args.version,
    )

    worst = max(result.agreement.y_error, key=result.agreement.y_error.get)
    print(f"exported run {args.run_id} ({cfg.model}) with the {args.exporter} exporter")
    print(f"  verified {result.steps_verified} open-loop steps against the trained "
          f"model: worst output {worst} {result.agreement.y_error[worst]:.2e}, "
          f"hidden state (width {result.state_width}) "
          f"{result.agreement.state_error:.2e}")
    print(f"  shipped {result.steps_shipped} steps as the reference fixture")
    if result.kept:
        print(f"  kept the project's own {', '.join(result.kept)}")
    print(f"wrote {result.out_dir}")


if __name__ == "__main__":
    main()
