"""
Export trained models as a standalone ONNX package.

  nt export model --run_id <sys-id run> --data data/test.csv \
      --out delivery/site --package site_model
  nt export policy --run_id <control run> --out delivery/site --package site_model

`model` exports a dynamics model: it rolls both the trained system and the
exported package out over every row of `--data` (a held-out split) and
refuses to write unless they agree. `policy` exports a trained DPC policy: it
runs the trained closed loop over the control run's own test split and
refuses to write unless the exported policy issues the same commands on
every window it saw. Both may target the same `--out`; see
`neurotemplate.export.package` for the layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neurotemplate.common.data_utils import get_data, get_data_splits, normalize_df
from neurotemplate.control.load import load_trained_policy
from neurotemplate.export.dynamics import export_model
from neurotemplate.export.policy import export_policy
from neurotemplate.export.trace import EXPORTERS
from neurotemplate.modeling.cli.load import load_trained_dynamics


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run_id", required=True, help="MLflow run to export.")
    parser.add_argument("--out", required=True, help="Package directory to write.")
    parser.add_argument("--package", required=True,
                        help="Python package name for the runtime (an identifier).")
    parser.add_argument("--description", default=None,
                        help="One line describing it, for README and describe().")
    parser.add_argument("--exporter", choices=EXPORTERS, default="dynamo",
                        help="torch.export (default) or legacy TorchScript tracing, "
                             "which writes smaller files.")
    parser.add_argument("--version", default="0.1.0",
                        help="Version written to the package's pyproject.toml.")
    parser.add_argument("--run_dir", default=None,
                        help="Optional override for the run's artifacts directory.")


def get_model_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained dynamics model.")
    _common_args(parser)
    parser.add_argument("--data", required=True,
                        help="Held-out CSV: verified over every row, and shipped "
                             "as the reference fixture.")
    parser.add_argument("--fixture_steps", type=int, default=None,
                        help="Ship only the first N steps as the fixture "
                             "(default: all of --data). Verification still "
                             "covers every row.")
    return parser.parse_args(argv)


def get_policy_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained DPC policy.")
    _common_args(parser)
    parser.add_argument("--fixture_windows", type=int, default=256,
                        help="How many of the verified closed-loop windows to "
                             "ship as the fixture (evenly spaced).")
    return parser.parse_args(argv)


def model_main() -> None:
    args = get_model_args(sys.argv[1:])
    dyn_sys, cfg, nm_mapping, norm_stats = load_trained_dynamics(
        args.run_id, run_dir=args.run_dir)
    frame, timestep_seconds = get_data(args.data)

    result = export_model(
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
    _report_kept(result.kept, result.out_dir)


def policy_main() -> None:
    args = get_policy_args(sys.argv[1:])
    policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = (
        load_trained_policy(args.run_id, run_dir=args.run_dir))
    _, _, test_raws, _ = get_data_splits(cfg)

    result = export_policy(
        policy_node=policy_node, cl_system=cl_system, cfg=cfg, sysid_cfg=sysid_cfg,
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        test_dfs=[normalize_df(raw, norm_stats) for raw in test_raws],
        run_id=args.run_id, out_dir=Path(args.out), package=args.package,
        description=args.description, exporter=args.exporter,
        fixture_windows=args.fixture_windows, version=args.version,
    )

    worst = max(result.error, key=result.error.get)
    print(f"exported policy run {args.run_id} ({cfg.policy}) with the "
          f"{args.exporter} exporter")
    print(f"  verified {result.windows_verified} closed-loop windows against the "
          f"trained policy: worst command {worst} {result.error[worst]:.2e}")
    print(f"  shipped {result.windows_shipped} windows as the reference fixture")
    _report_kept(result.kept, result.out_dir)


def _report_kept(kept: list[str], out_dir: Path) -> None:
    if kept:
        print(f"  kept the project's own {', '.join(kept)}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    model_main()
