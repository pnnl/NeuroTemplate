"""
`nt control gap --run_id <control run>`: the best open-loop plan on the surrogate,
beside the controller, on the windows the controller was scored on (`control.plan`).

It costs more than training does, so it runs when you ask. It writes
`results/control_gap.md` in the run's artifacts, and prints it.
"""

from __future__ import annotations

import argparse
import sys

from neurotemplate.control.load import load_trained_policy
from neurotemplate.control.pipeline import evaluate_closed_loop
from neurotemplate.control.plan import plan_on_surrogate
from neurotemplate.control.reportcard import render_gap_report
from neurotemplate.control.scenarios import load_control_splits
from neurotemplate.common.reportcard import RESULTS_DIR
from neurotemplate.modeling.experiment import set_seeds, set_threads
from neurotemplate.modeling.load import resolve_artifacts_dir


def gap_report(run_id: str, run_dir=None, sysid_run_dir=None, starts: int = 4) -> tuple:
    """The plan, and the report on it, for a finished control run:
    `(PlanResult, markdown)`."""
    policy_node, cl_system, cfg, sysid_cfg, nm_mapping, norm_stats = load_trained_policy(
        run_id, run_dir=run_dir, sysid_run_dir=sysid_run_dir)
    set_threads(cfg.threads)
    set_seeds(cfg.seed)
    _, _, test_dfs = load_control_splits(cfg, nm_mapping)
    results, out = evaluate_closed_loop(cfg, sysid_cfg, cl_system, test_dfs, nm_mapping,
                                        norm_stats)
    plan = plan_on_surrogate(cfg, sysid_cfg, cl_system, policy_node, test_dfs, out,
                             nm_mapping, norm_stats, starts=starts)
    return plan, render_gap_report(plan, results, run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="The best plan, beside a trained controller.")
    parser.add_argument("--run_id", required=True, help="The control run.")
    parser.add_argument("--run_dir", default=None,
                        help="Optional override for the run's artifacts directory.")
    parser.add_argument("--sysid_run_dir", default=None,
                        help="Optional override for its surrogate's artifacts directory.")
    parser.add_argument("--starts", type=int, default=4,
                        help="Starts per window: the controller's commands, and random ones.")
    args = parser.parse_args(sys.argv[1:])
    if args.starts < 1:
        raise SystemExit("--starts must be at least 1.")

    _, report = gap_report(args.run_id, args.run_dir, args.sysid_run_dir, args.starts)
    path = resolve_artifacts_dir(args.run_dir, args.run_id) / RESULTS_DIR / "control_gap.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written: {path}")


if __name__ == "__main__":
    main()
