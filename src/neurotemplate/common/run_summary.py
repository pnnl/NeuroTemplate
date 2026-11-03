"""
Machine-readable run summary, written into each run's MLflow artifacts.

Run identity and config claims otherwise travel from a `Finished run:` line
on stdout, through a reader, into a hand-typed deliverable — and every hop
is lossy. `run_summary.json` lets a downstream reader copy a block instead
of retyping one, so an audit compares a value to itself.

`params` are read off the same config object that `RunLogger` hands to
`mlflow.log_params`, so the file and the run's params cannot drift. A field
that stops existing on the config raises `AttributeError` here rather than
quietly dropping out of the summary.

This file reports *what ran*, not how well it ran — verdicts stay in the
reportcards.
"""

from typing import Any, Optional

from neurotemplate._typing import beartype
from neurotemplate.common.reportcard import RESULTS_DIR


SCHEMA_VERSION = 1
RUN_SUMMARY_NAME = f"{RESULTS_DIR}/run_summary.json"

# The fields a submission audit compares against the config it claims to
# have run. Keep in step with the audit, not with the full config.
_PARAM_FIELDS = {
    "model": ("model", "epochs", "lr", "batch_size", "train_pred_steps",
              "history", "test_window", "norm_type", "seed"),
    "control": ("epochs", "lr", "batch_size", "train_pred_steps", "policy_window",
                "losses", "gains", "seed", "sysid_run_id"),
}


@beartype
def build_run_summary(
    *,
    stage: str,
    run_id: str,
    config_path: Optional[str],
    cfg: Any,
    metrics: dict[str, float],
    mapping: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Build the summary envelope for one finished run. `config_path` is
    None for a config built in code rather than loaded from a file."""
    if stage not in _PARAM_FIELDS:
        raise ValueError(f"stage must be one of {sorted(_PARAM_FIELDS)}, "
                         f"got {stage!r}")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "run_id": run_id,
        "config_path": config_path,
        "params": {k: getattr(cfg, k) for k in _PARAM_FIELDS[stage]},
        "metrics": metrics,
    }
    if mapping is not None:
        summary["mapping"] = dict(mapping)
    return summary


@beartype
def write_run_summary(
    logger,
    *,
    stage: str,
    config_path: Optional[str],
    cfg: Any,
    metrics: dict[str, float],
    mapping: Optional[dict[str, list[str]]] = None,
) -> dict:
    """Log the summary as a run artifact. Returns it for the caller to reuse."""
    summary = build_run_summary(
        stage=stage,
        run_id=logger.run.info.run_id,
        config_path=config_path,
        cfg=cfg,
        metrics=metrics,
        mapping=mapping,
    )
    logger.log_dict(summary, name=RUN_SUMMARY_NAME)
    return summary
