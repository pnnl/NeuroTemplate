"""
Markdown reportcard plumbing shared by the modeling and control sides.

Both subsystems render a per-run markdown card built from `## ` sections and
drop it in the run's artifact directory. The section formatting and the
write-to-disk-plus-MLflow step are identical; only the content and the output
filename differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def _fmt_float(x: Optional[float], precision: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{precision}f}"


def write_reportcard(
    markdown: str,
    out_dir: Path,
    filename: str,
    logger=None,
) -> Path:
    """
    Write the reportcard to <out_dir>/<filename> and to MLflow.

    The on-disk copy is convenient for local inspection but may be removed
    by the logger's clean_up() (which rmtrees savedir at end of a run);
    the MLflow copy via log_text is the durable one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(markdown, encoding="utf-8")
    if logger is not None:
        logger.log_text(markdown, filename)
    return path
