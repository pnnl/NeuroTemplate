"""
Markdown reportcard plumbing shared by the modeling and control sides.

Both subsystems render a per-run markdown card built from `## ` sections and
drop it in the run's artifact directory. The section formatting and the
write-to-disk-plus-MLflow step are identical; only the content and the output
filename differ.
"""

from __future__ import annotations

from pathlib import Path


#: Where a run keeps each instrument's full output, one JSON file each. The
#: reportcard is the summary; these are for tools and for looking things up.
RESULTS_DIR = "results"


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def _names(names: list[str], limit: int = 6, quote: bool = True) -> str:
    """`a`, `b`, `c` — or, past `limit`, the first `limit` and how many more.
    The full list is always in the run's results; the card names enough to
    recognise the pattern."""
    shown = [f"`{n}`" if quote else n for n in names[:limit]]
    more = len(names) - len(shown)
    return ", ".join(shown) + (f", … (+{more})" if more else "")


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
