"""`nt guide [step]`: the guided flow, printed one step at a time.

The text is GUIDE.md beside this file, shipped with the package, so the guide an
installed library prints is the one for that version. `nt guide` prints the
overview (everything before the first step); `nt guide <step>` prints that step.
"""
from __future__ import annotations

import sys
from pathlib import Path

GUIDE = Path(__file__).with_name("GUIDE.md")


def guide_sections(text: str) -> tuple[str, dict[str, str]]:
    """`(overview, {step: section})`, split on the `## <step>` headings."""
    overview, *rest = text.split("\n## ")
    steps = {}
    for chunk in rest:
        name, _, body = chunk.partition("\n")
        steps[name.strip()] = f"## {name.strip()}\n{body.rstrip()}\n"
    return overview.rstrip() + "\n", steps


def main() -> None:
    overview, steps = guide_sections(GUIDE.read_text(encoding="utf-8"))
    args = sys.argv[1:]
    if not args:
        print(overview)
        return
    if args[0] not in steps:
        raise SystemExit(f"nt guide: no step {args[0]!r}; the steps are {', '.join(steps)}.")
    print(steps[args[0]])
