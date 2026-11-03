"""`nt guide [step]` and `nt docs [name]`: the shipped guide and references.

The text is GUIDE.md and docs/ beside this file, shipped with the package, so what an
installed library prints is the one for that version. `nt guide` prints the
overview (everything before the first step); `nt guide <step>` prints that step.
`nt docs` lists the references; `nt docs <name>` prints one.
"""
from __future__ import annotations

import sys
from pathlib import Path

GUIDE = Path(__file__).with_name("GUIDE.md")
DOCS = Path(__file__).with_name("docs")


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


def doc_paths() -> dict[str, Path]:
    """`{name: path}` for every reference, named by its file in lower case."""
    return {path.stem.lower(): path for path in sorted(DOCS.glob("*.md"))}


def docs_main() -> None:
    docs = doc_paths()
    args = sys.argv[1:]
    if not args:
        for name, path in docs.items():
            title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
            print(f"  nt docs {name:<10} {title}")
        return
    if args[0] not in docs:
        raise SystemExit(f"nt docs: no reference {args[0]!r}; they are {', '.join(docs)}.")
    print(docs[args[0]].read_text(encoding="utf-8"))
