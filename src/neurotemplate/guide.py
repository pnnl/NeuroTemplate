"""`nt guide [step]` and `nt docs [name]`: the shipped guide and references.

The text is GUIDE.md and docs/ beside this file, shipped with the package, so what an
installed library prints is the one for that version. `nt guide` prints the
overview (everything before the first step); `nt guide <step>` prints that step.
`nt docs` lists everything; `nt docs <name>` prints a reference, `nt docs hooks` one line
per hook, `nt docs <hook>` that hook's section of docs/HOOKS.md, and `nt docs config` every
config field, generated from the dataclasses.
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


def hook_sections() -> tuple[str, dict[str, str]]:
    """`(the one-line-per-hook overview, {hook: its section})`, from docs/HOOKS.md."""
    return guide_sections((DOCS / "HOOKS.md").read_text(encoding="utf-8"))


def docs_main() -> None:
    docs = doc_paths()
    overview, hooks = hook_sections()
    args = sys.argv[1:]
    if not args:
        for name, path in docs.items():
            if name != "hooks":
                title = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
                print(f"  nt docs {name:<22} {title}")
        print(f"  nt docs {'config':<22} every config field")
        print(f"  nt docs {'hooks':<22} one line per hook")
        for hook in hooks:
            print(f"  nt docs {hook:<22} that hook: signature, explanation, example")
        return
    name = args[0]
    if name == "hooks":
        print(overview)
    elif name in hooks:
        print(hooks[name])
    elif name == "config":
        from neurotemplate.common.config_docs import config_reference
        print(config_reference())
    elif name in docs:
        print(docs[name].read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"nt docs: no reference {name!r}; `nt docs` lists them.")
