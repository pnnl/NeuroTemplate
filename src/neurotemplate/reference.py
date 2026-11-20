"""`nt docs [name]`: the shipped references.

The text is docs/ beside this file, shipped with the package, so what an
installed library prints is the one for that version. `nt docs` lists
everything; `nt docs <name>` prints a reference, `nt docs hooks` one line per
hook, `nt docs <hook>` that hook's section of docs/HOOKS.md, and `nt docs
config` every config field, generated from the dataclasses.
"""
from __future__ import annotations

import sys
from pathlib import Path

from neurotemplate.common.terminal import reflow

DOCS = Path(__file__).with_name("docs")


def sections(text: str) -> tuple[str, dict[str, str]]:
    """`(overview, {name: section})`, split on the `## <name>` headings."""
    overview, *rest = text.split("\n## ")
    found = {}
    for chunk in rest:
        name, _, body = chunk.partition("\n")
        found[name.strip()] = f"## {name.strip()}\n{body.rstrip()}\n"
    return overview.rstrip() + "\n", found


def doc_paths() -> dict[str, Path]:
    """`{name: path}` for every reference, named by its file in lower case."""
    return {path.stem.lower(): path for path in sorted(DOCS.glob("*.md"))}


def hook_sections() -> tuple[str, dict[str, str]]:
    """`(the one-line-per-hook overview, {hook: its section})`, from docs/HOOKS.md."""
    return sections((DOCS / "HOOKS.md").read_text(encoding="utf-8"))


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
        print(reflow(overview))
    elif name in hooks:
        print(reflow(hooks[name]))
    elif name == "config":
        from neurotemplate.common.config_docs import config_reference
        print(reflow(config_reference()))
    elif name in docs:
        print(reflow(docs[name].read_text(encoding="utf-8")))
    else:
        raise SystemExit(f"nt docs: no reference {name!r}; `nt docs` lists them.")
