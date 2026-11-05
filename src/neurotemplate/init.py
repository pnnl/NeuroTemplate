"""`nt init [dir]`: start an application from the template shipped with the package.

The template is the `template/` directory beside this file, so the hooks and
configs an installed library writes are the ones for that version. It reads
nothing, and never overwrites: if any file it would write already exists, it writes
nothing.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TEMPLATE = Path(__file__).with_name("template")


def template_files() -> list[Path]:
    """Every file in the template, relative to it (bytecode caches left out)."""
    return sorted(
        path.relative_to(TEMPLATE) for path in TEMPLATE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def write_template(target: Path) -> list[Path]:
    """Copy the template into `target` and make an empty `data/`; return the files written."""
    files = template_files()
    existing = [str(f) for f in files if (target / f).exists()]
    if existing:
        raise FileExistsError(
            f"{target} already has {', '.join(existing)}; nt init writes nothing "
            "over an existing file. Move them aside, or init another directory."
        )
    for f in files:
        (target / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(TEMPLATE / f, target / f)
    (target / "data").mkdir(exist_ok=True)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(prog="nt init", description=__doc__.splitlines()[0])
    parser.add_argument("dir", nargs="?", default=".",
                        help="Application directory (default: the current one).")
    target = Path(parser.parse_args().dir)
    try:
        files = write_template(target)
    except FileExistsError as exc:
        raise SystemExit(f"nt init: {exc}") from exc
    for f in files:
        print(f"  wrote {target / f}")
    print(f"\nPut your train.csv, dev.csv and test.csv in {target / 'data'}/, then run "
          "`nt model train`: it lists the record's columns for data.yaml. `nt guide setup` "
          "has the rest.")
