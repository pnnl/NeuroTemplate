"""`nt init [dir]`: start an application from the template shipped with the package.

The template is the `template/` directory beside this file, so the hooks and
configs an installed library writes are the ones for that version. It reads
nothing, and never overwrites: if any file it would write already exists, it writes
nothing.

Then it loads, once, what the commands load (`neurotemplate.warmup`): on a fresh
install that takes a minute, and here it is setup rather than the first few runs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
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


def warm_up() -> None:
    """Run `neurotemplate.warmup` in its own process, then its imports again in another
    (which opens the bytecode the first wrote), saying how long it took. Their output
    is the libraries' noise, so it is dropped; a failure costs nothing but time in the
    first run, so it is reported and passed over."""
    print("\nLoading the libraries once, so the first runs don't have to. On a fresh "
          "install this takes a minute or two.", flush=True)
    start = time.perf_counter()
    done = None
    for extra in ([], ["--imports-only"]):
        done = subprocess.run([sys.executable, "-m", "neurotemplate.warmup", *extra],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if done.returncode != 0:
            break
    took = time.perf_counter() - start
    if done.returncode == 0:
        print(f"Ready ({took:.0f} s).")
    else:
        print(f"That stopped early ({took:.0f} s); the first run will load the rest.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="nt init", description=__doc__.splitlines()[0])
    parser.add_argument("dir", nargs="?", default=".",
                        help="Application directory (default: the current one).")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip loading the libraries once; the first runs do it instead.")
    args = parser.parse_args()
    target = Path(args.dir)
    try:
        files = write_template(target)
    except FileExistsError as exc:
        raise SystemExit(f"nt init: {exc}") from exc
    for f in files:
        print(f"  wrote {target / f}")
    if not args.no_warmup:
        warm_up()
    print(f"\nStart with {target / 'guides' / '1-first-model.md'}.")
