"""Unified `nt` command-line entry point.

Dispatches `nt <group> <verb> [args...]` to the matching subcommand `main()`.
The dispatcher only consumes the two leading tokens (group + verb); everything
after is left in `sys.argv` so the subcommand parses its own flags exactly as
before. Groups mirror the package layout:

    nt model train          # neurotemplate.modeling.cli.train
    nt model optuna         # neurotemplate.modeling.cli.optuna
    nt model load           # neurotemplate.modeling.cli.load
    nt model debug-heads    # neurotemplate.modeling.cli.debug_heads
    nt data heads           # neurotemplate.modeling.cli.data (which head does this Y column want?)
    nt data decay           # neurotemplate.modeling.cli.data (what is the envelope time constant?)
    nt data latch           # neurotemplate.modeling.cli.data (does the equipment latch?)
    nt control train        # neurotemplate.control.cli.train
    nt export model         # neurotemplate.export.cli (trained dynamics -> ONNX package)
    nt export policy        # neurotemplate.export.cli (trained DPC policy -> ONNX package)
    nt guide [step]         # neurotemplate.guide (the guided flow; takes no verb)
"""
from __future__ import annotations

import sys
import warnings
from importlib import import_module
from typing import Optional

# (group, verb) -> "module:function". One line per console command that used
# to live in [project.scripts]; this dict is the single source of truth and
# the help text is derived from it.
COMMANDS: dict[tuple[str, str], str] = {
    ("model", "train"):       "neurotemplate.modeling.cli.train:main",
    ("model", "optuna"):      "neurotemplate.modeling.cli.optuna:main",
    ("model", "load"):        "neurotemplate.modeling.cli.load:main",
    ("model", "debug-heads"): "neurotemplate.modeling.cli.debug_heads:main",
    ("data", "heads"):        "neurotemplate.modeling.cli.data:heads_main",
    ("data", "decay"):        "neurotemplate.modeling.cli.data:decay_main",
    ("data", "latch"):        "neurotemplate.modeling.cli.data:latch_main",
    ("control", "train"):     "neurotemplate.control.cli.train:main",
    ("export", "model"):      "neurotemplate.export.cli:model_main",
    ("export", "policy"):     "neurotemplate.export.cli:policy_main",
}


def _help_text() -> str:
    lines = ["Usage: nt <group> <verb> [args...]", "",
             "Start with `nt guide`: the flow, one step at a time.", "", "Commands:",
             "  nt guide [step]"]
    for group, verb in COMMANDS:
        lines.append(f"  nt {group} {verb}")
    lines.append("")
    lines.append("Pass --help after a command for its own flags, e.g. `nt model train --help`.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Reports print symbols (Δ, ⚠, −) that Windows' default cp1252 cannot
    # encode when stdout is a pipe or a log file; a console already takes UTF-8.
    # Line buffering keeps a redirected log (`nt ... > run.log`) current
    # instead of dumping training progress in blocks minutes apart.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    # Dependencies' import-time noise (an invalid escape in a docstring, a
    # deprecated alias) is not the user's to act on, and every line of it
    # lands in an agent's context. The library's own warnings still show.
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    for package in ("tensorly", "neuromancer", "neuralop", "torch_harmonics"):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=package)
        warnings.filterwarnings("ignore", category=FutureWarning, module=package)

    if not argv or argv[0] in ("-h", "--help"):
        print(_help_text())
        return

    if argv[0] == "guide":
        from neurotemplate.guide import main as guide_main
        sys.argv = ["nt guide", *argv[1:]]
        guide_main()
        return

    if len(argv) < 2:
        print(_help_text(), file=sys.stderr)
        raise SystemExit(f"nt: '{argv[0]}' needs a verb (e.g. `nt {argv[0]} train`).")

    group, verb = argv[0], argv[1]
    target = COMMANDS.get((group, verb))
    if target is None:
        print(_help_text(), file=sys.stderr)
        raise SystemExit(f"nt: unknown command '{group} {verb}'.")

    module_name, func_name = target.split(":")
    func = getattr(import_module(module_name), func_name)

    # Hand the subcommand a clean argv: drop the two dispatch tokens, keep the
    # rest. The subcommands all read sys.argv themselves, so rewrite it here.
    sys.argv = [f"nt {group} {verb}", *argv[2:]]
    func()


if __name__ == "__main__":
    main()
