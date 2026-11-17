"""Unified `nt` command-line entry point.

Dispatches `nt <group> <verb> [args...]` to the matching subcommand `main()`.
The dispatcher only consumes the two leading tokens (group + verb); everything
after is left in `sys.argv` so the subcommand parses its own flags. Groups
mirror the package layout:

    nt model train          # neurotemplate.modeling.cli.train
    nt model load           # neurotemplate.modeling.cli.load
    nt data ladder          # neurotemplate.modeling.cli.data (which RC network fits the zone?)
    nt data thermostat      # neurotemplate.modeling.cli.data (what heat does a setpoint deliver?)
    nt data plot            # neurotemplate.modeling.cli.data_plot (look at a record, one channel at a time)
    nt control train        # neurotemplate.control.cli.train
    nt control gap          # neurotemplate.control.cli.gap (the best plan, beside a controller)
    nt export model         # neurotemplate.export.cli (trained dynamics -> ONNX package)
    nt export policy        # neurotemplate.export.cli (trained DPC policy -> ONNX package)
    nt docs [name]          # neurotemplate.reference (the references; takes no verb)
    nt init [dir]           # neurotemplate.init (a new application from the template; takes no verb)
"""
from __future__ import annotations

import os
import sys
import warnings
from importlib import import_module
from typing import Optional

from neurotemplate.common.terminal import install

# (group, verb) -> "module:function", the single source of truth for the
# commands; the help text is derived from it.
COMMANDS: dict[tuple[str, str], str] = {
    ("model", "train"):       "neurotemplate.modeling.cli.train:main",
    ("model", "load"):        "neurotemplate.modeling.cli.load:main",
    ("data", "ladder"):       "neurotemplate.modeling.cli.data:ladder_main",
    ("data", "thermostat"):   "neurotemplate.modeling.cli.data:thermostat_main",
    ("data", "plot"):         "neurotemplate.modeling.cli.data_plot:main",
    ("control", "train"):     "neurotemplate.control.cli.train:main",
    ("control", "gap"):       "neurotemplate.control.cli.gap:main",
    ("export", "model"):      "neurotemplate.export.cli:model_main",
    ("export", "policy"):     "neurotemplate.export.cli:policy_main",
}


def _help_text() -> str:
    lines = ["Usage: nt <group> <verb> [args...]", "",
             "Start:   `nt init`, then its guides/1-first-model.md.",
             "", "Commands:", "  nt init [dir]",
             "  nt docs [name]"]
    for group, verb in COMMANDS:
        lines.append(f"  nt {group} {verb}")
    lines.append("")
    lines.append("Pass --help after a command for its own flags, e.g. `nt model train --help`.")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Dependencies' import-time noise (a deprecated alias) is not the user's
    # to act on, and every line of it lands in an agent's context. The
    # library's own warnings, and the user's hooks', still show.
    for package in ("tensorly", "neuromancer", "neuralop", "torch_harmonics"):
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=package)
        warnings.filterwarnings("ignore", category=FutureWarning, module=package)
    # Their docstrings' LaTeX (`\Sigma`) trips Python 3.12's invalid-escape
    # SyntaxWarning the first time a fresh venv compiles them. A compile-time
    # warning's "module" is the source path, so match the package directory.
    warnings.filterwarnings(
        "ignore", category=SyntaxWarning,
        module=r".*[/\\](tensorly|neuromancer|neuralop|torch_harmonics)[/\\]")
    # Neuromancer hands its LR scheduler the loss tensor, and torch warns once per run.
    warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad",
                            module="torch.optim.lr_scheduler")

    if not argv or argv[0] in ("-h", "--help"):
        print(_help_text())
        return

    if argv[0] == "docs":
        from neurotemplate.reference import docs_main
        sys.argv = ["nt docs", *argv[1:]]
        docs_main()
        return

    if argv[0] == "init":
        from neurotemplate.init import main as init_main
        sys.argv = ["nt init", *argv[1:]]
        init_main()
        return

    if argv[0] == "example":
        raise SystemExit("nt: `nt example` is gone. Run `nt init`, then "
                         "`uv run python guides/make_data.py` for a practice record "
                         "that every guide runs on.")

    if argv[0] == "guide":
        raise SystemExit("nt: `nt guide` is gone: the guides are files now, in your "
                         "project's guides/ folder. Start with guides/1-first-model.md.")

    if argv[0] not in {group for group, _ in COMMANDS}:
        print(_help_text(), file=sys.stderr)
        raise SystemExit(f"nt: unknown command '{argv[0]}'.")

    if len(argv) < 2:
        print(_help_text(), file=sys.stderr)
        verbs = ", ".join(verb for group, verb in COMMANDS if group == argv[0])
        raise SystemExit(f"nt: '{argv[0]}' needs a verb: {verbs}.")

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


def run() -> None:
    """The `nt` command: `main` with the process's output set up for a terminal."""
    # Reports print symbols (Δ, ⚠, −) that Windows' default cp1252 cannot
    # encode when stdout is a pipe or a log file; a console already takes UTF-8.
    # Line buffering keeps a redirected log (`nt ... > run.log`) current
    # instead of dumping training progress in blocks minutes apart.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    # Every line at most 80 columns, through to the error printed at exit.
    install()
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        # The library's refusals are written to be read: a wrong column name, a
        # missing file, a config that does not add up. The traceback only
        # buries them; NT_TRACEBACK=1 brings it back.
        if os.environ.get("NT_TRACEBACK"):
            raise
        sys.stdout.flush()
        raise SystemExit(f"\nnt: {e}\n(NT_TRACEBACK=1 shows where it came from.)") from None


if __name__ == "__main__":
    run()
