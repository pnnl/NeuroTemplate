"""
`nt docs config`, and the template's default config files: every field of both
configs, generated from the dataclasses and the comments above each field, so neither
can drift from what the library reads.

    python -m neurotemplate.common.config_docs    # rewrite the template's default files
"""

from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, fields
from pathlib import Path

import yaml

from neurotemplate.common.terminal import WIDTH, fill

_FIELD = re.compile(r"^    (\w+): ")
_SECTION = re.compile(r"^    # (?:──|----) (.+?) (?:─|-)+$")
#: Set by the library, not by you.
_INTERNAL = {"steps_per_hour"}
#: Left out of a written config: a fixed value would make every run share one.
_PER_RUN = {"savedir"}


def _comments(cls) -> dict[str, tuple[str, str]]:
    """`{field: (section, comment)}`, the comment being the lines right above the field,
    read from the class's own source. A comment shared by several fields in a row is
    repeated on each only as a pointer to the first."""
    found, section, block = {}, "", []
    for line in inspect.getsource(cls).splitlines():
        heading = _SECTION.match(line)
        match = _FIELD.match(line)
        if heading:
            section, block = heading.group(1).strip(), []
        elif line.startswith("    #"):
            block.append(line.strip().lstrip("#").strip())
        elif match:
            found[match.group(1)] = (section, " ".join(block))
            block = []
        elif not line.strip():
            block = []
    return found


def _default(f) -> str:
    if f.default is not MISSING:
        return repr(f.default)
    if f.default_factory is not MISSING:
        value = f.default_factory()
        return "unique per run" if isinstance(value, str) else repr(value)
    return "required"


def config_reference() -> str:
    """The markdown `nt docs config` prints."""
    from neurotemplate.control.config import ControlConfig, SOLVER_DEFAULTS
    from neurotemplate.modeling.config import ModelingConfig

    lines = ["# Config",
             "",
             "Every field, for `model_configs/*.yaml` and `control_configs/*.yaml`, or `--set "
             "key=value`. Windows are times: `5h`, `20m`, `6s`, `100ms`, `10us`; a bare 0 is fine and "
             "any other bare number is refused. Each run prints what you set apart from the "
             "defaults it used.", ""]
    for title, cls in (("Model", ModelingConfig), ("Control", ControlConfig)):
        lines += [f"## {title}", ""]
        comments = _comments(cls)
        section = None
        for f in fields(cls):
            if f.name in _INTERNAL:
                continue
            where, comment = comments.get(f.name, ("", ""))
            if where != section:
                section = where
                lines += ["", f"### {where.title()}", ""] if where else []
            lines.append(f"- `{f.name}` (default {_default(f)})" + (f": {comment}" if comment else ""))
        lines.append("")
    lines += [f"`solver` defaults: {SOLVER_DEFAULTS}.", ""]
    return "\n".join(lines)


def _value(value) -> str:
    """One value as YAML, on one line."""
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
    return text.removesuffix("...").strip()


def _comment(text: str) -> list[str]:
    """`text` as comment lines, what `null` or empty comes to on a line of its own."""
    return [line for part in re.split(r" (?=(?:null|Empty): )", text) if part
            for line in fill(part, WIDTH, indent="# ", hang="# ")]


def config_yaml(cls, header: str) -> str:
    """Every setting of `cls` at its default, under its comment and section: the
    default config a project starts with. `header` is the file's opening comment."""
    instance, comments = cls(), _comments(cls)
    lines, section = [*_comment(header)], None
    for f in fields(cls):
        if f.name in _INTERNAL | _PER_RUN:
            continue
        where, comment = comments.get(f.name, ("", ""))
        if where != section:
            section = where
            title = f"# ---- {where.title()} "
            lines += ["", title + "-" * (WIDTH - len(title))]
        lines += [*_comment(comment), f"{f.name}: {_value(getattr(instance, f.name))}"]
    return "\n".join(lines) + "\n"


_MODEL_HEADER = (
    "Every setting for `nt model train`, at its default: what a run with no "
    "--config reads. It is the reference; change a setting in custom.yaml. "
    "`nt docs config` explains each field; a window is a time: 5h, 20m, 6s.")
_CONTROL_HEADER = (
    "Every setting for `nt control train`, at its default: what a run with no "
    "--config reads. It is the reference; change a setting in custom.yaml. "
    "`nt docs config` explains each field; a window is a time: 5h, 20m, 6s. "
    "The objective, `losses` and `gains`, has no default: a run needs one.")


def default_config_files() -> dict[str, str]:
    """`{template path: text}` for the two default config files."""
    from neurotemplate.control.config import ControlConfig
    from neurotemplate.modeling.config import ModelingConfig

    return {"model_configs/modeling.yaml": config_yaml(ModelingConfig, _MODEL_HEADER),
            "control_configs/control.yaml": config_yaml(ControlConfig, _CONTROL_HEADER)}


if __name__ == "__main__":
    template = Path(__file__).parents[1] / "template"
    for name, text in default_config_files().items():
        (template / name).write_text(text, encoding="utf-8")
        print(f"  wrote {template / name}")
