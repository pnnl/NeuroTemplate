"""
`nt docs config`: every field of both configs, generated from the dataclasses and the
comments above each field, so it cannot drift from what the library reads.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, fields

_FIELD = re.compile(r"^    (\w+): ")
_SECTION = re.compile(r"^    # (?:──|----) (.+?) (?:─|-)+$")
#: Set by the library, not by you.
_INTERNAL = {"steps_per_hour"}


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
