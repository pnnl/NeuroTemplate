"""
Everything `nt` prints is at most `WIDTH` columns, broken at spaces with each line's
indent kept, rather than cut wherever the terminal's edge falls.

The rule holds at the stream, not at each print: the `nt` command writes through
`FixedWidth` for its whole life (`install`), so a message, a warning, an error or a
traceback is wrapped however it was written. A word longer than the width (a path)
is left whole. Some lines are left alone: one rewritten in place (it holds a
carriage return), a traceback's `File "..."` line, which an editor reads as a link,
and a markdown table row or fenced code, which mean nothing once broken. `reflow`
fills a markdown document's paragraphs to the width as wholes, and prints a table
too wide for it as a list, for `nt docs`; the docs keep their code
blocks within the width at the source.
"""

from __future__ import annotations

import io
import re
import sys
import textwrap
from typing import TextIO

#: The terminal's traditional width.
WIDTH = 80

#: A line's indent, and the list marker after it if any: a continuation lines up
#: with the text after both.
_LEAD = re.compile(r"^(\s*)((?:[-*•]|\d+\.) )?")
_TRACEBACK_FILE = re.compile(r'^\s*File "')
_CODE = re.compile(r"`[^`\n]+`")
#: Lines no wrapping touches: a table row and a heading.
_KEPT = re.compile(r"^\s*(\||#)")
#: A line that starts a list item.
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+\.) ")
#: Stands in for a space inside a code span while a line is broken.
_HELD = "\x00"


def wrap_line(line: str, width: int = WIDTH) -> list[str]:
    """`line` as lines of at most `width`, each continuation indented to where the
    first line's text starts, past its indent and any list marker (`- `, `1. `)."""
    if (len(line) <= width or "\r" in line or _TRACEBACK_FILE.match(line)
            or line.lstrip().startswith("|")):
        return [line]
    indent, marker = _LEAD.match(line).groups()
    hang = " " * len(marker or "")
    lines = fill(line[len(indent):], width - len(indent), hang=hang)
    return [indent + w for w in lines] or [line]


def fill(text: str, width: int, indent: str = "", hang: str = "") -> list[str]:
    """`text` as lines of at most `width`, broken at spaces but never inside a
    `code span`: a command or a value reads whole. `indent` starts the first line
    and `hang` every other."""
    # A space inside backticks is held as a character textwrap does not break at.
    held = _CODE.sub(lambda m: m.group(0).replace(" ", _HELD), text)
    lines = textwrap.wrap(held, width, initial_indent=indent, subsequent_indent=hang,
                          break_long_words=False, break_on_hyphens=False)
    return [line.replace(_HELD, " ") for line in lines]


def wrap(text: str, width: int = WIDTH) -> str:
    """Every line of `text` at most `width` columns, its line breaks kept."""
    return "\n".join(w for line in text.split("\n") for w in wrap_line(line, width))


def reflow(markdown: str, width: int = WIDTH) -> str:
    """`markdown` with each paragraph and list item filled to `width` as a whole,
    not line by line, so a source written wider reads as if written for the width.
    A heading and a fenced code block are kept as they are, and so is a table that
    fits; one that does not is printed as a list (`_table`)."""
    out: list[str] = []
    paragraph: list[str] = []
    lead = ""

    def finish() -> None:
        if paragraph:
            indent, marker = _LEAD.match(lead).groups()
            hang = indent + " " * len(marker or "")
            text = " ".join(part.strip() for part in paragraph)
            out.extend(fill(text, width, indent=indent, hang=hang) or [lead.rstrip()])
            paragraph.clear()

    fenced, table = False, []
    for line in [*markdown.split("\n"), ""]:
        if table and not (line.lstrip().startswith("|") and not fenced):
            out.extend(_table(table, width))
            table = []
        if line.lstrip().startswith("```"):
            finish()
            fenced = not fenced
            out.append(line)
        elif not fenced and line.lstrip().startswith("|"):
            finish()
            table.append(line)
        elif fenced or _KEPT.match(line) or not line.strip():
            finish()
            out.append(line)
        elif not paragraph or _ITEM.match(line):
            finish()
            lead, paragraph[:] = line, [line]
        else:
            paragraph.append(line)
    finish()
    return "\n".join(out[:-1])


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _table(rows: list[str], width: int) -> list[str]:
    """A markdown table as it is when every row fits, and otherwise as one list item
    per row: its first cell, then each other cell under its column's header."""
    if all(len(row) <= width for row in rows):
        return rows
    header, *body = rows
    names = _cells(header)
    body = [row for row in body if not set(row.strip()) <= set("|-: ")]
    lines = []
    for row in body:
        first, *rest = _cells(row)
        lines += fill(first, width, indent="- ", hang="  ")
        for name, cell in zip(names[1:], rest):
            lines += fill(f"{name}: {cell}" if name else cell, width, indent="  ", hang="    ")
    return lines


class FixedWidth(io.TextIOBase):
    """`stream`, with every whole line written through `wrap_line`. A partial line
    is held until its newline, or passed on as it is at a flush, so a line rewritten
    in place (`end=""`, then a flush) still updates."""

    def __init__(self, stream: TextIO, width: int = WIDTH):
        self.stream, self.width, self._pending, self._fenced = stream, width, "", False

    def write(self, text: str) -> int:
        *lines, self._pending = (self._pending + text).split("\n")
        for line in lines:
            if line.lstrip().startswith("```"):
                self._fenced = not self._fenced
            kept = self._fenced or line.lstrip().startswith("```")
            self.stream.write(line + "\n" if kept
                              else "\n".join(wrap_line(line, self.width)) + "\n")
        return len(text)

    def flush(self) -> None:
        if self._pending:
            self.stream.write(self._pending)
            self._pending = ""
        self.stream.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    @property
    def encoding(self) -> str:
        return self.stream.encoding

    @property
    def errors(self):
        return self.stream.errors


def install(width: int = WIDTH) -> None:
    """`sys.stdout` and `sys.stderr` at most `width` columns from here on, through
    to the error Python prints as the process exits."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if not isinstance(stream, FixedWidth):
            setattr(sys, name, FixedWidth(stream, width))
