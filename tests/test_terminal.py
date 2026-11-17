"""What `nt` prints is at most 80 columns, broken where a reader would break it."""
import io
import subprocess
import sys

from neurotemplate.common.terminal import WIDTH, FixedWidth, reflow, wrap, wrap_line

LONG = "word " * 30


def test_a_long_line_breaks_at_spaces_and_keeps_its_indent_and_list_marker():
    lines = wrap_line("    - " + LONG.strip())
    assert len(lines) > 1 and all(len(line) <= WIDTH for line in lines)
    assert lines[0].startswith("    - word")
    assert all(line.startswith("      word") for line in lines[1:])


def test_a_code_span_a_path_a_rewritten_line_and_a_traceback_link_stay_whole():
    command = "`nt model train --config custom`"
    for line in wrap_line(LONG + command):
        assert command in line or "`" not in line
    path = "/" + "x" * 100
    assert path in wrap_line(f"see {path} there")
    rewritten = "\r" + LONG
    assert wrap_line(rewritten) == [rewritten]
    link = f'  File "{path}", line 12, in main'
    assert wrap_line(link) == [link]


def test_a_line_breaks_keep_their_place():
    assert wrap(f"one\n\n{LONG}").startswith("one\n\nword")


def test_the_stream_wraps_whole_lines_and_passes_a_partial_one_at_a_flush():
    out = io.StringIO()
    stream = FixedWidth(out)
    stream.write(LONG)
    assert out.getvalue() == ""              # held until its newline
    stream.write("\nepoch 1")
    stream.flush()                           # a line rewritten in place still shows
    assert all(len(line) <= WIDTH for line in out.getvalue().splitlines())
    assert out.getvalue().endswith("\nepoch 1")


def test_the_nt_command_prints_within_the_width_through_to_its_exit_error():
    for args in (["docs", "hooks"], ["docs", "config"], ["docs", "modeling"],
                 ["model", "train", "--config", "nowhere"]):
        done = subprocess.run([sys.executable, "-m", "neurotemplate.cli", *args],
                              capture_output=True, text=True)
        for line in (done.stdout + done.stderr).splitlines():
            assert len(line) <= WIDTH or " " not in line.strip() or 'File "' in line, line


def test_reflow_fills_a_paragraph_and_a_list_item_as_wholes():
    text = reflow("A first line that is\nwrapped short.\n\n- an item\n  that goes on\n")
    assert text == "A first line that is wrapped short.\n\n- an item that goes on\n"


def test_reflow_keeps_code_a_heading_and_a_table_that_fits():
    code = "```\n" + "x = 1  " * 20 + "\n```"
    table = "| a | b |\n|---|---|\n| 1 | 2 |"
    for kept in (code, "# " + LONG, table):
        assert reflow(kept) == kept


def test_reflow_prints_a_table_too_wide_as_a_list_under_its_headers():
    text = reflow(f"| step | decides |\n|---|---|\n| `setup` | {LONG} |")
    assert text.startswith("- `setup`\n  decides: word")
    assert all(len(line) <= WIDTH for line in text.splitlines())


def test_every_code_block_the_docs_print_fits_the_width():
    from neurotemplate.reference import DOCS
    for path in DOCS.glob("*.md"):
        fenced = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            elif fenced:
                assert len(line) <= WIDTH, f"{path.name}:{number} is {len(line)} wide"


def test_a_setup_mistake_prints_the_message_without_a_traceback(tmp_path):
    done = subprocess.run([sys.executable, "-c", "from neurotemplate.cli import run; run()",
                           "model", "train", "--config", "nowhere"],
                          capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode != 0
    assert "Traceback" not in done.stderr and "nt: " in done.stderr, done.stderr
