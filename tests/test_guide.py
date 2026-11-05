"""The guide `nt guide` prints: every step and guide the overview lists has a section."""
import re

from neurotemplate.guide import GUIDE, guide_sections


def test_every_step_in_the_overview_has_a_section():
    overview, steps = guide_sections(GUIDE.read_text(encoding="utf-8"))
    listed = re.findall(r"`nt guide ([\w-]+)`", overview)
    assert listed, "the overview lists no steps"
    assert listed == list(steps), (listed, list(steps))


def test_the_guide_is_tracked_so_a_build_from_git_ships_it():
    """Images and wheels are built from committed history (`git archive`), and
    the repo ignores markdown by default; an untracked GUIDE.md built an image
    whose `nt guide` raised FileNotFoundError."""
    import subprocess
    repo = GUIDE.parents[2]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(GUIDE)],
                             cwd=repo, capture_output=True)
    assert tracked.returncode == 0, "GUIDE.md is not tracked by git"


def test_every_reference_is_tracked_and_every_nt_docs_mention_resolves():
    """The references ship with the package, so `nt docs <name>` in the guide, the
    template or a doc must name one, and each must be tracked to reach the wheel."""
    import subprocess
    from neurotemplate.guide import DOCS, doc_paths, hook_sections
    from neurotemplate.init import TEMPLATE
    docs = doc_paths()
    repo = GUIDE.parents[2]
    tracked = subprocess.run(["git", "ls-files", str(DOCS)], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.split()
    assert {repo / t for t in tracked} >= set(docs.values())
    shipped = [GUIDE, *docs.values(), *TEMPLATE.rglob("*.py"), *TEMPLATE.rglob("*.md")]
    mentioned = {name for path in shipped
                 for name in re.findall(r"`nt docs (\w+)`", path.read_text(encoding="utf-8"))}
    known = set(docs) | set(hook_sections()[1]) | {"config"}
    assert mentioned and mentioned <= known, mentioned - known


def test_nt_docs_prints_a_hook_and_the_config_reference(capsys):
    from neurotemplate.cli import main as nt
    nt(["docs", "block_mapping"])
    assert "def block_mapping" in capsys.readouterr().out
    nt(["docs", "config"])
    out = capsys.readouterr().out
    assert "`past`" in out and "`control_bounds`" in out
