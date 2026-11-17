"""The references `nt docs` prints, and every mention of one."""
import re
import subprocess

from neurotemplate.reference import DOCS, doc_paths, hook_sections


def test_every_reference_is_tracked_and_every_nt_docs_mention_resolves():
    """The references ship with the package, so `nt docs <name>` in the template,
    its guides or a doc must name one, and each must be tracked to reach the
    wheel: images and wheels are built from committed history, and the repo
    ignores markdown by default."""
    from neurotemplate.init import TEMPLATE
    docs = doc_paths()
    repo = DOCS.parents[2]
    tracked = subprocess.run(["git", "ls-files", str(DOCS)], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.split()
    assert {repo / t for t in tracked} >= set(docs.values())
    shipped = [*docs.values(), *TEMPLATE.rglob("*.py"), *TEMPLATE.rglob("*.md")]
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
