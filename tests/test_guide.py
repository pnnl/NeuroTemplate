"""The guide `nt guide` prints: every step the overview lists has a section."""
import re

from neurotemplate.guide import GUIDE, guide_sections


def test_every_step_in_the_overview_has_a_section():
    overview, steps = guide_sections(GUIDE.read_text(encoding="utf-8"))
    listed = re.findall(r"`nt guide (\w+)`", overview)
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
