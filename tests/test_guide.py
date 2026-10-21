"""The guide `nt guide` prints: every step the overview lists has a section."""
import re

from neurotemplate.guide import GUIDE, guide_sections


def test_every_step_in_the_overview_has_a_section():
    overview, steps = guide_sections(GUIDE.read_text(encoding="utf-8"))
    listed = re.findall(r"`nt guide (\w+)`", overview)
    assert listed, "the overview lists no steps"
    assert listed == list(steps), (listed, list(steps))
