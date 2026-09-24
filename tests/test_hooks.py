"""Hook discovery: a hooks file named by `user_hooks_path` runs once, can
import its siblings, and never stands in for the canonical hooks module."""

import sys
from types import SimpleNamespace

import pytest

from neurotemplate.hooks import load_control_hooks, load_modeling_hooks


@pytest.fixture
def hooks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield tmp_path
    for name in ("site_hooks", "site_helpers", "broken_hooks"):
        sys.modules.pop(name, None)


def test_a_hooks_file_runs_once_and_imports_its_siblings(hooks_dir):
    (hooks_dir / "site_helpers.py").write_text("ZONE = 'Zone [°C]'\n")
    (hooks_dir / "site_hooks.py").write_text(
        "import builtins\n"
        "builtins.site_hooks_runs = getattr(builtins, 'site_hooks_runs', 0) + 1\n"
        "from site_helpers import ZONE\n"
        # The canonical module is the real fixture, not this file.
        "import modeling_user_defined\n"
        "assert hasattr(modeling_user_defined, 'data_mapping')\n"
    )
    cfg = SimpleNamespace(user_hooks_path=str(hooks_dir / "site_hooks.py"))
    import builtins
    builtins.site_hooks_runs = 0

    first = load_modeling_hooks(cfg)
    assert load_control_hooks(cfg) is first
    assert load_modeling_hooks(cfg) is first
    assert first.ZONE == "Zone [°C]"
    assert builtins.site_hooks_runs == 1


def test_a_failing_hooks_file_is_not_left_half_imported(hooks_dir):
    (hooks_dir / "broken_hooks.py").write_text("raise RuntimeError('bad hooks')\n")
    cfg = SimpleNamespace(user_hooks_path=str(hooks_dir / "broken_hooks.py"))
    with pytest.raises(RuntimeError, match="bad hooks"):
        load_modeling_hooks(cfg)
    assert "broken_hooks" not in sys.modules
