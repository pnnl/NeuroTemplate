"""Hook discovery: a hooks file named by `user_hooks_path` runs once, can
import its siblings, and never stands in for the canonical hooks module. And
what `data_mapping` returns is checked by the library, not the hook."""

import sys
from types import SimpleNamespace

import pytest

from neurotemplate.hooks import load_control_hooks, load_modeling_hooks


@pytest.fixture
def hooks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield tmp_path
    for name in list(sys.modules):
        if name.startswith(("site_hooks", "site_helpers", "broken_hooks")):
            del sys.modules[name]


def test_a_hooks_file_runs_once_and_imports_its_siblings(hooks_dir):
    (hooks_dir / "site_helpers.py").write_text("ZONE = 'Zone [°C]'\n")
    (hooks_dir / "site_hooks.py").write_text(
        "import builtins\n"
        "builtins.site_hooks_runs = getattr(builtins, 'site_hooks_runs', 0) + 1\n"
        "from site_helpers import ZONE\n"
        # The canonical module is the real fixture, not this file.
        "import nt_model\n"
        "assert hasattr(nt_model, 'data_mapping')\n"
    )
    cfg = SimpleNamespace(user_hooks_path=str(hooks_dir / "site_hooks.py"))
    import builtins
    builtins.site_hooks_runs = 0

    first = load_modeling_hooks(cfg)
    assert load_control_hooks(cfg) is first
    assert load_modeling_hooks(cfg) is first
    assert first.ZONE == "Zone [°C]"
    assert builtins.site_hooks_runs == 1


def test_the_library_drops_excluded_columns_and_flags_a_typo():
    import pandas as pd
    from neurotemplate.modeling.pipeline import resolve_nm_mapping

    df = pd.DataFrame(columns=["Zone [°C]", "Zone Spare [°C]", "Setpoint [°C]", "sec_elapsed"])
    hooks = SimpleNamespace(data_mapping=lambda df: {
        "Y": ["Zone [°C]", "Zone Spare [°C]"], "U": ["Setpoint [°C]"], "D": [],
        "t": ["sec_elapsed"]})

    mapping = resolve_nm_mapping(hooks, df, SimpleNamespace(exclude_columns=["Spare"]))
    assert mapping["Y"] == ["Zone [°C]"]
    with pytest.raises(ValueError, match="match no column: \\['Sapre'\\]"):
        resolve_nm_mapping(hooks, df, SimpleNamespace(exclude_columns=["Sapre"]))
