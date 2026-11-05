"""`nt init`: the template it writes, and that it never writes over a file."""
import subprocess

import pytest

from neurotemplate.example import EXAMPLES, write_example
from neurotemplate.init import TEMPLATE, template_files, write_template


def test_init_writes_the_template_and_refuses_to_overwrite(tmp_path):
    (tmp_path / "data").mkdir()  # data copied in before init is fine
    write_template(tmp_path)
    for name in ("nt_model.py", "nt_control.py", "AGENTS.md", "CLAUDE.md", "data.yaml",
                 "model_configs/debug.yaml", "control_configs/control.yaml"):
        assert (tmp_path / name).is_file(), name

    (tmp_path / "nt_model.py").write_text("# filled in")
    with pytest.raises(FileExistsError, match="nt_model.py"):
        write_template(tmp_path)
    assert (tmp_path / "nt_model.py").read_text() == "# filled in"


def test_every_template_file_is_tracked_so_a_build_from_git_ships_it():
    """The repo ignores *.yaml and *.md by default, and a wheel leaves ignored
    files out: an untracked config would be missing from every `nt init`."""
    repo = TEMPLATE.parents[2]
    tracked = subprocess.run(["git", "ls-files", str(TEMPLATE)], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.split()
    tracked = {str((repo / t).relative_to(TEMPLATE)) for t in tracked}
    assert {str(f) for f in template_files()} <= tracked
    shipped = subprocess.run(["git", "ls-files", str(EXAMPLES["thermal"])], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.split()
    assert any(name.endswith("control_configs/control.yaml") for name in shipped)
