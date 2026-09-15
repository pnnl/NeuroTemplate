"""Reloading a run trained by an older neurotemplate.

A finished run's MLflow params are a historical record. When a later version
drops a config field, every model trained before that removal still has the old
key logged against it -- and a strict `dict_to_dataclass` would make all of them
permanently unloadable. `_config_from_run_params` drops such keys and says which
ones, so the loss is provenance rather than the model.

The real case this was written for: `synth_gain` and `synth_n_steps`, removed
when the synthetic-scenario training penalty became a post-training physics
assertion.
"""

import pytest

from neurotemplate.modeling.cli import load as load_mod
from neurotemplate.modeling.cli.load import _config_from_run_params


_LIVE = {"model": "nssm", "epochs": 7, "in_steps": 4}


@pytest.fixture
def params(monkeypatch):
    """Install a fake param record; the test mutates what the run logged."""
    record = {}

    def install(extra):
        record.clear()
        record.update({**_LIVE, **extra})
        monkeypatch.setattr(load_mod, "_load_run_params", lambda run_id: dict(record))

    return install


def test_a_dropped_field_does_not_make_the_run_unloadable(params):
    params({"synth_gain": 0.5, "synth_n_steps": 12})
    with pytest.warns(UserWarning, match="synth_gain, synth_n_steps"):
        cfg = _config_from_run_params("run-abc")
    # The fields that survived still arrive, so the architecture is rebuildable.
    assert (cfg.model, cfg.epochs, cfg.in_steps) == ("nssm", 7, 4)
    assert not hasattr(cfg, "synth_gain")


def test_an_unknown_key_is_still_dropped_not_silently_accepted(params):
    # Guards the failure mode where tolerance is implemented by making the
    # dataclass itself permissive: the key must not survive onto the config.
    params({"totally_made_up_field": 1})
    with pytest.warns(UserWarning, match="totally_made_up_field"):
        cfg = _config_from_run_params("run-abc")
    assert not hasattr(cfg, "totally_made_up_field")
