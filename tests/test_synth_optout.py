"""synth_cfg=None / synth_gain=0 must be skipped without error."""

import pytest

from neurotemplate.modeling.scenarios import build_scenarios


def test_build_scenarios_returns_none_when_cfg_none():
    """Opt-out path: synth_cfg=None → no scenarios constructed."""
    result = build_scenarios(
        system=None,  # not consulted on opt-out
        nm_mapping={"Y": ["y0"], "U": [], "D": []},
        norm_stats={},
        synth_cfg=None,
        in_steps=4, n_scenarios=4, n_steps=8, device="cpu",
    )
    assert result is None


def test_loss_construction_skips_synth_when_gain_zero():
    """build_loss must ignore synth_cfg entirely when synth_gain=0."""
    import torch

    from neurotemplate.modeling.config import ModelingConfig as Config
    from neurotemplate.common.config import dict_to_dataclass
    from neurotemplate.modeling.loss import build_loss

    cfg = dict_to_dataclass({
                "device": "cpu",
        "synth_gain": 0.0,
        "grad_gain": 0.0,
    }, Config)
    cfg.device = torch.device("cpu")

    class _StubSystem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.training = True
        def forward(self, *a, **kw): return {}

    nm_mapping = {"Y": ["y0"], "U": [], "D": []}
    norm_stats = {"y0": {"shift": 0.0, "stretch": 1.0,
                          "min": 0.0, "max": 1.0,
                          "mean": 0.0, "stddev": 1.0, "median": 0.0}}
    # synth_cfg=None should be fine when synth_gain=0.
    loss = build_loss(
        cfg=cfg, raw_model=torch.nn.Linear(1, 1),
        system=_StubSystem(), grad_masks={},
        nm_mapping=nm_mapping, norm_stats=norm_stats,
        synth_cfg=None,
    )
    assert loss is not None


def test_loss_raises_when_synth_gain_set_but_cfg_none():
    """If caller passes synth_cfg=None but sets gain>0, fail fast."""
    import torch

    from neurotemplate.modeling.config import ModelingConfig as Config
    from neurotemplate.common.config import dict_to_dataclass
    from neurotemplate.modeling.loss import build_loss

    cfg = dict_to_dataclass({
        "device": "cpu",
        "synth_gain": 0.5, "grad_gain": 0.0,
    }, Config)
    cfg.device = torch.device("cpu")

    class _StubSystem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.training = True
        def forward(self, *a, **kw): return {}

    nm_mapping = {"Y": ["y0"], "U": [], "D": []}
    norm_stats = {"y0": {"shift": 0.0, "stretch": 1.0,
                         "min": 0.0, "max": 1.0,
                         "mean": 0.0, "stddev": 1.0, "median": 0.0}}
    with pytest.raises(ValueError, match="synth_gain"):
        build_loss(
            cfg=cfg, raw_model=torch.nn.Linear(1, 1),
            system=_StubSystem(), grad_masks={},
            nm_mapping=nm_mapping, norm_stats=norm_stats,
            synth_cfg=None,
        )
