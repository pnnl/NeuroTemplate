"""synth_cfg=None must skip the physics checks without error.

The scenarios contribute nothing to the loss, so opting out is purely a
question of whether `build_scenarios` returns None and every caller handles
it. There is no gain to set and no penalty branch to exercise.
"""

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


def test_loss_construction_does_not_take_a_synth_spec():
    """build_loss has no synth path at all — the spec never reaches training."""
    import inspect

    import torch

    from neurotemplate.modeling.config import ModelingConfig as Config
    from neurotemplate.common.config import dict_to_dataclass
    from neurotemplate.modeling.loss import build_loss

    assert "synth_cfg" not in inspect.signature(build_loss).parameters

    cfg = dict_to_dataclass({"device": "cpu", "grad_gain": 0.0}, Config)
    cfg.device = torch.device("cpu")

    class _StubSystem(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.training = True

        def forward(self, *a, **kw):
            return {}

    norm_stats = {"y0": {"shift": 0.0, "stretch": 1.0, "min": 0.0, "max": 1.0,
                         "mean": 0.0, "stddev": 1.0, "median": 0.0}}
    loss = build_loss(
        cfg=cfg, raw_model=torch.nn.Linear(1, 1),
        system=_StubSystem(), grad_masks={},
        nm_mapping={"Y": ["y0"], "U": [], "D": []}, norm_stats=norm_stats,
    )
    assert [o.name for o in loss.objectives] == ["ref_loss"]
