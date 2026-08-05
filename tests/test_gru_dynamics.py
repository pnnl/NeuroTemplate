"""
Tests for the stateful GRU surrogate and its hidden-state warmup.

The GRU reads one timestep at a time, so `cfg.in_steps` is a warmup horizon
rather than an input-window width: `make_warmup_init` replays the seeded
history through the hidden state before the rollout starts. These tests pin
(a) that a GRU and an MLP at the same in_steps get the same training windows,
(b) that the replay actually happens and lands on the right indices, and
(c) that in_steps=1 still behaves exactly as it did before warmup existed.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.common.data_utils import get_sw_dict
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics

NY, NU, ND = 2, 1, 1
P = 5  # rollout steps past the history


@pytest.fixture
def nm_mapping():
    return {"Y": ["y0", "y1"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}


@pytest.fixture
def synthetic_df():
    n = 100
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "y0": rng.normal(20.0, 2.0, n),
        "y1": rng.normal(22.0, 1.5, n),
        "u0": rng.uniform(0.0, 1.0, n),
        "d0": rng.normal(15.0, 5.0, n),
        "sec_elapsed": np.arange(n) * 300.0,
    })


def make_cfg(model: str, in_steps: int) -> Config:
    cfg = Config()
    cfg.model = model
    cfg.in_steps = in_steps
    cfg.train_pred_steps = P
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.device = torch.device("cpu")
    cfg.verbose = False
    return cfg


def make_batch(in_steps: int, batch: int = 3) -> dict:
    """Mirrors the layout get_sw_dict produces: yn is the first in_steps of Y."""
    total = in_steps + P
    torch.manual_seed(0)
    Y = torch.randn(batch, total, NY)
    return {
        "yn": Y[:, :in_steps, :].contiguous(),
        "Y": Y,
        "U": torch.randn(batch, total, NU),
        "D": torch.randn(batch, total, ND),
    }


def build_system(model: str, in_steps: int, nm_mapping):
    """Deterministic weights so cross-run comparisons are meaningful."""
    torch.manual_seed(1234)
    return get_dynamics(make_cfg(model, in_steps), nm_mapping)


# ── Window parity: the reason this change exists ────────────────


def test_gru_and_mlp_produce_identical_windows(synthetic_df, nm_mapping):
    """At the same in_steps both families must see the same training set."""
    gru = get_sw_dict(synthetic_df, make_cfg("gru", 12), nm_mapping)
    mlp = get_sw_dict(synthetic_df, make_cfg("mlp", 12), nm_mapping)

    for key in ["Y", "U", "D", "yn"]:
        assert gru[key].shape == mlp[key].shape, f"{key} window shape diverged"
        assert torch.equal(gru[key], mlp[key])


def test_in_steps_one_inflates_the_window_count(synthetic_df, nm_mapping):
    """Regression on the original symptom: stride == in_steps, so in_steps=1
    yields ~12x the windows of in_steps=12 over the same data."""
    wide = get_sw_dict(synthetic_df, make_cfg("gru", 12), nm_mapping)
    narrow = get_sw_dict(synthetic_df, make_cfg("gru", 1), nm_mapping)

    assert narrow["Y"].shape[0] > 10 * wide["Y"].shape[0]


def test_gru_and_mlp_rollouts_have_the_same_shape(nm_mapping):
    in_steps = 12
    data = make_batch(in_steps)
    gru_out = build_system("gru", in_steps, nm_mapping)(dict(data))
    mlp_out = build_system("mlp", in_steps, nm_mapping)(dict(data))

    assert gru_out["yn"].shape == mlp_out["yn"].shape
    assert gru_out["yn"].shape == (data["yn"].shape[0], in_steps + P + 1, NY)


# ── Warmup behaviour ────────────────────────────────────────────


def test_warmup_changes_the_first_prediction(nm_mapping):
    """A warmed hidden state must not agree with a cold one."""
    in_steps = 4
    data = make_batch(in_steps)
    system = build_system("gru", in_steps, nm_mapping)
    out = system(dict(data))

    model = system.nodes[0].callable
    model.fy.reset()
    cold = model(
        data["yn"][:, in_steps - 1, :],
        data["U"][:, in_steps - 1, :],
        data["D"][:, in_steps - 1, :],
    )

    assert not torch.allclose(out["yn"][:, in_steps, :], cold)


def test_warmup_replays_exactly_the_expected_indices(nm_mapping):
    """Off-by-one detector: replaying steps [0, in_steps-2] by hand and taking
    one more step must reproduce the system's first prediction."""
    in_steps = 4
    data = make_batch(in_steps)
    system = build_system("gru", in_steps, nm_mapping)
    out = system(dict(data))

    model = system.nodes[0].callable
    model.fy.reset()
    with torch.no_grad():
        for i in range(in_steps - 1):
            model(data["yn"][:, i, :], data["U"][:, i, :], data["D"][:, i, :])
        expected = model(
            data["yn"][:, in_steps - 1, :],
            data["U"][:, in_steps - 1, :],
            data["D"][:, in_steps - 1, :],
        )

    assert torch.allclose(out["yn"][:, in_steps, :], expected, atol=1e-6)


def test_hidden_state_resets_between_forwards(nm_mapping):
    """The init_func must reset, or state would leak across batches."""
    in_steps = 4
    data = make_batch(in_steps)
    system = build_system("gru", in_steps, nm_mapping)

    first = system(dict(data))["yn"]
    second = system(dict(data))["yn"]

    assert torch.allclose(first, second)


def test_warmup_leaves_the_seeded_history_untouched(nm_mapping):
    """init_func must return the data dict unmodified."""
    in_steps = 4
    data = make_batch(in_steps)
    out = build_system("gru", in_steps, nm_mapping)(dict(data))

    assert torch.equal(out["yn"][:, :in_steps, :], data["yn"])


def test_in_steps_one_matches_a_hand_rolled_rollout(nm_mapping):
    """With no history there is nothing to warm up — behaviour is unchanged."""
    in_steps = 1
    data = make_batch(in_steps)
    system = build_system("gru", in_steps, nm_mapping)
    assert system.nodes[0].input_map == {}

    out = system(dict(data))

    model = system.nodes[0].callable
    model.fy.reset()
    y = data["yn"][:, 0, :]
    expected = [y]
    with torch.no_grad():
        for i in range(data["Y"].shape[1]):
            y = model(y, data["U"][:, i, :], data["D"][:, i, :])
            expected.append(y)

    assert torch.allclose(out["yn"], torch.stack(expected, dim=1), atol=1e-6)


# ── Alignment with the loss and the eval metric ─────────────────


@pytest.mark.parametrize("in_steps", [1, 12])
def test_slicing_alignment_holds(in_steps, nm_mapping):
    """loss.py slices yn[:, in_steps:-1] against Y[:, in_steps:]; both must
    line up, and yn must carry exactly one extra timestep."""
    data = make_batch(in_steps)
    out = build_system("gru", in_steps, nm_mapping)(dict(data))

    assert out["yn"].shape[1] == out["Y"].shape[1] + 1
    assert (
        out["yn"][:, in_steps:-1, :].shape == out["Y"][:, in_steps:, :].shape
    )
