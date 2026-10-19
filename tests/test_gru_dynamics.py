"""
Tests for the stateful GRU surrogate.

The GRU reads the same `cfg.history_steps` window every other family reads, but
folds each timestep into its hidden state only once: cold it consumes the
whole window, warm it consumes only the newest step. These tests pin
(a) that a GRU and an MLP at the same history_steps get the same training windows,
(b) that warm incremental consumption equals a cold pass over the whole
    history, which is the invariant the whole design rests on,
(c) that a block instantiated on its own — no System, no pipeline — is
    correct on its first call, and
(d) that history_steps=1 still behaves exactly as it did before windowing existed.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.common.data_utils import get_sw_dict
from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.gru import GRUBlock
from model_fixtures import model_cfg, uniform_history

NY, NU, ND = 2, 1, 1
P = 5  # rollout steps past the history


@pytest.fixture
def nm_mapping():
    return {key: list(columns) for key, columns in NM_MAPPING.items()}


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


NM_MAPPING = {"Y": ["y0", "y1"], "U": ["u0"], "D": ["d0"], "t": ["sec_elapsed"]}


def make_cfg(model: str, history_steps: int) -> Config:
    return model_cfg(model=model, history=uniform_history(NM_MAPPING, history_steps),
                     train_pred_steps=P, num_layers=2)


def make_batch(history_steps: int, batch: int = 3) -> dict:
    """Mirrors the layout get_sw_dict produces: yn is the first history_steps of Y."""
    total = history_steps + P
    torch.manual_seed(0)
    Y = torch.randn(batch, total, NY)
    return {
        "yn": Y[:, :history_steps, :].contiguous(),
        "Y": Y,
        "U": torch.randn(batch, total, NU),
        "D": torch.randn(batch, total, ND),
    }


def build_system(model: str, history_steps: int, nm_mapping):
    """Deterministic weights so cross-run comparisons are meaningful."""
    torch.manual_seed(1234)
    return get_dynamics(make_cfg(model, history_steps), nm_mapping)


def flat_window(data: dict, lo: int, hi: int) -> tuple:
    """The (y, u, d) window SSMGrad receives for a rollout step ending at hi-1.

    Each key is flattened time-major on its own, which is what the Node's
    input_map produces.
    """
    return tuple(
        data[key][:, lo:hi, :].reshape(data[key].shape[0], -1)
        for key in ("yn", "U", "D")
    )


# ── Window parity: the reason this change exists ────────────────


def test_gru_and_mlp_produce_identical_windows(synthetic_df, nm_mapping):
    """At the same history_steps both families must see the same training set."""
    gru = get_sw_dict(synthetic_df, make_cfg("gru", 12), nm_mapping)
    mlp = get_sw_dict(synthetic_df, make_cfg("mlp", 12), nm_mapping)

    for key in ["Y", "U", "D", "yn"]:
        assert gru[key].shape == mlp[key].shape, f"{key} window shape diverged"
        assert torch.equal(gru[key], mlp[key])


# ── The core invariant ──────────────────────────────────────────


def test_warm_consumption_equals_a_cold_pass_over_the_whole_history():
    """The hidden state must summarise every step consumed so far: stepping a
    window forward one step at a time has to land exactly where a single cold
    pass over the full span lands."""
    torch.manual_seed(0)
    history_steps, extra = 4, 3
    data = make_batch(history_steps + extra + P)

    stepped = GRUBlock(NY, NU, ND, history_steps, outsize=NY, hidden_size=8, num_layers=2)
    stepped.set_rollout_state(None)
    with torch.no_grad():
        for lo in range(extra + 1):
            warm = stepped(torch.cat(flat_window(data, lo, lo + history_steps), dim=-1))

    # Same weights, but a window wide enough to swallow the whole span cold.
    cold = GRUBlock(NY, NU, ND, history_steps + extra, outsize=NY,
                    hidden_size=8, num_layers=2)
    cold.load_state_dict(stepped.state_dict())
    cold.set_rollout_state(None)
    with torch.no_grad():
        once = cold(torch.cat(flat_window(data, 0, history_steps + extra), dim=-1))

    assert torch.allclose(warm, once, atol=1e-6)


def test_manual_vmap_path_matches_the_fused_kernel():
    """jacrev/vmap uses hand-rolled GRU math; it must compute the same function
    as nn.GRU, including across layers."""
    torch.manual_seed(0)
    history_steps = 4
    block = GRUBlock(NY, NU, ND, history_steps, outsize=NY, hidden_size=8, num_layers=3)
    x = torch.randn(history_steps * (NY + NU + ND))

    with torch.no_grad():
        block.set_rollout_state(None)
        manual = block(x)                  # 1D -> _manual_gru
        block.set_rollout_state(None)
        fused = block(x.unsqueeze(0))      # 2D -> nn.GRU

    assert torch.allclose(manual, fused[0], atol=1e-6)


# ── System wiring ───────────────────────────────────────────────


def test_rollout_enters_warmed_from_its_own_window(nm_mapping):
    """The first scored prediction must come from a block warmed over the
    seeded history, i.e. the window ending at history_steps-1."""
    history_steps = 4
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)
    out = system(dict(data))

    model = system.nodes[0].callable
    model.fy.set_rollout_state(None)
    with torch.no_grad():
        expected = model(*flat_window(data, 0, history_steps))

    assert torch.allclose(out["yn"][:, history_steps, :], expected, atol=1e-6)


def test_history_actually_reaches_the_prediction(nm_mapping):
    """Guards the regression this replaced: the block must not ignore its
    window. Two windows sharing a newest step but differing earlier must
    predict differently."""
    history_steps = 4
    data = make_batch(history_steps)
    model = build_system("gru", history_steps, nm_mapping).nodes[0].callable

    other = dict(data)
    other["yn"] = torch.cat(                 # same newest step, history reversed
        [data["yn"][:, :-1, :].flip(1), data["yn"][:, -1:, :]], dim=1
    )

    with torch.no_grad():
        model.fy.set_rollout_state(None)
        first = model(*flat_window(data, 0, history_steps))
        model.fy.set_rollout_state(None)
        second = model(*flat_window(other, 0, history_steps))

    assert not torch.allclose(first, second)


def test_hidden_state_resets_between_forwards(nm_mapping):
    """The init_func must reset, or state would leak across batches."""
    history_steps = 4
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)

    first = system(dict(data))["yn"]
    second = system(dict(data))["yn"]

    assert torch.allclose(first, second)


def test_warmup_leaves_the_seeded_history_untouched(nm_mapping):
    """init_func must return the data dict unmodified."""
    history_steps = 4
    data = make_batch(history_steps)
    out = build_system("gru", history_steps, nm_mapping)(dict(data))

    assert torch.equal(out["yn"][:, :history_steps, :], data["yn"])


def test_gradients_flow_through_the_warmup(nm_mapping):
    """The warmed hidden state is a function of the weights, so the loss must
    be able to shape it."""
    history_steps = 4
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)

    system(dict(data))["yn"].square().mean().backward()

    grads = [p.grad for p in system.parameters() if p.requires_grad]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads)


def test_a_one_step_window_matches_a_hand_rolled_rollout(nm_mapping):
    """With no history there is nothing to warm up — behaviour is unchanged."""
    history_steps = 1
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)

    out = system(dict(data))

    model = system.nodes[0].callable
    model.fy.set_rollout_state(None)
    y = data["yn"][:, 0, :]
    expected = [y]
    with torch.no_grad():
        for i in range(data["Y"].shape[1]):
            y = model(y, data["U"][:, i, :], data["D"][:, i, :])
            expected.append(y)

    assert torch.allclose(out["yn"], torch.stack(expected, dim=1), atol=1e-6)


# ── Alignment with the loss and the eval metric ─────────────────


@pytest.mark.parametrize("history_steps", [1, 12])
def test_slicing_alignment_holds(history_steps, nm_mapping):
    """loss.py slices yn[:, history_steps:-1] against Y[:, history_steps:]; both must
    line up, and yn must carry exactly one extra timestep."""
    data = make_batch(history_steps)
    out = build_system("gru", history_steps, nm_mapping)(dict(data))

    assert out["yn"].shape[1] == out["Y"].shape[1] + 1
    assert (
        out["yn"][:, history_steps:-1, :].shape == out["Y"][:, history_steps:, :].shape
    )
