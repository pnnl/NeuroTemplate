"""
Tests for the stateful GRU surrogate.

A GRU block folds each timestep of its window into its hidden state only
once: cold it consumes the whole window, warm it consumes only the newest
step. These tests pin that warm incremental consumption equals a cold pass
over the whole history, which is the invariant the whole design rests on,
and that a rollout starts warmed from its own window and resets between
batches.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from neurotemplate.modeling.config import ModelingConfig as Config
from neurotemplate.modeling.dynamics import get_dynamics
from neurotemplate.modeling.gru import GRUBlock
from model_fixtures import model_cfg, uniform_history, unit_stats
from neurotemplate.modeling.contract import contract_model

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
    return get_dynamics(make_cfg(model, history_steps), nm_mapping, unit_stats(nm_mapping))


def window(data: dict, lo: int, hi: int) -> tuple:
    """The (y, u, d) windows a rollout step ending at hi-1 hands the model."""
    return tuple(data[key][:, lo:hi, :] for key in ("yn", "U", "D"))


# ── The core invariant ──────────────────────────────────────────


def test_warm_consumption_equals_a_cold_pass_over_the_whole_history():
    """The hidden state must summarise every step consumed so far: stepping a
    window forward one step at a time has to land exactly where a single cold
    pass over the full span lands."""
    torch.manual_seed(0)
    history_steps, extra = 4, 3
    data = make_batch(history_steps + extra + P)

    stepped = GRUBlock(NY + NU + ND, history_steps, outsize=NY, hidden_size=8, num_layers=2)
    stepped.set_rollout_state(None)
    with torch.no_grad():
        for lo in range(extra + 1):
            warm = stepped(*window(data, lo, lo + history_steps))

    # Same weights, but a window wide enough to swallow the whole span cold.
    cold = GRUBlock(NY + NU + ND, history_steps + extra, outsize=NY,
                    hidden_size=8, num_layers=2)
    cold.load_state_dict(stepped.state_dict())
    cold.set_rollout_state(None)
    with torch.no_grad():
        once = cold(*window(data, 0, history_steps + extra))

    assert torch.allclose(warm, once, atol=1e-6)


# ── System wiring ───────────────────────────────────────────────


def test_rollout_enters_warmed_from_its_own_window(nm_mapping):
    """The first scored prediction must come from a block warmed over the
    seeded history, i.e. the window ending at history_steps-1."""
    history_steps = 4
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)
    out = system(dict(data))

    model = contract_model(system)
    model.core.cores[0].set_rollout_state(None)
    with torch.no_grad():
        expected = model(*window(data, 0, history_steps))

    assert torch.allclose(out["yn"][:, history_steps, :], expected, atol=1e-6)


def test_hidden_state_resets_between_forwards(nm_mapping):
    """The init_func must reset, or state would leak across batches."""
    history_steps = 4
    data = make_batch(history_steps)
    system = build_system("gru", history_steps, nm_mapping)

    first = system(dict(data))["yn"]
    second = system(dict(data))["yn"]

    assert torch.allclose(first, second)
