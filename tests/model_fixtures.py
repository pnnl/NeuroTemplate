"""ModelingConfig builders shared by the tests."""

import torch

from neurotemplate.modeling.config import ModelingConfig


def model_cfg(**over) -> ModelingConfig:
    """A small model on the CPU, quiet; `over` sets any field."""
    cfg = ModelingConfig()
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.device = torch.device("cpu")
    cfg.progress = "silent"
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def uniform_history(nm_mapping: dict[str, list[str]], steps: int,
                    at_zero: tuple[str, ...] = ()) -> dict[str, int]:
    """A `history` table reading every Y, U and D column `steps` rows deep,
    except the columns in `at_zero`, read at the current step only."""
    return {c: 0 if c in at_zero else steps - 1
            for key in ("Y", "U", "D") for c in nm_mapping[key]}


def trainable_cfg(savedir, nm_mapping: dict[str, list[str]], **over) -> ModelingConfig:
    """A config small and fast enough to actually train, end to end, in a test."""
    return model_cfg(**{
        "model": "mlp", "history": uniform_history(nm_mapping, 4), "log2_hidden_size": 2,
        "epochs": 2, "warmup": 0, "patience": 5,
        "train_pred_steps": 4, "dev_pred_steps": 4,
        "test_window": 10, "test_stride": 10, "batch_size": 8,
        "norm_type": "min_max", "plot": False, "seed": 7,
        "savedir": str(savedir), **over,
    })


def unit_stats(nm_mapping: dict[str, list[str]]) -> dict[str, dict]:
    """Identity norm stats: every column reaches a built-in family in its own
    units, so a test's numbers mean the same on both sides of the contract."""
    return {c: {"norm_type": "none", "shift": 0.0, "stretch": 1.0, "min": -1.0,
                "max": 1.0, "mean": 0.0, "stddev": 1.0, "median": 0.0, "unit": None}
            for key in ("Y", "U", "D") for c in nm_mapping[key]}
