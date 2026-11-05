"""ModelingConfig builders shared by the tests."""

import torch

from neurotemplate.modeling.config import ModelingConfig


def steps(n: int):
    """`n` steps of the fixtures' 5-minute data, as a config writes a window."""
    return f"{5 * n}m" if n else 0


def model_cfg(**over) -> ModelingConfig:
    """A small model on the CPU at 5-minute steps, quiet; `over` sets any field."""
    cfg = ModelingConfig()
    cfg.steps_per_hour = 12.0
    cfg.past, cfg.context = 0, 0
    cfg.horizon, cfg.test_horizon = steps(48), steps(287)
    cfg.train_stride, cfg.dev_stride, cfg.test_stride = steps(4), steps(4), steps(288)
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.device = torch.device("cpu")
    cfg.progress = "silent"
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def uniform_past(nm_mapping: dict[str, list[str]], rows: int,
                    at_zero: tuple[str, ...] = ()) -> dict[str, int]:
    """A `past` table reading every Y, U and D column `rows` rows deep, except the
    columns in `at_zero`, read at the current step only."""
    return {c: 0 if c in at_zero else steps(rows - 1)
            for key in ("Y", "U", "D") for c in nm_mapping[key]}


def trainable_cfg(savedir, nm_mapping: dict[str, list[str]], **over) -> ModelingConfig:
    """A config small and fast enough to actually train, end to end, in a test."""
    return model_cfg(**{
        "model": "mlp", "past": uniform_past(nm_mapping, 4), "log2_hidden_size": 2,
        "epochs": 2, "warmup": 0, "patience": 5,
        "horizon": steps(4), "test_horizon": steps(6), "test_stride": steps(10), "batch_size": 8,
        "norm_type": "min_max", "plot": False, "seed": 7,
        "savedir": str(savedir), **over,
    })


def unit_stats(nm_mapping: dict[str, list[str]]) -> dict[str, dict]:
    """Identity norm stats: every column reaches a built-in family in its own
    units, so a test's numbers mean the same on both sides of the contract."""
    return {c: {"norm_type": "none", "shift": 0.0, "stretch": 1.0, "min": -1.0,
                "max": 1.0, "mean": 0.0, "stddev": 1.0, "median": 0.0, "unit": None}
            for key in ("Y", "U", "D") for c in nm_mapping[key]}
