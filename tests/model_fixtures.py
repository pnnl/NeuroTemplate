"""ModelingConfig builders shared by the tests."""

import torch

from neurotemplate.modeling.config import ModelingConfig


def model_cfg(**over) -> ModelingConfig:
    """A small model on the CPU, quiet; `over` sets any field."""
    cfg = ModelingConfig()
    cfg.log2_hidden_size = 3
    cfg.num_layers = 1
    cfg.device = torch.device("cpu")
    cfg.verbose = False
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def trainable_cfg(savedir, **over) -> ModelingConfig:
    """A config small and fast enough to actually train, end to end, in a test."""
    return model_cfg(**{
        "model": "mlp", "in_steps": 4, "log2_hidden_size": 2,
        "epochs": 2, "warmup": 0, "patience": 5,
        "train_pred_steps": 4, "dev_pred_steps": 4,
        "test_window": 10, "test_stride": 10, "batch_size": 8,
        "norm_type": "min_max", "plot": False, "seed": 7,
        "savedir": str(savedir), **over,
    })
