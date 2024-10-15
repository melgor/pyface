from typing import Any

import pytest
import torch.nn as nn

from torch.optim import SGD

from pyface.optimizers import OPTIMIZERS, SCHEDULERS


@pytest.mark.parametrize("optim_name, params", [(name, {}) for name in OPTIMIZERS.keys()])
def test_optimizer(optim_name: str, params: dict[str, Any]):
    """Test creation of each optimizer"""
    model = nn.Linear(100, 100)
    optimizer = OPTIMIZERS[optim_name](model.parameters(), **params)
    assert optimizer is not None


@pytest.mark.parametrize(
    "scheduler_name, params",
    [
        ("exponential_lr", {"gamma": 0.8}),
        ("StepLR", {"step_size": 100}),
        ("PolynomialLR", {}),
        ("ReduceLROnPlateau", {}),
        ("CosineAnnealingLR", {"T_max": 100}),
        ("CosineAnnealingWarmRestarts", {"T_0": 100}),
        ("CyclicLR", {"base_lr": 0.1, "max_lr": 1.0}),
        ("none", {}),
    ],
)
def test_schedulers(scheduler_name: str, params: dict[str, Any]):
    """Test creation of each scheduler"""
    model = nn.Linear(100, 100)
    optimizer = SGD(model.parameters())
    scheduler = SCHEDULERS[scheduler_name](optimizer, **params)
