from typing import Any

from torch import nn
from torch.optim import SGD, Adamax, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

from .config import OptimisationConfig


def get_optimizer(config: OptimisationConfig, model: nn.Module) -> Optimizer:
    name = config.optimizer
    if name == "sgd":
        return SGD(model.parameters(), **config.optimizer_config)
    elif name == "adamw":
        return AdamW(model.parameters(), **config.optimizer_config)
    elif name == "rmsprop":
        return RMSprop(model.parameters(), **config.optimizer_config)
    elif name == "adamax":
        return Adamax(model.parameters(), **config.optimizer_config)
    else:
        raise NotImplementedError(f"Optimizer {name} is not supported.")


def get_scheduler(config: OptimisationConfig, optimizer: Optimizer) -> Any:
    name = config.scheduler
    if name == "exponential_lr":
        return ExponentialLR(optimizer, **config.scheduler_config)
    elif name == "step":
        return StepLR(optimizer, **config.scheduler_config)
    elif name == "polynomial":
        return PolynomialLR(optimizer, **config.scheduler_config)
    elif name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, **config.scheduler_config)
    elif name == "cosine_annealing":
        return CosineAnnealingLR(optimizer, **config.scheduler_config)
    elif name == "cosine_annealing_warm_rest":
        return CosineAnnealingWarmRestarts(optimizer, **config.scheduler_config)
    elif name == "cyclic":
        return CyclicLR(optimizer, **config.scheduler_config)
    elif name == "none":
        return None
    else:
        raise NotImplementedError(f"Scheduler {name} is not supported.")
