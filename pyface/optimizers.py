from typing import Any, Callable

from torch.optim import SGD, Adamax, AdamW, Optimizer, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LRScheduler,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)

OPTIMIZERS: dict[str, type[Optimizer]] = {"SGD": SGD, "AdamW": AdamW, "RMSprop": RMSprop, "Adamax": Adamax}


SCHEDULERS: dict[str, type[LRScheduler] | Callable[[Any], None]] = {
    "exponential_lr": ExponentialLR,
    "StepLR": StepLR,
    "PolynomialLR": PolynomialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "CyclicLR": CyclicLR,
    "none": lambda x: None,
}
