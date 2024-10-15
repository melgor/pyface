from torch import nn

from .focal import FocalLoss

LOSSES: dict[str, type[nn.Module]] = {"Focal": FocalLoss, "CrossEntropy": nn.CrossEntropyLoss}
