from torch import nn

from .focal import FocalLoss

LOSSES: dict[str, type[nn.Module]] = {"Focal": FocalLoss, "Softmax": nn.CrossEntropyLoss}
