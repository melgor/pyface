from torch import nn

from .contrastive_loss import ContrastiveLoss
from .focal import FocalLoss

LOSSES: dict[str, type[nn.Module]] = {
    "Focal": FocalLoss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "ContrastiveLoss": ContrastiveLoss,
}
