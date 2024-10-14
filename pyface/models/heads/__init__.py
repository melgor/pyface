from torch import nn

from .classification import ClassificationLayer

HEADS: dict[str, type[nn.Module]] = {"ClassificationLayer": ClassificationLayer}
