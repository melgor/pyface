from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of Classification layer
"""


class HeadLayer(ABC, nn.Module):
    def __init__(self, embedding_size: int, nb_classes: int):
        super().__init__()
        self._embedding_size = embedding_size
        self._nb_classes = nb_classes

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor: ...


class ClassificationLayer(HeadLayer):
    def __init__(self, embedding_size: int, nb_classes: int):
        super().__init__(embedding_size, nb_classes)
        self._layer = nn.Linear(in_features=embedding_size, out_features=nb_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self._layer(features)
        return output
