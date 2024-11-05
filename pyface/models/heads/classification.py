from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

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
    def __init__(self, embedding_size: int, nb_classes: int, dropout_rate: float = 0.0):
        super().__init__(embedding_size, nb_classes)
        logger.info(f"ClassificationLayer with {nb_classes} classes")
        self._layer = nn.Linear(in_features=embedding_size, out_features=nb_classes)
        self._dropout = nn.Dropout(dropout_rate)
        self._activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features = self._dropout(features)
        features = self._activation(features)
        output = self._layer(features)
        return output
