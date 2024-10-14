import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of Classification layer
"""


class ClassificationLayer(nn.Module):
    def __init__(self, input_features: int, nb_classes: int):
        super().__init__()
        self._input_features = input_features
        self._nb_classes = nb_classes
        self._layer = nn.Linear(in_features=input_features, out_features=nb_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self._layer(features)
        return output
