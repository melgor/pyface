from typing import Type

from torch import nn

from .casianet import CasiaNet
from .deepface import DeepFace
from .deepidplus import DeepID2Plus
from .fudan import FudanResNet
from .resnet import ResNet_50

BACKBONES: dict[str, Type[nn.Module]] = {
    "DeepFace": DeepFace,
    "DeepID2Plus": DeepID2Plus,
    "CasiaNet": CasiaNet,
    "FudanResNet": FudanResNet,
    "ResNet50": ResNet_50,  # type: ignore
}
