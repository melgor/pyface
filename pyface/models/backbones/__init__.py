from torch import nn

from .deepface import DeepFace

BACKBONES: dict[str, type[nn.Module]] = {"DeepFace": DeepFace}
