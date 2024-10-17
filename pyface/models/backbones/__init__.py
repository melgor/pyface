from torch import nn

from .deepface import DeepFace
from .deepidplus import DeepID2Plus

BACKBONES: dict[str, type[nn.Module]] = {"DeepFace": DeepFace, "DeepID2Plus": DeepID2Plus}
