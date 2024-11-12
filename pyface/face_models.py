from typing import Protocol, Union

import torch

from loguru import logger
from torch import nn
from torch.nn import init

from pyface.config import TrainingConfig
from pyface.models.backbones import BACKBONES
from pyface.models.heads import HEADS


class ForwardProtocol(Protocol):
    """
    Protocol that defines the forward method signature.
    Different implementations can specify different return types.
    """

    def forward(
        self, images: torch.Tensor
    ) -> Union[tuple[torch.Tensor, torch.Tensor], list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        pass

    def forward_features(
        self, images: torch.Tensor
    ) -> Union[torch.Tensor, list[torch.Tensor], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        pass

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                # init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    # init.constant_(m.bias.data, val=0.5)
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight.data, 0, 0.01)
                # init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    # init.constant_(m.bias.data, val=0.5)
                    m.bias.data.zero_()


class FaceRecognitionModel(nn.Module, ForwardProtocol):
    """
    Class combining backbone and head in a single class
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        # self.image_normalisation = torch.nn.BatchNorm2d(3)
        logger.info(f"Create {config.model_config.backbone_name} as backbone")
        self.backbone = BACKBONES[config.model_config.backbone_name](
            embedding_size=config.model_config.embedding_size, **config.model_config.backbone_parameters
        )
        logger.info(f"Create {config.model_config.head_name} as head")
        self.head = HEADS[config.model_config.head_name](
            embedding_size=config.model_config.embedding_size, **config.model_config.head_parameters
        )
        self._init_weights()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        output = self.head(features)
        return output, features

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        # images = self.image_normalisation(images)
        features = self.backbone(images)
        return features


class DeepIDFaceRecognitionModel(nn.Module, ForwardProtocol):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        # self.image_normalisation = torch.nn.BatchNorm2d(3)
        self.backbone = BACKBONES[config.model_config.backbone_name](
            embedding_size=config.model_config.embedding_size, **config.model_config.backbone_parameters
        )
        self.heads: nn.ModuleList = nn.ModuleList(
            [
                HEADS[config.model_config.head_name](
                    embedding_size=config.model_config.embedding_size, **config.model_config.head_parameters
                )
                for idx in range(4)
            ]
        )

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        features = self.forward_features(images)
        outputs = [head(feature) for head, feature in zip(self.heads, features)]
        return outputs, features

    def forward_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        # images = self.image_normalisation(images)
        features = self.backbone(images)
        return features
