import time

from typing import Optional, Protocol, Union

import lightning as pl
import torch

from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import OptimisationConfig, TrainingConfig
from .models.backbones import BACKBONES
from .models.heads import HEADS
from .models.losses import LOSSES
from .optimizers import OPTIMIZERS, SCHEDULERS


def get_optimizer(config: OptimisationConfig, model: nn.Module) -> Optimizer:
    return OPTIMIZERS[config.optimizer_name](model.parameters(), **config.optimizer_config)


def get_scheduler(config: OptimisationConfig, optimizer: Optimizer) -> Optional[LRScheduler]:
    return SCHEDULERS[config.scheduler_name](optimizer, **config.scheduler_config)


class ForwardProtocol(Protocol):
    """
    Protocol that defines the forward method signature.
    Different implementations can specify different return types.
    """

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        pass


class FaceRecognitionModel(nn.Module, ForwardProtocol):
    """
    Class combining backbone and head in a single class
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.image_normalisation = torch.nn.BatchNorm2d(3)
        self.backbone = BACKBONES[config.model_config.backbone_name](
            embedding_size=config.model_config.embedding_size, **config.model_config.backbone_parameters
        )
        self.head = HEADS[config.model_config.head_name](
            embedding_size=config.model_config.embedding_size, **config.model_config.head_parameters
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.image_normalisation(images)
        features = self.backbone(images)
        output = self.head(features)
        return output


class DeepIDFaceRecognitionModel(nn.Module, ForwardProtocol):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.image_normalisation = torch.nn.BatchNorm2d(3)
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

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        images = self.image_normalisation(images)
        features = self.backbone(images)
        outputs = [head(feature) for head, feature in zip(self.heads, features)]
        return outputs


class FaceRecognitionLightningModule(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.epoch_start_time: Optional[float] = None
        self.epoch_end_time: Optional[float] = None
        if self.config.channel_last:
            self.memory_format = torch.channels_last
        else:
            self.memory_format = torch.contiguous_format

        self.model = FaceRecognitionModel(config).to(memory_format=self.memory_format)  # type:ignore
        if config.compile_model:
            self.model: nn.Module = torch.compile(self.model)  # type: ignore
        self.loss_function = LOSSES[config.model_config.loss_name](**config.model_config.loss_parameters)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        return self._step(batch, batch_idx, mode="valid")

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, mode: str) -> torch.Tensor:
        _step_start_ts = time.time()

        images, labels = batch
        output = self.forward(images)
        loss = self.loss_function(output, labels)

        self.log(
            f"{mode}/loss",
            loss.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        _step_delta = time.time() - _step_start_ts
        self.log(
            f"{mode}/forward_time",
            _step_delta,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        x = x.to(memory_format=self.memory_format)
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = get_optimizer(config=self.config.optimisation_config, model=self.model)

        scheduler = get_scheduler(config=self.config.optimisation_config, optimizer=optimizer)
        scheduler_config = LRSchedulerConfig(
            scheduler=scheduler, interval=self.config.optimisation_config.scheduler_interval, frequency=1
        )
        return [optimizer], [scheduler_config]

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.epoch_end_time = time.time()
        t_delta = self.epoch_end_time - self.epoch_start_time
        self.log(
            "train/epoch_time_sec", t_delta, batch_size=1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )


class DeepIDLightningModule(FaceRecognitionLightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.model = DeepIDFaceRecognitionModel(config).to(memory_format=self.memory_format)  # type:ignore

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list[torch.Tensor]]:
        x = x.to(memory_format=self.memory_format)
        return self.model(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, mode: str) -> torch.Tensor:
        _step_start_ts = time.time()

        images, labels = batch
        outputs = self.forward(images)
        loss: torch.Tensor = torch.stack([self.loss_function(output, labels) for output in outputs]).sum()

        self.log(
            f"{mode}/loss",
            loss.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        _step_delta = time.time() - _step_start_ts
        self.log(
            f"{mode}/forward_time",
            _step_delta,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss
