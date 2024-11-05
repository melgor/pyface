import time

from typing import Optional, Union

import lightning as pl
import numpy as np
import torch
import torchmetrics

from loguru import logger
from sklearn.preprocessing import normalize
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .config import OptimisationConfig, TrainingConfig
from .datasets.lfw_evaluator import LFWEvaluator
from .evaulation.metrics import ModelFeatures, evaluate_identification_metric
from .face_models import DeepIDFaceRecognitionModel, FaceRecognitionModel
from .models.losses import LOSSES, ContrastiveLoss
from .models.losses.infonce import InfoNCELoss
from .optimizers import OPTIMIZERS, SCHEDULERS


def get_optimizer(config: OptimisationConfig, model: nn.Module) -> Optimizer:
    return OPTIMIZERS[config.optimizer_name](model.parameters(), **config.optimizer_config)


def get_scheduler(config: OptimisationConfig, optimizer: Optimizer) -> Optional[LRScheduler]:
    return SCHEDULERS[config.scheduler_name](optimizer, **config.scheduler_config)


class FaceRecognitionLightningModule(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.epoch_start_time: Optional[float] = None
        self.epoch_end_time: Optional[float] = None
        self.memory_format = torch.channels_last if self.config.channel_last else torch.contiguous_format
        self.model = self._get_model(config).to(memory_format=self.memory_format)  # type:ignore
        if config.compile_model:
            self.model: nn.Module = torch.compile(self.model)  # type: ignore

        self.loss_function = LOSSES[config.model_config.loss_name](**config.model_config.loss_parameters)
        self._validation_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}
        self._umd_query_set: int = 40000
        self._lfw_query_set: int = 1680
        self._lfw_evaluator = LFWEvaluator(
            config.dataset_config.lfw_file_path, config.dataset_config.lfw_pairs_path, method="l2"
        )
        self._accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=config.model_config.head_parameters["nb_classes"]
        )

    def _get_model(self, config: TrainingConfig) -> nn.Module:
        return FaceRecognitionModel(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]:
        x = x.to(memory_format=self.memory_format)
        return self.model(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=self.memory_format)
        features = self.model.forward_features(x)
        return features

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        _step_start_ts = time.time()
        images, labels = batch
        output, _ = self.forward(images)
        loss = self.loss_function(output, labels)

        # log step metric
        self._accuracy(output, labels)
        self.log("train/accuracy", self._accuracy)

        self.log(
            f"train/loss",
            loss.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        _step_delta = time.time() - _step_start_ts
        self.log(
            f"train/forward_time",
            _step_delta,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        output = self.forward_features(images)
        self._validation_data[dataloader_idx].append((output, labels))

    def configure_optimizers(self):
        optimizer = get_optimizer(config=self.config.optimisation_config, model=self.model)
        scheduler = get_scheduler(config=self.config.optimisation_config, optimizer=optimizer)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.config.optimisation_config.scheduler_interval,
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        self.epoch_end_time = time.time()
        t_delta = self.epoch_end_time - self.epoch_start_time
        self.log(
            "train/epoch_time_sec", t_delta, batch_size=1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )
        self.log("train/accuracy_epoch", self._accuracy)

    def on_validation_epoch_end(self):
        # validate retrieval
        y_embeddings, targets = zip(*self._validation_data[0])
        y_embeddings = torch.cat(y_embeddings, dim=0)
        targets = torch.cat(targets, dim=0)
        if y_embeddings.size(0) < 10000:
            return

        y_embeddings = torch.nn.functional.normalize(y_embeddings)
        y_embeddings_np = y_embeddings.float().cpu().numpy()
        acc = self._lfw_evaluator.evaluate(y_embeddings_np)
        self.log("valid/lfw", acc, on_step=False, on_epoch=True, sync_dist=True)

        # evaluate second dataset
        y_embeddings, targets = zip(*self._validation_data[1])
        y_embeddings = torch.cat(y_embeddings, dim=0)
        targets = torch.cat(targets, dim=0)
        y_embeddings = torch.nn.functional.normalize(y_embeddings)
        y_embeddings_np = y_embeddings.float().cpu().numpy()
        targets_np = targets.cpu().numpy()

        query_data = ModelFeatures(y_embeddings_np[: self._umd_query_set], targets_np[: self._umd_query_set])
        gallery_data = ModelFeatures(y_embeddings_np[self._umd_query_set :], targets_np[self._umd_query_set :])
        try:
            mAP = evaluate_identification_metric(query_data, gallery_data)
        except ValueError:
            mAP = 0.0
        self.log("valid/umd_mAP", mAP, on_step=False, on_epoch=True, sync_dist=True)

        # clear history of data
        self._validation_data = {0: [], 1: []}
        logger.info(f"LFW:{acc}  UMD:{mAP}")

        # targets = torch.cat(targets).cpu().numpy()
        #
        # # TODO: Normalize before centering?
        # if y_embeddings.shape[0] <= self.config.batch_size * 2:
        #     # sanity check run
        #     return
        #
        # y_embeddings = normalize(y_embeddings)
        # query_data = ModelFeatures(y_embeddings[: self._umd_query_set], targets[: self._umd_query_set])
        # gallery_data = ModelFeatures(y_embeddings[self._umd_query_set :], targets[self._umd_query_set :])
        #
        # try:
        #     mAP = evaluate_umd_faces(query_data, gallery_data)
        # except ValueError:
        #     mAP = 0.0
        # self.log("valid/umd_mAP", mAP, on_step=False, on_epoch=True, sync_dist=True)


class DeepIDLightningModule(FaceRecognitionLightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.contrastive_loss = InfoNCELoss()

    def _get_model(self, config: TrainingConfig) -> nn.Module:
        return DeepIDFaceRecognitionModel(config)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(memory_format=self.memory_format)
        features = self.model.forward_features(x)[0]
        return features

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        _step_start_ts = time.time()

        images, labels = batch
        output_classes, embeddings = self.forward(images)
        loss_classes: torch.Tensor = torch.stack(
            [self.loss_function(output, labels) for output in output_classes]
        ).sum()
        # loss_classes = torch.Tensor(0).cuda()
        loss_contrastive: torch.Tensor = torch.stack(
            [self.contrastive_loss(output, labels) for output in output_classes]
        ).sum()

        loss = loss_classes + 0.25 * loss_contrastive

        # log step metric
        self._accuracy(output_classes[0], labels)
        self.log("train/accuracy", self._accuracy)

        self.log(
            f"train/loss",
            loss.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            f"train/loss_classes",
            loss_classes.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"train/loss_contrastive",
            loss_contrastive.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        _step_delta = time.time() - _step_start_ts
        self.log(
            f"train/forward_time",
            _step_delta,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss


class CasiaNetLightningModule(FaceRecognitionLightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.contrastive_loss = InfoNCELoss()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        _step_start_ts = time.time()

        images, labels = batch
        output_classes, embeddings = self.forward(images)
        loss_classes: torch.Tensor = self.loss_function(output_classes, labels)
        loss_contrastive: torch.Tensor = self.contrastive_loss(embeddings, labels)

        loss = loss_classes + 0.25 * loss_contrastive

        # log step metric
        self._accuracy(output_classes, labels)
        self.log("train/accuracy", self._accuracy)
        self.log(
            f"train/loss",
            loss.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            f"train/loss_classes",
            loss_classes.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"train/loss_contrastive",
            loss_contrastive.item(),
            batch_size=images.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        _step_delta = time.time() - _step_start_ts
        self.log(
            f"train/forward_time",
            _step_delta,
            batch_size=1,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        return loss
