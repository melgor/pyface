import os

from typing import Optional

import lightning as pl

from config import TrainingConfig
from data import FaceDataModule
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


class FaceRecognitionTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.trainer: Optional[pl.Trainer] = None

    def train(self):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stop_callback = EarlyStopping(
            monitor=self.config.early_stop_metric,
            min_delta=self.config.early_stop_metric_min_delta,
            patience=self.config.optimisation_config.early_stop_patience,
            verbose=True,
            mode=self.config.early_stop_metric_mode,
        )
        weight_dir = os.path.join(self.config.logging_dir, "model", self.config.experiment_name, "weights")
        model_checkpoints = ModelCheckpoint(
            dirpath=weight_dir,
            monitor=self.config.model_checkpoint_metric,
            save_top_k=self.config.model_checkpoint_metric_save_top,
            mode=self.config.model_checkpoint_metric_mode,
            filename="epoch={epoch:02d}-={" + f"{self.config.model_checkpoint_metric}" + "}:.4f",
            save_weights_only=False,
            auto_insert_metric_name=False,  # auto-inserting cause creating a folder when metric has slash
            save_last=True,
        )

        tensorboard_logger = TensorBoardLogger(
            os.path.join(self.config.logging_dir, "output", "tensorboard"), self.config.experiment_name, version="tb"
        )
        loggers = [tensorboard_logger]

        self.trainer = pl.Trainer(
            max_epochs=self.config.num_epochs + 1,
            callbacks=[lr_monitor, early_stop_callback, model_checkpoints],
            logger=loggers,
            precision=self.config.precision,
            benchmark=True,
            deterministic=False,
            default_root_dir=self.config.logging_dir,
            log_every_n_steps=self.config.log_every_n_steps,
            accumulate_grad_batches=self.config.optimisation_config.grad_accumulation,
            gradient_clip_val=self.config.gradient_clip,
            devices=self.config.num_devices,
            val_check_interval=self.config.validation_check_interval,
        )

        data_model = FaceDataModule(config=self.config)

        model = "a"
        try:
            self.trainer.fit(model, data_model, ckpt_path=self.config.resume_path)
        finally:
            self.trainer.save_checkpoint(weight_dir + "last.pth")
            model_checkpoints.to_yaml()
