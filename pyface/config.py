import os
import sys

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from omegaconf import OmegaConf

from .models.backbones import BACKBONES
from .models.heads import HEADS
from .models.losses import LOSSES
from .optimizers import OPTIMIZERS, SCHEDULERS


@dataclass
class DatasetConfig:
    train_file_path: Optional[str] = None
    train_root_dir: Optional[str] = None
    validation_file_path: Optional[str] = None
    validation_root_dir: Optional[str] = None
    lfw_file_path: Optional[str] = None
    lfw_root_dir: Optional[str] = None
    data_input_size: tuple[int, int] = 112, 112
    network_input_size: tuple[int, int] = 112, 112
    train_resize_for_crop: int = 128


@dataclass
class ModelConfig:
    backbone_name: str = "DeepFace"
    head_name: str = "ClassificationLayer"
    loss_name: str = "CrossEntropy"
    embedding_size: int = 4096
    backbone_parameters: dict[str, Any] = field(default_factory=lambda: {})
    head_parameters: dict[str, Any] = field(default_factory=lambda: {"input_features": 4096, "nb_classes": 100})
    loss_parameters: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class OptimisationConfig:
    learning_rate: float = 0.001
    optimizer_name: str = "AdamW"
    weigh_decay: float = 1e-4
    momentum: float = 0.9
    early_stop_patience: int = 5
    optimizer_config: dict[str, Any] = field(default_factory=lambda: {})
    scheduler_name: str = "StepLR"
    scheduler_config: dict[str, Any] = field(default_factory=lambda: {})
    scheduler_interval: str = "epoch"
    grad_accumulation: int = 1


@dataclass
class TrainingConfig:
    experiment_name: Optional[str] = None
    logging_dir: Optional[str] = None
    batch_size: int = 512
    num_epochs: int = 20
    precision: str = "bf16-mixed"
    dataset_config: DatasetConfig = DatasetConfig()
    model_config: ModelConfig = ModelConfig()
    optimisation_config: OptimisationConfig = OptimisationConfig()
    channel_last: bool = True
    resume_path: Optional[str] = None
    log_every_n_steps: int = 10
    gradient_clip: float = 5.0
    validation_check_interval: float = 0.3
    num_devices: int = 1
    seed: int = 42
    pin_memory: bool = True
    num_workers: int = 8
    early_stop_metric: str = "valid/metric_iou_none_epoch"
    early_stop_metric_min_delta: float = 0.01
    early_stop_metric_mode: str = "max"
    model_checkpoint_metric: str = "valid/metric_iou_none_epoch"
    model_checkpoint_metric_mode: str = "max"
    model_checkpoint_metric_save_top: int = 2


def validate_config_and_init_paths(config: TrainingConfig):
    """Add root location to images and labels location"""
    print(config.dataset_config.train_file_path)

    assert config.dataset_config.train_file_path and config.dataset_config.train_root_dir, "setup train files"
    assert os.path.isfile(config.dataset_config.train_file_path) and os.path.isdir(
        config.dataset_config.train_root_dir
    ), "wrong train files"
    assert (
        config.dataset_config.validation_file_path and config.dataset_config.validation_root_dir
    ), "setup validation files"
    assert os.path.isfile(config.dataset_config.validation_file_path) and os.path.isdir(
        config.dataset_config.validation_root_dir
    ), "wrong validation files"
    assert config.dataset_config.lfw_file_path and config.dataset_config.lfw_root_dir, "setup LFW files"
    assert os.path.isfile(config.dataset_config.lfw_file_path) and os.path.isdir(
        config.dataset_config.lfw_root_dir
    ), "wrong LFW files"

    assert config.experiment_name is not None, "Set experimentation name"
    assert config.logging_dir is not None, "Set logging dir"

    assert (
        config.model_config.backbone_name in BACKBONES.keys()
    ), f"{config.model_config.backbone_name} not supported. Supported: {BACKBONES.keys()}"
    assert (
        config.model_config.head_name in HEADS.keys()
    ), f"{config.model_config.head_name} not supported. Supported: {HEADS.keys()}"
    assert (
        config.model_config.loss_name in LOSSES.keys()
    ), f"{config.model_config.loss_name} not supported. Supported: {LOSSES.keys()}"
    assert (
        config.optimisation_config.optimizer_name in OPTIMIZERS.keys()
    ), f"{config.optimisation_config.optimizer_name} not supported. Supported: {OPTIMIZERS.keys()}"
    assert (
        config.optimisation_config.scheduler_name in SCHEDULERS.keys()
    ), f"{config.optimisation_config.scheduler_name} not supported. Supported: {SCHEDULERS.keys()}"

    today = datetime.now()
    config.experiment_name = f"{config.experiment_name}_{today.strftime('%Y_%m_%d_%h_%H_%M_%s')}"


def load_config(config_path: Optional[str], load_args: bool = False) -> TrainingConfig:
    # Load the configuration file and merge it with the default configuration
    default_config = TrainingConfig()  # type: ignore
    default_config_omega: TrainingConfig = OmegaConf.structured(default_config)

    if config_path is not None:
        user_config = OmegaConf.load(config_path)
        all_configs = [default_config_omega, user_config]
        if load_args:
            # overwrite any argument by command line. To make it work, remove path to base config file from sys.argv
            sys.argv.pop(1)
            cli_config = OmegaConf.from_cli()
            all_configs.append(cli_config)

        default_config_omega = OmegaConf.merge(*all_configs)  # type: ignore
        validate_config_and_init_paths(default_config_omega)
    return default_config_omega


if __name__ == "__main__":
    config_path = None
    print(len(sys.argv))
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config_omega = load_config(config_path)
    print(OmegaConf.to_yaml(config_omega))
