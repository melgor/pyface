import os
import sys

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    train_file_path: str
    train_root_dir: str
    validation_file_path: str
    validation_root_dir: str
    lfw_file_path: str
    lfw_root_dir: str
    data_input_size: tuple[int, int] = 112, 112
    network_input_size: tuple[int, int] = 112, 112
    train_resize_for_crop: int = 128


@dataclass
class ModelConfig:
    backbone_name: str
    head_name: str
    loss_name: str
    embedding_size: int
    backbone_parameters: dict[str, Any] = field(default_factory=lambda: {})
    head_parameters: dict[str, Any] = field(default_factory=lambda: {})
    loss_parameters: dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class OptimisationConfig:
    learning_rate: float
    optimizer: str
    weigh_decay: float
    momentum: float
    early_stop_patience: int
    optimizer_config: dict[str, Any]
    scheduler: str
    scheduler_config: dict[str, Any]
    scheduler_interval: str = "epoch"
    grad_accumulation: int = 1


@dataclass
class TrainingConfig:
    experiment_name: str
    logging_dir: str
    batch_size: int
    num_epochs: int
    precision: str
    dataset_config: DatasetConfig
    model_config: ModelConfig
    optimisation_config: OptimisationConfig
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
    assert os.path.isfile(config.dataset_config.train_file_path) and os.path.isdir(
        config.dataset_config.train_root_dir
    ), "wrong train files"
    assert os.path.isfile(config.dataset_config.validation_file_path) and os.path.isdir(
        config.dataset_config.validation_root_dir
    ), "wrong validation files"
    assert os.path.isfile(config.dataset_config.lfw_file_path) and os.path.isdir(
        config.dataset_config.lfw_root_dir
    ), "wrong LFW files"


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
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config_omega = load_config(config_path)
    print(OmegaConf.to_yaml(config_omega))
