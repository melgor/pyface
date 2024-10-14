import os

from enum import Enum
from typing import Optional

import lightning as pl
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms

from config import TrainingConfig
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


class FaceRecognitionDataset(Dataset):
    def __init__(self, annotations_file: str, root_dir: str, transform: Optional[transforms.Compose]):
        self.img_labels = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        img_path = os.path.join(self.root_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx]["label"]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return self.img_labels.shape[0]


class DatasetType(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class FaceDataModule(pl.LightningDataModule):
    """
    Handler for datasets
    """

    def __init__(self, config: TrainingConfig, train_image_transformation: Optional[transforms.Compose] = None):
        super().__init__()
        self.config = config
        self.train_dataset: Optional[FaceRecognitionDataset] = None
        self.validation_dataset: Optional[FaceRecognitionDataset] = None

        self.validation_image_transformation = transforms.Compose(
            [transforms.Resize(*self.config.dataset_config.network_input_size)]
        )
        if train_image_transformation is not None:
            self.train_image_transformation = train_image_transformation
        else:
            data_height, data_width = self.config.dataset_config.data_input_size
            network_height, network_width = self.config.dataset_config.network_input_size
            self.train_image_transformation = transforms.Compose(
                [
                    transforms.Resize(
                        [
                            int(self.config.dataset_config.train_resize_for_crop * data_height / network_height),
                            int(self.config.dataset_config.train_resize_for_crop * data_width / network_width),
                        ]
                    ),
                    # TODO: No resize, add padding?
                    transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    transforms.RandomCrop(self.config.dataset_config.network_input_size),
                    transforms.RandomHorizontalFlip(),
                ]
            )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset, self.validation_dataset = self.create_dataset()
        else:
            raise NotImplementedError("Test phase data model is not implemented yet.")

    def create_dataset(self) -> tuple[FaceRecognitionDataset, FaceRecognitionDataset]:
        """
        Create training and validation datasets
        """
        train_transforms = self.get_image_transforms(DatasetType.TRAIN)
        validation_transforms = self.get_image_transforms(DatasetType.VALIDATION)

        train_dataset = FaceRecognitionDataset(
            self.config.dataset_config.train_file_path, self.config.dataset_config.train_root_dir, train_transforms
        )

        validation_dataset = FaceRecognitionDataset(
            self.config.dataset_config.validation_file_path,
            self.config.dataset_config.validation_root_dir,
            validation_transforms,
        )

        return train_dataset, validation_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def get_image_transforms(self, dataset_type: DatasetType) -> transforms.Compose:

        if dataset_type == DatasetType.TRAIN:
            image_transforms = self.train_image_transformation
        elif dataset_type == DatasetType.VALIDATION or dataset_type == DatasetType.TEST:
            image_transforms = self.validation_image_transformation
        else:
            raise NotImplementedError("Incorrect DatasetType")

        return image_transforms
