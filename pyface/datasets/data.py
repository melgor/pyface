import math
import os
import random

from collections import defaultdict
from enum import Enum
from typing import Optional, Tuple

import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms

from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.io import read_image

from pyface.config import TrainingConfig


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, target_vector: torch.Tensor, batch_size: int) -> None:
        super().__init__(None)
        self._target_vector = target_vector
        self._batch_size = batch_size

        self._class_index = self._build_class_index()
        self._classes = list(self._class_index.keys())
        self._classes_probability = self.calculate_probability_per_class()
        self._num_classes = len(self._classes)

        self._elem_per_class = 2
        self._classes_per_class = int(self.batch_size / self._elem_per_class)
        self._iter_number = 0
        self._iters_in_epoch = self.calculate_iters_per_epoch()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def calculate_iters_per_epoch(self) -> int:
        nb_valid_examples = sum(len(self._class_index[class_nb]) for class_nb in self._classes)
        return math.floor(nb_valid_examples / self._batch_size)

    def calculate_probability_per_class(self) -> list[float]:
        example_per_class = [len(self._class_index[class_nb]) for class_nb in self._classes]
        nb_valid_examples = sum(example_per_class)
        prob_per_class: list[float] = [float(prob) / nb_valid_examples for prob in example_per_class]
        return prob_per_class

    def __iter__(self):
        self._iter_number = 0
        return self

    def __next__(self):
        if self._iter_number < self._iters_in_epoch:
            self._iter_number += 1
            selected_labels = np.random.choice(self._classes, self._classes_per_class, p=self._classes_probability)
            element_indices = []
            for class_idx in selected_labels:
                # random element from the class
                random_elements = random.choices(self._class_index[class_idx], k=self._elem_per_class)
                element_indices.extend(random_elements)

            return element_indices
        raise StopIteration

    def _build_class_index(self):
        class_index: dict[int, list[int]] = defaultdict(list)
        for idx, elem in enumerate(self._target_vector):
            class_index[elem.item()].append(idx)
        return class_index

    def __len__(self) -> int:
        return self._iters_in_epoch


class FaceRecognitionDataset(Dataset):
    def __init__(self, annotations_file: str, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.img_labels = pd.read_csv(annotations_file)
        assert "img_path" in self.img_labels.columns, "No img_path column in data"
        assert "label" in self.img_labels.columns, "No label column in data"
        self.root_dir = root_dir
        self.transform = transform
        logger.info(f"Nb of classes: {self.img_labels.label.max()+1} {annotations_file}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.img_labels.iloc[idx]
        img_path = os.path.join(self.root_dir, row["img_path"])
        image = read_image(img_path)
        label = row["label"]
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

    normalize = transforms.Normalize(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[1, 1, 1],
        # std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )

    def __init__(self, config: TrainingConfig, train_image_transformation: Optional[transforms.Compose] = None):
        super().__init__()
        self._config = config
        self._train_dataset: Optional[FaceRecognitionDataset] = None
        self._validation_dataset: Optional[FaceRecognitionDataset] = None
        self._lfw_dataset: Optional[FaceRecognitionDataset] = None

        self._validation_image_transformation = transforms.Compose(
            [
                transforms.Resize(self._config.dataset_config.network_input_size),
                transforms.ToDtype(torch.float),
                self.normalize,
            ]
        )
        if train_image_transformation is not None:
            self._train_image_transformation = train_image_transformation
        else:
            self._train_image_transformation = transforms.Compose(
                [
                    transforms.Resize(self._config.dataset_config.network_input_size),
                    # TODO: No resize, add padding?
                    # transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    # transforms.RandomCrop(self._config.dataset_config.network_input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToDtype(torch.float),
                    self.normalize,
                ]
            )

    def setup(self, stage: str):
        if stage == "fit":
            logger.info("Prepare datasets")
            self._train_dataset, self._lfw_dataset, self._validation_dataset = self.create_dataset()
        else:
            raise NotImplementedError("Test phase data model is not implemented yet.")

    def create_dataset(self) -> tuple[FaceRecognitionDataset, FaceRecognitionDataset, FaceRecognitionDataset]:
        """
        Create training and validation datasets
        """
        train_transforms = self._get_image_transforms(DatasetType.TRAIN)
        validation_transforms = self._get_image_transforms(DatasetType.VALIDATION)

        train_dataset = FaceRecognitionDataset(
            self._config.dataset_config.train_file_path, self._config.dataset_config.train_root_dir, train_transforms
        )

        lfw_dataset = FaceRecognitionDataset(
            self._config.dataset_config.lfw_file_path,
            self._config.dataset_config.lfw_root_dir,
            validation_transforms,
        )

        validation_dataset = FaceRecognitionDataset(
            self._config.dataset_config.validation_file_path,
            self._config.dataset_config.validation_root_dir,
            validation_transforms,
        )

        return train_dataset, lfw_dataset, validation_dataset

    def train_dataloader(self) -> DataLoader[FaceRecognitionDataset]:
        # return DataLoader(
        #     self._train_dataset, batch_size=self._config.batch_size, shuffle=True, num_workers=self._config.num_workers
        # )
        train_targets = torch.tensor(self._train_dataset.img_labels["label"].tolist())
        return DataLoader(
            self._train_dataset,
            batch_sampler=ClassBalancedBatchSampler(train_targets, self._config.batch_size),
            num_workers=self._config.num_workers,
            # batch_size=self._config.batch_size,
            # shuffle=True,
        )

    def val_dataloader(self) -> list[DataLoader[FaceRecognitionDataset]]:
        return [
            DataLoader(
                self._lfw_dataset,
                batch_size=self._config.batch_size,
                num_workers=self._config.num_workers,
                shuffle=False,
            ),
            DataLoader(
                self._validation_dataset,
                batch_size=self._config.batch_size,
                num_workers=self._config.num_workers,
                shuffle=False,
            ),
        ]

    def _get_image_transforms(self, dataset_type: DatasetType) -> transforms.Compose:
        if dataset_type == DatasetType.TRAIN:
            image_transforms = self._train_image_transformation
        elif dataset_type == DatasetType.VALIDATION or dataset_type == DatasetType.TEST:
            image_transforms = self._validation_image_transformation
        else:
            raise NotImplementedError("Incorrect DatasetType")

        return image_transforms
