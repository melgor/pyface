import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyface.datasets.data import FaceRecognitionDataset
from pyface.datasets.lfw_evaluator import LFWEvaluator
from pyface.models.backbones import CasiaNet
from pyface.models.heads import ClassificationLayer

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train_file_path = "/home/blcv/Projects/pyface-data/CASIA-maxpy-clean-aligned-112/casia_webface.csv"
    train_root_dir = "/home/blcv/Projects/pyface-data/CASIA-maxpy-clean-aligned-112/"
    lfw_file_path = "/home/blcv/Projects/pyface-data/LFW/lfw_filenames.csv"
    lfw_root_dir = "/home/blcv/Projects/pyface-data/LFW/lfw_align_112"
    lfw_pairs_path = "/home/blcv/Projects/pyface-data/LFW/pairs.txt"

    normalize = transforms.Normalize(
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
    )

    train_image_transformation = transforms.Compose(
        [transforms.Resize((112, 112)), transforms.RandomHorizontalFlip(), transforms.ToDtype(torch.float), normalize]
    )

    valid_image_transformation = transforms.Compose(
        [transforms.Resize((112, 112)), transforms.ToDtype(torch.float), normalize]
    )

    train_dataset = FaceRecognitionDataset(train_file_path, train_root_dir, train_image_transformation)
    valid_dataset = FaceRecognitionDataset(lfw_file_path, lfw_root_dir, valid_image_transformation)
    lfw_evaluator = LFWEvaluator(lfw_file_path, lfw_pairs_path, method="l2")

    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=8, shuffle=False)

    model: nn.Module = torch.compile(CasiaNet(320))  # type: ignore
    classifier = ClassificationLayer(320, 10575, 0.0)
    optimizer = SGD(
        [
            {"params": model.parameters()},
            {"params": classifier.parameters()},
        ],
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=5, cooldown=1, min_lr=0.0001, threshold=0.1)

    model.cuda()
    classifier.cuda()
    criterion.cuda()

    for epoch in range(100):

        mean_loss = []
        for i, (input, target) in enumerate(tqdm(train_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            with torch.amp.autocast("cuda"):
                features = model(input)
                output = classifier(features)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())

        list_features = []
        with torch.no_grad():
            for i, (input, target) in enumerate(valid_loader):
                input = input.cuda()
                with torch.amp.autocast("cuda"):
                    features = model(input)
                    features = torch.nn.functional.normalize(features)
                list_features.append(features.cpu().data.numpy())
            features = np.vstack(list_features)

        acc = lfw_evaluator.evaluate(features)
        print(f"Loss: {np.mean(mean_loss)} LFW:{acc}")
        scheduler.step(acc)
