import torch
import torch.nn as nn

from torch.nn import init
from torchsummary import summary


class CasiaNet(nn.Module):
    def __init__(self, embedding_size: int):
        super(CasiaNet, self).__init__()

        # Conv11, Conv12 -> 3x3 Convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool1
        )

        # Conv21, Conv22 -> 3x3 Convolution layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool2
        )

        # Conv31, Conv32 -> 3x3 Convolution layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool3
        )

        # Conv41, Conv42 -> 3x3 Convolution layers
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pool4
        )

        # Conv51, Conv52 -> 3x3 Convolution layers
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 160, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, embedding_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # Pool5
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Pass through each layer sequentially
        features = self.conv1(images)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.conv5(features)

        # Flatten the output from the convolutional layers
        features = features.view(features.size(0), -1)
        return features


# Example usage
if __name__ == "__main__":
    model = CasiaNet(embedding_size=320)
    summary(model, (3, 100, 100), device="cpu")
