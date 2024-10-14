import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implementation of DeepFace from:
DeepFace: Closing the Gap to Human-Level Performance in Face Verification, 
Taigman, Yaniv and Yang, Ming and Ranzato, Marc'Aurelio and Wolf, Lior, CVPR 2014
"""


class LocallyConnected2d(nn.Module):
    """Implement an idea of local Convolution kernels. Instead of moving kernels around whole image,
    LC layer has unique weights for each location"""

    def __init__(
        self,
        input_size: tuple[int, int],
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(LocallyConnected2d, self).__init__()

        # Save parameters
        self.input_size = input_size  # (height, width) of the input image
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Compute the size of the output feature map
        self.output_size = (
            (input_size[0] - kernel_size + 2 * padding) // stride + 1,
            (input_size[1] - kernel_size + 2 * padding) // stride + 1,
        )

        # Locally connected weights
        self.weights = nn.Parameter(
            torch.randn(
                self.output_size[0], self.output_size[1], output_channels, input_channels, kernel_size, kernel_size
            )
        )

        # Bias for each output location
        self.bias = nn.Parameter(torch.randn(self.output_size[0], self.output_size[1], output_channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_channels, input_height, input_width)
        batch_size = features.size(0)

        # Initialize output tensor
        output = torch.zeros(
            batch_size, self.output_channels, self.output_size[0], self.output_size[1], device=features.device
        )

        # Apply the locally connected operation for each location in the output
        for x_coordinate in range(self.output_size[0]):
            for y_coordinate in range(self.output_size[1]):
                # Extract the corresponding patch from the input
                x_patch = features[
                    :,
                    :,
                    x_coordinate * self.stride : x_coordinate * self.stride + self.kernel_size,
                    y_coordinate * self.stride : y_coordinate * self.stride + self.kernel_size,
                ]

                # Perform convolution for this patch using location-specific weights
                # Compute: output[:, :, x_coordinate, y_coordinate] = x_patch * weights[x_coordinate, y_coordinate] + bias[x_coordinate, y_coordinate]
                output[:, :, x_coordinate, y_coordinate] = (
                    torch.sum(x_patch.unsqueeze(1) * self.weights[x_coordinate, y_coordinate], dim=(2, 3, 4))
                    + self.bias[x_coordinate, y_coordinate]
                )

        return output


class DeepFace(nn.Module):
    """
    Full architecture implementation. The only change compared to original is input size.
    This implementation works with 112x112, original paper with 152x152.
    """

    def __init__(self, num_features: int = 4096):
        super(DeepFace, self).__init__()

        # C1: Convolution Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=0)  # 32 filters, 11x11 kernel size
        # M2: Max-Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # 3x3 pool size, stride of 2

        # C3: Convolution Layer
        self.conv2 = nn.Conv2d(32, 16, kernel_size=9, stride=1, padding=0)  # 16 filters, 9x9 kernel size

        # Locally connected layers L4, L5, L6 (16->64->128->256 channels)
        # L4: Locally Connected Layer
        self.local1 = LocallyConnected2d(
            input_size=(42, 42), input_channels=16, output_channels=64, kernel_size=5, stride=1, padding=0
        )
        # L5: Locally Connected Layer
        self.local2 = LocallyConnected2d(
            input_size=(38, 38), input_channels=64, output_channels=128, kernel_size=5, stride=1, padding=0
        )
        # L6: Locally Connected Layer
        self.local3 = LocallyConnected2d(
            input_size=(24, 24), input_channels=128, output_channels=256, kernel_size=3, stride=1, padding=0
        )

        # F7: Fully Connected Layer
        self.fc1 = nn.Linear(256 * 22 * 22, num_features)  # Adjusted for flattened size from locally connected layers

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv1(features))
        features = self.pool1(features)
        features = F.relu(self.conv2(features))

        features = self.local1(features)
        features = self.local2(features)
        features = self.local3(features)

        # Flatten the tensor for fully connected layers
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 256*22*22)

        # F7 -> Fully Connected Layer
        features = F.relu(self.fc1(features))
        return features
