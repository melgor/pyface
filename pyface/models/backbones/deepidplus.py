import torch
import torch.nn as nn
import torch.nn.functional as F


class LocallyConnectedLayer(nn.Module):
    """Custom locally connected layer without weight sharing."""

    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        super(LocallyConnectedLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Compute the output dimensions
        self.output_height = (input_size[0] - kernel_size + 2 * padding) // stride + 1
        self.output_width = (input_size[1] - kernel_size + 2 * padding) // stride + 1

        # Each position has a unique filter
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, self.output_height, self.output_width, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.randn(out_channels, self.output_height, self.output_width))

    def forward(self, x):
        batch_size = x.size(0)
        output = torch.zeros(batch_size, self.out_channels, self.output_height, self.output_width).to(x.device)

        # Apply the locally connected filters manually
        for i in range(self.output_height):
            for j in range(self.output_width):
                region = x[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ]
                output[:, :, i, j] = torch.einsum("bcih,coih->bo", region, self.weight[:, :, i, j]) + self.bias[:, i, j]

        return output


class DeepID2Plus(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super(DeepID2Plus, self).__init__()

        # First two convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=1), nn.MaxPool2d(kernel_size=2), nn.ReLU()  # Output: (20, 52, 44)
        )
        self.liner1 = nn.Linear(128 * 22 * 26, embedding_size)

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1),  # Output: (20, 52, 44)
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.liner2 = nn.Linear(128 * 10 * 12, embedding_size)
        # Third layer with local weight sharing in 2x2 regions
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, groups=16),  # Output: (20, 52, 44)
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.liner3 = nn.Linear(128 * 5 * 4, embedding_size)
        # Fourth layer is fully locally connected
        self.locally_connected4 = nn.Sequential(
            LocallyConnectedLayer(128, 128, (5, 4), kernel_size=2, stride=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            128 * 4 * 3 + 128 * 5 * 4, embedding_size
        )  # Adjust based on output size from locally connected layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        # Convolutional + pooling layers
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        features_1 = self.liner1(x1.reshape(batch_size, -1))
        features_2 = self.liner2(x2.reshape(batch_size, -1))
        features_3 = self.liner3(x3.reshape(batch_size, -1))
        # Apply the locally connected layer
        x4 = self.locally_connected4(x3)  # Output: (80, 4, 3)

        # Flatten the feature map
        x4_flat = x4.view(batch_size, -1)
        x3_flat = x3.reshape(batch_size, -1)
        x_merged = torch.cat([x3_flat, x4_flat], dim=1)

        # Fully connected layers
        features_final = F.relu(self.fc1(x_merged))
        return features_final, features_1, features_2, features_3
