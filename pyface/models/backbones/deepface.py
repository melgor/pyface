import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class LocallyConnectedLayer(nn.Module):
    """
    Locally connected layer implementation using einsum for efficient computation.
    Each output position has its own set of weights without sharing.
    """

    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride=1, padding=0):
        super(LocallyConnectedLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Calculate output dimensions
        self.output_height = (self.input_size[0] + 2 * padding - kernel_size) // stride + 1
        self.output_width = (self.input_size[1] + 2 * padding - kernel_size) // stride + 1

        # Initialize weights with shape:
        # (output_height, output_width, out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            torch.randn(self.output_height, self.output_width, out_channels, in_channels, kernel_size, kernel_size)
            / (kernel_size * kernel_size * in_channels) ** 0.5
        )
        init.orthogonal_(self.weight.data)

        self.bias = nn.Parameter(torch.ones(out_channels, self.output_height, self.output_width) * 0.5)

    def forward(self, x):
        batch_size = x.size(0)

        # Add padding if necessary
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)

        # Extract patches efficiently using unfold
        patches = F.unfold(x, kernel_size=(self.kernel_size, self.kernel_size), stride=self.stride).view(
            batch_size, self.in_channels, self.kernel_size, self.kernel_size, self.output_height, self.output_width
        )

        # Use einsum for efficient computation
        # b: batch, i: input_channels, p,q: kernel dims, h,w: spatial dims
        # c: output_channels
        output = torch.einsum("bipqhw,hwcipq->bchw", patches, self.weight)

        return output + self.bias

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"input_size={self.input_size}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}"
        )


class DeepFace(nn.Module):
    """
    Full architecture implementation. The only change compared to original is input size.
    This implementation works with 112x112, original paper with 152x152.

    Parameters count for 152x152: 100M
    Parameters count for 112x112: 36M
    """

    def __init__(self, embedding_size: int = 4096, channels: int = 16):
        super(DeepFace, self).__init__()

        # C1: Convolution Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=0)  # 32 filters, 11x11 kernel size
        # M2: Max-Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # 3x3 pool size, stride of 2

        # C3: Convolution Layer
        self.conv2 = nn.Conv2d(32, channels, kernel_size=9, stride=1, padding=0)  # 16 filters, 9x9 kernel size

        # L4: Locally Connected Layer
        self.local1 = LocallyConnectedLayer(
            input_size=(42, 42), in_channels=16, out_channels=16, kernel_size=9, stride=1, padding=0
        )
        # L5: Locally Connected Layer
        self.local2 = LocallyConnectedLayer(
            input_size=(34, 34), in_channels=channels, out_channels=channels, kernel_size=7, stride=2, padding=0
        )
        # L6: Locally Connected Layer
        self.local3 = LocallyConnectedLayer(
            input_size=(14, 14), in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=0
        )

        # for full CNN concept, uncomment these lines
        # self.local1 = nn.Conv2d(channels, channels, kernel_size=9, stride=1, padding=0)
        # self.local2 = nn.Conv2d(channels, channels, kernel_size=7, stride=2, padding=0)
        # self.local3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0)

        # F7: Fully Connected Layer
        self.fc1 = nn.Linear(
            channels * 12 * 12, embedding_size
        )  # Adjusted for flattened size from locally connected layers

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv1(features))
        features = self.pool1(features)
        features = F.relu(self.conv2(features))
        features = F.relu(self.local1(features))
        features = F.relu(self.local2(features))
        features = F.relu(self.local3(features))

        # Flatten the tensor for fully connected layers
        features = features.reshape(features.size(0), -1)

        # F7 -> Fully Connected Layer
        features = self.fc1(features)
        return features


if __name__ == "__main__":
    from torchsummary import summary

    mm = DeepFace(512)
    summary(mm, (3, 112, 112), batch_size=2, device="cpu")
