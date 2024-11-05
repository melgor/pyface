import torch
import torch.nn as nn
import torch.nn.init as init

"""
Network proposed in Multi-task Deep Neural Network for Joint Face Recognition and Facial Atribute Prediction

As stated at paper, after each conv and linear layer, batch norm is presented
"""


class FullBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(FullBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
        )

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out += residual

        return out


class FastBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(FastBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.PReLU(planes),
        )

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        out += residual

        return out


# Creates count residual blocks with specified number of features
def layer(inplanes, count, stride, block):
    if count < 1:
        return nn.Sequential()

    layers = list()
    for i in range(count):
        layers.append(block(inplanes, inplanes, stride))

    return nn.Sequential(*layers)


# CONV-RELU-MAXPOOL with increase fetures in conv
def increaseFeatures(n_in: int, n_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(n_out),
        nn.PReLU(n_out),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )


class FudanResNet(nn.Module):

    def __init__(self, embedding_size: int, blockFull: bool = False):
        self.inplanes = 64
        super(FudanResNet, self).__init__()

        block: type[nn.Module] = FastBlock
        if blockFull:
            block = FullBlock

        definition, n_feature = [3, 4, 6, 3], 512 * 5 * 5

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0, bias=False), nn.BatchNorm2d(32), nn.PReLU(32)
        )

        self.layer1 = increaseFeatures(32, 64, 1)
        self.layer2 = layer(64, definition[0], 1, block)
        self.layer3 = increaseFeatures(64, 128, 1)
        self.layer4 = layer(128, definition[1], 1, block)
        self.layer5 = increaseFeatures(128, 256, 1)
        self.layer6 = layer(256, definition[2], 1, block)
        self.layer7 = increaseFeatures(256, 512, 1)
        self.layer8 = layer(512, definition[3], 1, block)
        self.linear = nn.Linear(n_feature, embedding_size, bias=False)
        self.fc1_bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1_bn(self.linear(x))

        return x


if __name__ == "__main__":
    m = FudanResNet(512, blockFull=True).float()
    data = torch.Tensor(16, 3, 112, 112).float()
    print(m(data).size())
