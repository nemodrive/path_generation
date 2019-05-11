import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, resnet34
from argparse import Namespace


def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = list()
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class BaseModel(nn.Module):
    def __init__(self, cfg: Namespace, **kwargs):
        super(BaseModel, self).__init__()

        pretrained_base = getattr(cfg, "pretrained_base", False)
        in_channels = getattr(cfg, "in_channels", 3)

        # Spatial transformer localization-network
        self.image_conv = resnet34(pretrained=pretrained_base)

        self.localization_2 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(10, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.layer = make_layer(3, BasicBlock, 6, 3)

    # Spatial transformer network forward function
    def stn(self, x1, x2):
        batch_size = x1.size(0)
        xs = torch.cat([x1, x2])

        xs = self.image_conv(xs)

        print("After Image conv", xs.shape)

        xs = torch.cat([xs[:batch_size], xs[batch_size:]], dim=1)

        xs = self.localization(xs)

        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x1.size())
        x = F.grid_sample(x1, grid)

        return x

    def forward(self, x1, x2):
        # transform the input
        x = self.stn(x1, x2)

        # Perform the usual forward pass
        x = self.layer(x)

        return x


if __name__ == "__main__":
    import argparse

    shape = torch.Size((4, 3, 128, 128))

    cfg = Namespace()

    img1 = torch.rand(shape)
    img2 = torch.rand(shape)

    net = BaseModel(cfg)
