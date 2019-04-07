import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1
from models.resnetconvonly import resnet34, resnet18
import torchvision
import matplotlib.pyplot as plt

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
        hidden_size = getattr(cfg, "hidden_size", 512)

        # Spatial transformer localization-network
        self.image_conv = resnet34(pretrained=pretrained_base)
        # TODO Out size seems small 8x8. We removed Maxpool from resnet

        feature_size = 8
        image_conv_channels = 512
        hidden_channels = 512

        self.localization = nn.Sequential(
            nn.Conv2d(image_conv_channels * 2, hidden_channels, kernel_size=3),
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3),
            nn.ReLU(True),
        )

        out_localization_size = 4096
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(out_localization_size, hidden_size*2),
            nn.ReLU(True),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.layer1 = make_layer(3, BasicBlock, 128, 3)
        layer1_out = 128
        self.layer2 = nn.Conv2d(layer1_out, 3, kernel_size=1)

    # Spatial transformer network forward function
    def stn(self, x1, x2):
        batch_size = x1.size(0)
        xs = torch.cat([x1, x2])

        xs = self.image_conv(xs)

        # print("After Image conv", xs.shape)

        xs = torch.cat([xs[:batch_size], xs[batch_size:]], dim=1)

        xs = self.localization(xs)
        # print("AFTER localization", xs.shape)

        xs = xs.view(batch_size, -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x2.size())
        x = F.grid_sample(x2, grid)  # TODO Does it backprop through x1 here

        return x

    def forward(self, x1, x2):
        # transform the input
        target = self.stn(x1, x2)

        # Perform the usual forward pass
        predict = self.layer1(target)
        predict = self.layer2(predict)

        return predict, target


def train_small(image1, image2):
    import argparse
    import torch.optim as optim
    import cv2
    from utils.utils import convert_image_np_cv
    import numpy as np

    w, h = 128, 128
    shape = torch.Size((4, 3, w, h))
    cfg = Namespace()
    device = "cuda"
    num_iter = 100000
    log_freq = 10

    def read_image(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (w, h))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img.astype(np.float)/255 - mean) / std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        img = img.to(device)
        return img

    img1 = read_image(image1)
    img2 = read_image(image2)

    t1 = convert_image_np_cv(img1.cpu().squeeze(0))
    t2 = convert_image_np_cv(img2.cpu().squeeze(0))

    # Plot the results side-by-side
    cv2.imshow("T1", t1)
    cv2.imshow("T2", t2)
    cv2.waitKey(1)

    model = BaseModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_f = torch.nn.MSELoss()

    for i in range(num_iter):
        predict, target = model(img1, img2)

        loss = loss_f(target, img1)
        # loss = (target - img1).abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

        if (i+1) % log_freq == 0:
            # TODO visualize
            in_grid = convert_image_np_cv(
                torchvision.utils.make_grid(predict.detach().cpu()))
            # in_grid = convert_image_np_cv(img1.cpu().squeeze(0))

            out_grid = convert_image_np_cv(
                torchvision.utils.make_grid(target.detach().cpu()))

            # Plot the results side-by-side
            cv2.imshow("predict", in_grid)
            cv2.imshow("target", out_grid)
            cv2.waitKey(1)


if __name__ == "__main__":
    train_small('/home/nemodrive0/workspace/andreim/upb/vlcsnap-2019-03-16-11h25m29s182.png', '/home/nemodrive0/workspace/andreim/upb/vlcsnap-2019-03-16-11h26m07s032.png')
    # import argparse
    #
    # shape = torch.Size((4, 3, 128, 128))
    #
    # cfg = Namespace()
    #
    # img1 = torch.rand(shape)
    # img2 = torch.rand(shape)
    #
    # net = BaseModel(cfg)
    #
    # x = net(img1, img2)
