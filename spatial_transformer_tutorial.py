# -*- coding: utf-8 -*-
"""
Spatial Transformer Networks Tutorial
=====================================
**Author**: `Ghassen HAMROUNI <https://github.com/GHamrouni>`_

.. figure:: /_static/img/stn/FSeq.png

In this tutorial, you will learn how to augment your network using
a visual attention mechanism called spatial transformer
networks. You can read more about the spatial transformer
networks in the `DeepMind paper <https://arxiv.org/abs/1506.02025>`__

Spatial transformer networks are a generalization of differentiable
attention to any spatial transformation. Spatial transformer networks
(STN for short) allow a neural network to learn how to perform spatial
transformations on the input image in order to enhance the geometric
invariance of the model.
For example, it can crop a region of interest, scale and correct
the orientation of an image. It can be a useful mechanism because CNNs
are not invariant to rotation and scale and more general affine
transformations.

One of the best things about STN is the ability to simply plug it into
any existing CNN with very little modification.
"""
# License: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.resnet import BasicBlock, conv1x1
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch_geometric
import cv2

plt.ion()   # interactive mode

######################################################################

######################################################################
# Loading the data
# ----------------
#
# In this post we experiment with the classic MNIST dataset. Using a
# standard convolutional network augmented with a spatial transformer
# network.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform=None, fps_sample=3):
        """
        Args:
            time_diff : (seconds) Dime distance between frames
            csv_path (string): path to csv file
            fps_sample (fps): Max Sample from video
            transform: pytorch transforms for transforms and tensor conversion
        """

        self._time_diff = 1 # Dime distance between frames

        # Transforms
        self.transforms = transform

        # Read the csv file
        self.data_info = data_info = pd.read_csv(csv_path)

        diff_pts = data_info.pts.values[1:] - data_info.pts.values[:-1]
        diff_pts = diff_pts[diff_pts > 0]

        median_diff, mean_diff, std_diff = np.median(diff_pts), diff_pts.mean(), diff_pts.std()

        print(f"Video FPS -  median: {1./median_diff}, mean: {1./mean_diff}, std: {1./std_diff}")
        print(f"____WILL CONSIDER FIX FPS for frame offset calc_____ {1./median_diff}")

        self.dist_frames = median_diff

        # Third column is for pts viedo -> get groups based on fps sample
        bins = np.arange(0, data_info.pts.max(), 1. / fps_sample)
        data_info["interval"] = data_info.pts.apply(pd.cut, bins=bins, include_lowest=True)\
            .values.codes

        # Calculate len
        self.data_len = len(self.data_info.index)

        # Img pair file paths
        self.first_idx = None
        self.second_idx = None

        self.set_groups()

    def set_groups(self):
        df = self.data_info
        dist_frames = self.dist_frames
        time_diff = self.time_diff

        sample_start_idx = df.groupby(["video", "interval"]).apply(lambda x: x.sample())

        first_idx = sample_start_idx.index.levels[2]
        idx_dist = max(1, int(time_diff / dist_frames))
        second_idx = first_idx + idx_dist

        # Filter idx that does not exist
        has_df = second_idx < len(df)
        first_idx = first_idx[has_df]
        second_idx= second_idx[has_df]

        # filter same video
        same_video = df.loc[first_idx, "video"].values == df.loc[second_idx, "video"].values
        first_idx = first_idx[same_video]
        second_idx = second_idx[same_video]

        self.first_idx = np.asarray(df.loc[first_idx, "img"])
        self.second_idx = np.asarray(df.loc[second_idx, "img"])

        self.data_len = len(first_idx)

    def __getitem__(self, index):
        # Open image
        img_first = cv2.imread(self.first_idx[index])
        img_second = cv2.imread(self.second_idx[index])

        if self.transforms is not None:
            # Transform image to tensor
            img_first = self.transforms(img_first)
            img_second = self.transforms(img_second)

        return (img_first, img_second)

    def __len__(self):
        return self.data_len

    @property
    def time_diff(self):
        return self._time_diff

    @time_diff.setter
    def time_diff(self, value):
        self._time_diff = value

# Training dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=64, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=64, shuffle=True, num_workers=4)

######################################################################

from torch.utils.data.dataset import Dataset


######################################################################
# Depicting spatial transformer networks
# --------------------------------------
#
# Spatial transformer networks boils down to three main components :
#
# -  The localization network is a regular CNN which regresses the
#    transformation parameters. The transformation is never learned
#    explicitly from this dataset, instead the network learns automatically
#    the spatial transformations that enhances the global accuracy.
# -  The grid generator generates a grid of coordinates in the input
#    image corresponding to each pixel from the output image.
# -  The sampler uses the parameters of the transformation and applies
#    it to the input image.
#
# .. figure:: /_static/img/stn/stn-arch.png
#
# .. Note::
#    We need the latest version of PyTorch that contains
#    affine_grid and grid_sample modules.
#


def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, planes * block.expansion, stride),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization_2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
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
        xs1 = self.localization1(x1)
        xs2 = self.localization2(x2)

        xs = torch.cat([xs1, xs2])
        xs = self.localization(xs)

        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x1.size())
        x = F.grid_sample(x1, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.layer(x)

        return x


model = Net().to(device)

######################################################################
# Training the model
# ------------------
#
# Now, let's use the SGD algorithm to train the model. The network is
# learning the classification task in a supervised way. In the same time
# the model is learning STN automatically in an end-to-end fashion.


optimizer = optim.SGD(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
#
# A simple test procedure to measure STN the performances on MNIST.
#


def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

######################################################################
# Visualizing the STN results
# ---------------------------
#
# Now, we will inspect the results of our learned visual attention
# mechanism.
#
# We define a small helper function in order to visualize the
# transformations while training.


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

for epoch in range(1, 20 + 1):
    train(epoch)
    test()

# Visualize the STN transformation on some input batch
visualize_stn()

plt.ioff()
plt.show()
