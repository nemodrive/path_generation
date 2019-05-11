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
import os
from argparse import Namespace
from liftoff.config import read_config

from utils import dataloader
from utils.utils import convert_image_np
from models import get_model

plt.ion()   # interactive mode


def train(epoch, train_loader, model, optimizer, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.rand((64, 1, 224, 224))
        target = torch.rand((64, 1, 224, 224))
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


def test(test_loader, model, device):
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


def visualize_stn(test_loader, model, device):
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


def run(cfg: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO change train loader
    train_loader = dataloader.CustomDatasetFromImages('data/dataset2.csv')

    model = get_model(cfg.model).to(device)

    _optimizer = getattr(torch.optim, cfg.train.algorithm)
    optim_args = vars(cfg.train.algorithm_args)
    optimizer = _optimizer(model.parameters(), **optim_args)

    for epoch in range(1, 20 + 1):
        train(epoch, train_loader, model, optimizer, device)
        # test(test_loader, model, device)

    # Visualize the STN transformation on some input batch
    # visualize_stn(test_loader, model, device)

    plt.ioff()
    plt.show()


def main():
    # Reading args
    args = read_config()  # type: Args
    args.out_dir = "results"

    if not hasattr(args, "out_dir"):
        from time import time
        if not os.path.isdir('./results'):
            os.mkdir('./results')
        out_dir = f'./results/{str(int(time())):s}_{args.experiment:s}'
        os.mkdir(out_dir)
        args.out_dir = out_dir
    else:
        assert os.path.isdir(args.out_dir), "Given directory does not exist"

    if not hasattr(args, "run_id"):
        args.run_id = 0

    run(args)


if __name__ == "__main__":
    main()
