import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
from argparse import Namespace

from utils import dataloader, config
from utils.utils import convert_image_np
from models import get_model
from sequence_datasets.sequence_folders import SequenceFolder
import custom_transforms

plt.ion()   # interactive mode


def train(epoch, train_loader, model, optimizer, device):
    model.train()
    for batch_idx, x in enumerate(train_loader):
        data = x[1][0]
        target = x[0]

        # data = data.cpu().numpy()
        # target = target.cpu().numpy()
        #
        # data = data[0]
        # target = target[0]
        #
        # data = np.transpose(data, (1, 2, 0))
        # target = np.transpose(target, (1, 2, 0))
        #
        # a = np.zeros((100, data.shape[1], 3))
        #
        # data = np.append(data, a, axis=0)
        # target = np.append(target, a, axis=0)
        #
        # print(target.shape)
        #
        # cv2.imshow('target', target)
        # cv2.waitKey(0)
        # cv2.imshow('data', data)
        # cv2.waitKey(0)

        # a = torch.rand(size=(4, 3, 1001-128, data.shape[3]))
        #
        # data = torch.cat((data, a), 2)
        # target = torch.cat((target, a), 2)

        # a = torch.rand(size=(4, 3, data.shape[2], 1280 - data.shape[3]))

        # data = torch.cat((data, a), 3)
        # target = torch.cat((target, a), 3)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data, target)

        loss = model.calculate_loss((data, target), None, output)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def visualize_results(loader, model, device):
    model.eval()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        model_out = model(data, target)

        projected_img, valid_points, mask_prediction = model_out

        projected_img = projected_img[0].detach().cpu().numpy()
        projected_img = np.transpose(projected_img, (1, 2, 0))

        cv2.imshow('projected_img', projected_img)
        cv2.waitKey(0)

        valid_points = valid_points[0].detach().cpu().numpy()
        valid_points = np.transpose(valid_points, (1, 2, 0))

        cv2.imshow('valid_points', valid_points)
        cv2.waitKey(0)

        mask_prediction = mask_prediction[0].detach().cpu().numpy()
        mask_prediction = np.transpose(mask_prediction, (1, 2, 0))

        cv2.imshow('mask_prediction', mask_prediction)
        cv2.waitKey(0)

        break


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
        transformed_input_tensor = model(data).cpu()

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
    dataset_csv = cfg.dataset_csv

    torch.autograd.set_detect_anomaly(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    config.add_to_cfg(cfg, subgroups=["model", "train"], new_arg='device', new_arg_value=device)
    config.add_to_cfg(cfg.model, subgroups=[], new_arg='device', new_arg_value=device)

    # TODO change train loader
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor()
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    train_set = SequenceFolder(
        cfg.data,
        transform=train_transform,
        seed=cfg.seed,
        train=True,
        sequence_length=cfg.sequence_length
    )

    val_set = SequenceFolder(
        cfg.data,
        transform=valid_transform,
        seed=cfg.seed,
        train=False,
        sequence_length=cfg.sequence_length
    )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    model = get_model(cfg.model).to(device)

    _optimizer = getattr(torch.optim, cfg.train.algorithm)
    optim_args = vars(cfg.train.algorithm_args)
    optimizer = _optimizer(model.parameters(), **optim_args)

    for epoch in range(1000):
        train(epoch, train_loader, model, optimizer, device)
        # test(test_loader, model, device)

    # Visualize the STN transformation on some input batch
    # visualize_stn(train_loader, model, device)
    visualize_results(train_loader, model, device)

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
    from liftoff import parse_opts
    run(parse_opts())
