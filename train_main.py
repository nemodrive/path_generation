import torch
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from argparse import Namespace
from liftoff.config import read_config
import os
from typing import List

from utils import dataloader
from models import get_model

from utils.logger import MultiLogger
from utils.save_training import SaveData
from train_loop.train_base import TrainBase
from train_loop import get_train

plt.ion()   # interactive mode


MAIN_CFG_ARGS = ["train", "model"]


def add_to_cfg(cfg: Namespace, subgroups: List[str], new_arg: str, new_arg_value) -> None:
    for arg in subgroups:
        if hasattr(cfg, arg):
            setattr(getattr(cfg, arg), new_arg, new_arg_value)


def run(cfg: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = cfg.dataset_csv
    batch_size = cfg.batch_size
    shuffle = cfg.shuffle
    num_workers = cfg.num_workers
    epochs = cfg.epochs
    test_freq = cfg.test_freq

    out_dir = getattr(cfg, "out_dir")
    add_to_cfg(cfg, MAIN_CFG_ARGS, "out_dir", out_dir)
    print(out_dir)

    # ----------------------------------------------------------------------------------------------
    # Data loaders
    img_size = cfg.input_size

    # TODO change train loader
    train_transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
        dataloader.CustomDatasetFromImages(dataset, img_size, transform=train_transformer),
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = None

    # ----------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------
    # Load model

    model = get_model(cfg.model).to(device)

    # ----------------------------------------------------------------------------------------------
    # Load optim

    _optimizer = getattr(torch.optim, cfg.train.algorithm)
    optim_args = vars(cfg.train.algorithm_args)
    optimizer = _optimizer(model.parameters(), **optim_args)

    # ----------------------------------------------------------------------------------------------
    # Load logger

    logger = MultiLogger(out_dir, cfg.tb)

    # Log command and all script arguments

    logger.info("{}\n".format(cfg))

    # ----------------------------------------------------------------------------------------------
    # Load training status
    saver = SaveData(out_dir, save_best=cfg.save_best, save_all=cfg.save_all)

    checkpoint, crt_epoch = None, 1
    try:
        # Continue from last point
        checkpoint = saver.load_training_data(best=False)
        logger.info("Training data exists & loaded successfully\n")
    except OSError:
        logger.info("Could not load training data\n")

    # ----------------------------------------------------------------------------------------------
    # Load trainer
    trainer = get_train(cfg.train, train_loader, test_loader,
                        model, optimizer, device, saver, logger)  # type: TrainBase
    if checkpoint is not None:
        trainer.load(checkpoint)
        crt_epoch = checkpoint["epoch"]

    # ----------------------------------------------------------------------------------------------

    for epoch in range(crt_epoch, epochs + 1):
        trainer.train()
        if test_freq and epoch % test_freq == 0:
            trainer.eval()

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
