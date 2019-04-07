# AndreiN, 2019

import os
import torch
import numpy as np
import shutil
import itertools
import glob
import re


def get_training_data_path(model_dir, best=False, index=None):
    if best:
        return os.path.join(model_dir, "training_data_best.pt")

    if index is not None:
        fld = os.path.join(model_dir, "checkpoints")
        if not os.path.isdir(fld):
            os.mkdir(fld)
        return os.path.join(fld, f"training_data_{index}.pt")

    return os.path.join(model_dir, "training_data.pt")


def get_last_training_path_idx(model_dir):
    if os.path.exists(model_dir):
        path = os.path.join(model_dir, "training_data_*.pt")

        max_index = 0
        for path in glob.glob(path):
            try:
                max_index = max(max_index,
                                int(re.findall("training_data_([1-9]\d*|0).pt", path)[0]))
            except:
                 pass

        return max_index
    return 0


class SaveData:
    def __init__(self, out_dir, save_best=True, save_all=False):
        self.out_dir = out_dir
        self.save_best = save_best
        self.save_all = save_all

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        self.best_loss = np.inf

        start_idx = get_last_training_path_idx(out_dir)
        self.index = itertools.count(start=start_idx, step=1)

    def load_training_data(self, out_dir=None, best=False):
        """ If best is set to false, the last training model is loaded """
        out_dir = out_dir if out_dir is not None else self.out_dir

        training_data = None
        if best:
            path = get_training_data_path(out_dir, best=best)
            if os.path.isfile(path):
                training_data = torch.load(path)

        if training_data is None:
            path = get_training_data_path(out_dir, best=False)
            try:
                training_data = torch.load(path)
            except OSError:
                training_data = None # dict({"loss": np.inf, "epoch": 0})

        if training_data is not None and "loss" in training_data:
            self.best_loss = training_data["loss"]

        return training_data

    def save_training_data(self, data, loss, other=None, model_dir=None):
        model_dir = model_dir if model_dir is not None else self.out_dir

        trainig_data = dict()
        trainig_data = data

        if other is not None:
            trainig_data.update(other)

        # Save standard
        path = get_training_data_path(model_dir)
        torch.save(trainig_data, path)

        if loss < self.best_loss:
            self.best_loss = loss
            best_path = get_training_data_path(model_dir, best=True)
            shutil.copyfile(path, best_path)

        if self.save_all:
            index_path = get_training_data_path(model_dir, index=next(self.index))
            shutil.copyfile(path, index_path)




