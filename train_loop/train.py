from train_loop.train_base import TrainBase
import torch.nn.functional as F
import torch
import torchvision
import cv2
import numpy as np

from utils.utils import convert_image_np_cv


class TrainDefault(TrainBase):
    def __init__(self, cfg, train_loader, test_loader, model, optimizer, device, saver, logger):
        super().__init__(cfg, train_loader, test_loader, model, optimizer, device, saver, logger)
        self.view_freq = 0
        # self.header.update({"loss2": ("μ", None, "{}")})

    def _train(self):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        train_loader = self.train_loader
        log_freq = self.batch_log_freq
        epoch = self.epoch
        logger = self.logger
        log = self.get_base_log()
        loss_f = torch.nn.MSELoss()

        for self.batch_idx, (img1, img2) in enumerate(train_loader):
            batch_idx = self.batch_idx

            img1, img2 = img1.to(device), img2.to(device)

            optimizer.zero_grad()
            predict, target = model(img1, img2)

            loss = loss_f(target, img1)

            loss.backward()
            optimizer.step()

            # -- Update log
            # TODO caution, might be slow to do update each step
            self.std_update_log(log)  # Standard update log
            log["loss"].append(loss.item())

            if (batch_idx + 1) % log_freq == 0:

                logger.write(log)

                log = self.get_base_log(reset=True)

        # ------------------------------------------------------------------------------------------
        # Small view demo

        if self.view_freq != 0 and epoch % self.view_freq    == 0:
            cv2.imshow("menu", np.zeros((100, 100), dtype=np.uint8))
            key = cv2.waitKey(1)
            if key != -1:
                cv2.waitKey(0)

            self.show_tensor_img(img1, show="img1")
            self.show_tensor_img(predict, show="Predict")
            self.show_tensor_img(target, show="Target")
        # ------------------------------------------------------------------------------------------

        mean_loss = 0
        info = {}
        return mean_loss, info

    def show_tensor_img(self, batch, show=None):
        in_grid = convert_image_np_cv(
            torchvision.utils.make_grid(batch.detach().cpu()))
        if show is not None:
            cv2.imshow(show, in_grid)
            return cv2.waitKey(1)
        return None

    def _eval(self):
        pass

    def _save(self):
        raise NotImplemented

    def _load(self):
        raise NotImplemented
