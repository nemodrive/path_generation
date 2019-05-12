import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, resnet34
from argparse import Namespace
from . import get_model
from utils.image_utils import reverse_warp


"""
Not implemented: 
    @ Unsupervised Learning of Depth and Ego-Motion from Video
    ------
    1/(α∗sigmoid(x)+β) with α = 10 and β = 0.1 to constrain the 
    predicted depth to be always positive within a reasonable range
    ----
    
"""


class DepthWild(nn.Module):
    def __init__(self, cfg: Namespace, **kwargs):
        super(DepthWild, self).__init__()

        pretrained_base = getattr(cfg, "pretrained_base", False)
        in_channels = getattr(cfg, "in_channels", 3)
        depth_network_cfg = getattr(cfg, "depth_net", None)
        motion_network_cfg = getattr(cfg, "motion_net", None)
        projection_network_cfg = getattr(cfg, "projection_net", None)
        input_size = getattr(cfg, "input_size", None)
        device = getattr(cfg, "device", None)

        # Loss coefficients
        self.loss_cfg = loss_cfg = getattr(cfg, "loss_cfg", Namespace())
        loss_cfg.l1_coeff = loss_cfg.l1_coeff if hasattr(loss_cfg, "l1_coeff") else 1.
        loss_cfg.mask_coeff = loss_cfg.l1_coeff if hasattr(loss_cfg, "mask_coeff") else 0.

        img_test = torch.rand(1, in_channels, *input_size, device=device)

        #  -- Network 1
        # Depth network's input is the source image in [B, C, H, W]
        self.depth_network = get_model(depth_network_cfg, n_channels=in_channels, n_classes=1)
        self.depth_network = self.depth_network.to(device)

        # -- Network 2
        # Motion network`s input must be 2 images (source, target)
        # 2 X [B, C, H, W]
        # -> Returns (embedding, mask)
        self.motion_network = get_model(motion_network_cfg)
        self.motion_network = self.motion_network.to(device)

        # Get motion out size
        embedding, mask = self.motion_network(img_test, img_test)
        motion_out_size = embedding.size()[1:]

        # -- Network 2
        # Pose network get's the input of motion network and outputs:
        # Translation, Rotation and Intrinsic camera parameters
        self.projection_net = get_model(projection_network_cfg, in_size=motion_out_size)

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        depth = self.depth_network(x1)
        depth = F.softplus(depth)
        depth = depth.squeeze(1)

        motion_embedding, mask_prediction = self.motion_network(x1, x2)

        tr, rot, intrinsic, distort_coef = self.projection_net(motion_embedding)

        pose = torch.cat([tr, rot], dim=1).view(batch_size, -1)

        projected_img, valid_points = reverse_warp(x2, depth, pose, intrinsic, distort_coef)

        return projected_img, valid_points, mask_prediction

    def calculate_loss(self, data, target, model_out):
        x1, x2 = data
        projected_img, valid_points, mask_prediction = model_out

        loss_cfg = self.loss_cfg

        diff = (x1 - projected_img).abs().mean(1).unsqueeze(1)

        # diff_wm = diff * valid_points.float() #* mask_prediction
        diff_wm = diff * mask_prediction

        loss_diff = diff_wm.mean()

        # Cross entropy for mask
        mask_prediction.zero_()
        target = torch.ones_like(mask_prediction)
        loss_reg_mask = - nn.BCELoss()(mask_prediction, target)

        loss = loss_cfg.l1_coeff * loss_diff + loss_reg_mask

        return loss






