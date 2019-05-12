import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, resnet34
from argparse import Namespace


class ProjectionNet(nn.Module):
    def __init__(self, cfg: Namespace, in_size=None, **kwargs):
        super(ProjectionNet, self).__init__()
        device = getattr(cfg, "device", None)

        hidden_size = getattr(cfg, "hidden_size", 512)
        assert in_size is not None, "In size of projection network is none"

        channels, w, h = in_size

        self.translation_net = nn.Sequential(
            nn.Conv2d(channels, hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 3, (w, h)),
        )

        self.rotation_net = nn.Sequential(
            nn.Conv2d(channels, hidden_size, 1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, 3, (w, h)),
        )

        groups = 6
        hidden_groups = (hidden_size * 2) - (hidden_size * 2) % groups
        self.intrinsic = nn.Sequential(
            nn.Conv2d(channels, hidden_groups, 1),
            nn.BatchNorm2d(hidden_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_groups, 6, (w, h), groups=6),
        )

        self.intrinsic_base = torch.zeros(1, 9, device=device)
        self.intrinsic_pid = torch.tensor([[0, 4, 2, 5]], device=device).long()  # Fx, Fy, x, y
        self.intrinsic_base[0, -1] = 1

    def forward(self, embedding):
        intrinsic_base = self.intrinsic_base
        intrinsic_pid = self.intrinsic_pid

        batchs = embedding.size(0)

        # Get translation
        tr = self.translation_net(embedding)

        # Get rotation
        rot = self.rotation_net(embedding)

        # Get camera Intrinsic
        param = self.intrinsic(embedding).view(batchs, -1)

        intrinsic_param = param[:, :4]  # Fx, Fy, x, y
        distort_coef = param[:, 4:]  # Quadratic si quartic coeff

        # Transform to intrinsic
        intrinsic_base = intrinsic_base.expand(batchs, 9)
        intrinsic_pid = intrinsic_pid.expand(batchs, intrinsic_pid.size(1))

        # Fx, Fy, x, y
        intrinsic_base = intrinsic_base.scatter_add(1, intrinsic_pid, intrinsic_param)
        intrinsic = intrinsic_base.view(-1, 3, 3)

        return tr, rot, intrinsic, distort_coef
