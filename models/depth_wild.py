import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, resnet34
from argparse import Namespace



class DepthWild(nn.Module):
    def __init__(self, cfg: Namespace, **kwargs):
        super(DepthWild, self).__init__()

        pretrained_base = getattr(cfg, "pretrained_base", False)
        in_channels = getattr(cfg, "in_channels", 3)
        depth_network_cfg = getattr(cfg, "depth_network", None)
        flow_network_cfg = getattr(cfg, "flow_net", None)

        self.depth_network = get_model(depth_network_cfg, n_channels=in_channels, n_classes=1)
        self.flow_network = get_model(flow_network_cfg, n_channels=in_channels, n_classes=1)

    def forward(self, x1, x2):

        depth = self.depth_network(x1)
        depth = F.softplus(depth)

        return None
