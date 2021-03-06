import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, conv1x1, resnet34
from argparse import Namespace

import sys
sys.path.append("flownet2_pytorch")


from flownet2_pytorch.networks import FlowNetC


class FlowNet2C(FlowNetC.FlowNetC):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2C, self).__init__(args, batchNorm=batchNorm, div_flow=div_flow)
        self.rgb_max = args.rgb_max
        self.last_conv = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2, 1, 1),
            nn.ReLU(),
        )
        self.upsampled_ = nn.ConvTranspose2d(1, 1, 4, 4)

    def forward(self, x1, x2):
        inputs = torch.stack([x1, x2], dim=2)

        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(
            inputs.size()[:2] + (1, 1, 1,))

        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        # FlownetC top input stream
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)  # False
        out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv3a)

        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)

        # Merged conv layers
        out_conv3_1 = self.conv3_1(in_conv3_1)

        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))

        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)

        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)

        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)

        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)

        flow2 = self.predict_flow2(concat2)

        # Reduce to 1 channel
        flow2 = self.last_conv(flow2)

        # last_flow = self.upsample1(flow2)
        last_flow = self.upsampled_(flow2)

        return out_conv6, last_flow


if __name__ == "__main__":
    from argparse import Namespace
    cfg = Namespace()
    cfg.rgb_max = 255
    cfg.fp16 = False

    img = torch.rand(3, 3, 256, 256).cuda()

    net = FlowNet2C(cfg).cuda()

    out_conv6, last_flow = net(img, img)
    print(last_flow.size())
