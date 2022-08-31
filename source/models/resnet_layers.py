"""
ResNet block based on https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
Reference:
https://arxiv.org/pdf/1512.03385.pdf
"""
import math
import warnings
from typing import List

import torch
from torch import nn


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Downsampler, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.downsample(x)
        return out

    def freeze(self):
        self.downsample[0].weight.requires_grad = False
        self.downsample[0].bias.requires_grad = False
        self.downsample[1].weight.requires_grad = False
        self.downsample[1].bias.requires_grad = False


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride: int = 1, downsample=None,
                 with_film: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2)
        nn.init.xavier_normal_(self.conv1.weight, gain=math.sqrt(2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        nn.init.xavier_normal_(self.conv2.weight, gain=math.sqrt(2))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.with_film = with_film

    def forward(self, x, alpha=None, beta=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.with_film:
            alpha = torch.reshape(alpha, (out.shape[0], out.shape[1]))
            alpha = alpha[:, :, None, None].expand(-1, -1, out.shape[2], out.shape[3])
            beta = torch.reshape(beta, (out.shape[0], out.shape[1]))
            beta = beta[:, :, None, None].expand(-1, -1, out.shape[2], out.shape[3])
            out = alpha*out + beta
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out

    def freeze(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.bn1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.bn2.weight.requires_grad = False
        self.bn2.bias.requires_grad = False
        if self.downsample is not None:
            self.downsample.freeze()


class ResNet(nn.Module):
    def __init__(self, num_params, end_with_fcl: bool = True, num_channels: int = 128,
                 with_film: bool = False):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, num_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=False)
        )
        nn.init.xavier_normal_(self.conv1[0].weight, gain=math.sqrt(2))
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.block1_1 = ResNetBlock(num_channels, num_channels, 3, with_film=with_film)
        self.block1_2 = ResNetBlock(num_channels, num_channels, 3, with_film=with_film)
        self.block2_1 = ResNetBlock(num_channels, 2 * num_channels, 3, 2,
                                    Downsampler(num_channels, 2 * num_channels, 2),
                                    with_film=with_film)
        self.block2_2 = ResNetBlock(2 * num_channels, 2 * num_channels, 3,
                                    with_film=with_film)
        self.block3_1 = ResNetBlock(2 * num_channels, 4 * num_channels, 3, 2,
                                    Downsampler(2 * num_channels, 4 * num_channels, 2),
                                    with_film=with_film)
        self.block3_2 = ResNetBlock(4 * num_channels, 4 * num_channels, 3,
                                    with_film=with_film)
        self.avgpool = nn.AvgPool2d(8)
        self.end_with_fcl = end_with_fcl
        # if end_with_fcl:
        #    self.fcl = nn.Linear(256, num_params)
        #    nn.init.xavier_normal_(self.fcl.weight, gain=math.sqrt(2))
        self.with_film = with_film

    def forward(self, x, alphas: List[torch.Tensor], betas: List[torch.Tensor]):
        # (..., 1, 513, 137)
        out = self.conv1(x)
        # (..., 128, 257, 69)
        out = self.maxpool(out)
        # (..., 128, 129, 35)
        if not self.with_film:
            # alphas = [None] * 6
            # betas = [None] * 6
            alphas = [torch.tensor([])] * 6
            betas = [torch.tensor([])] * 6
        out = self.block1_1(out, alpha=alphas[0], beta=betas[0])
        # (..., 128, 129, 35)
        out = self.block1_2(out, alpha=alphas[1], beta=betas[1])
        # (..., 128, 129, 35)
        out = self.block2_1(out, alpha=alphas[2], beta=betas[2])
        # (..., 256, 65, 18)
        out = self.block2_2(out, alpha=alphas[3], beta=betas[3])
        # (..., 256, 65, 18)
        out = self.block3_1(out, alpha=alphas[4], beta=betas[4])
        # (..., 512, 33, 9)
        out = self.block3_2(out, alpha=alphas[5], beta=betas[5])
        # (..., 512, 33, 9)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        # if self.end_with_fcl:
        #    out = self.fcl(out)
        return out

    def freeze(self, block: int):
        match block:
            case 0:
                self.conv1[0].weight.requires_grad = False
                self.conv1[1].weight.requires_grad = False
                self.conv1[1].bias.requires_grad = False
            case 1:
                self.block1_1.freeze()
                self.block1_2.freeze()
            case 2:
                self.block2_1.freeze()
                self.block2_2.freeze()
            case 3:
                self.block3_1.freeze()
                self.block3_2.freeze()
            case _:
                warnings.warn("I did not understand what you want me to freeze.", UserWarning)
