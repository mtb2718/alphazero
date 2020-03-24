import numpy as np
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super(ResidualBlock, self).__init__()
        self._c1 = nn.Conv2d(channels, channels, (3, 3), 1, 1, bias=False)
        self._bn1 = nn.BatchNorm2d(channels)
        self._c2 = nn.Conv2d(channels, channels, (3, 3), 1, 1, bias=False)
        self._bn2 = nn.BatchNorm2d(channels)
        self._relu = nn.ReLU(True)

    def forward(self, x):
        xi = x
        x = self._c1(x)
        x = self._bn1(x)
        x = self._relu(x)
        x = self._c2(x)
        x = self._bn2(x)
        x += xi
        x = self._relu(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, shape_in, shape_out):
        super(PolicyHead, self).__init__()
        self._cbr = nn.Sequential(
            nn.Conv2d(shape_in[0], 2, (1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(True)
        )
        self._fc = nn.Linear(2 * np.prod(shape_in[1:]), np.prod(shape_out))
        self._shape_out = shape_out

    def forward(self, x):
        x = self._cbr(x)
        x = x.view(x.shape[0], -1)
        x = self._fc(x)
        x = x.view(-1, *self._shape_out)
        return x


class ValueHead(nn.Module):
    def __init__(self, shape_in):
        super(ValueHead, self).__init__()
        self._cbr = nn.Sequential(
            nn.Conv2d(shape_in[0], 1, (1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )
        self._fc = nn.Sequential(
            nn.Linear(np.prod(shape_in[1:]), 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self._cbr(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self._fc(x)
        return x


class AlphaZero(nn.Module):
    def __init__(self, shape_in, shape_out, num_blocks=39, block_channels=256):
        super(AlphaZero, self).__init__()
        self._cbr = nn.Sequential(
            nn.Conv2d(shape_in[0], block_channels, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(block_channels),
            nn.ReLU(True),
        )
        self._tower = nn.Sequential(*([ResidualBlock(channels=block_channels)] * num_blocks))
        self._policy_head = PolicyHead((block_channels, *shape_in[1:]), shape_out)
        self._value_head = ValueHead((block_channels, *shape_in[1:]))

    def forward(self, x, p_valid):
        x = self._cbr(x)
        x = self._tower(x)
        p = self._policy_head(x)
        v = self._value_head(x)
        return p, v


class UniformModel(nn.Module):
    def forward(self, x, p_valid):
        B = p_valid.shape[0]
        p = torch.ones_like(p_valid, dtype=torch.float32)
        v = torch.zeros(B, 1, dtype=torch.float32, device=x.device)
        return p, v
