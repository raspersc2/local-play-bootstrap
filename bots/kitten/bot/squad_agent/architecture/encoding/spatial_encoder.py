from typing import Tuple

import numpy as np
from torch import Tensor, channels_last, nn


def layer_init(
    layer: nn.Conv2d,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
    init_bias: bool = True,
) -> nn.Conv2d:
    nn.init.orthogonal_(layer.weight, std)
    if init_bias:
        nn.init.constant_(layer.bias, bias_const)
    return layer


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        kernel_size: Tuple = (3, 3),
        stride: Tuple = (1, 1),
        padding: int = 1,
    ) -> None:
        super(ConvLayer, self).__init__()

        self.conv = layer_init(
            nn.Conv2d(
                in_planes,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # Optimization: If a nn.Conv2d layer is directly
                # followed by a nn.BatchNorm2d layer,
                # then the bias in the convolution is not needed
                bias=False,
            ),
            init_bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self, in_planes: int = 128, planes: int = 128, stride: tuple[int, int] = (1, 1)
    ) -> None:
        super(ResBlock, self).__init__()
        self.conv1: ConvLayer = ConvLayer(in_planes, planes, (3, 3), stride=stride)
        self.conv2: ConvLayer = ConvLayer(planes, planes, (3, 3), stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        del residual
        return x


class SpatialEncoder(nn.Module):
    def __init__(self, dropout_prob: float = 0.5):
        super(SpatialEncoder, self).__init__()

        self.ds_1 = ConvLayer(38, 64)
        self.ds_2 = ConvLayer(64, 128)
        self.ds_3 = ConvLayer(128, 128)

        self.res = nn.ModuleList()
        for i in range(3):
            self.res.append(ResBlock(128))

        # adding dropout forces the network to learn different
        # patterns / avoids over fitting
        # removed during evaluation
        self.dropout = nn.Dropout(p=dropout_prob)
        # flatten the output into a one dimensional tensor
        # which represents the encoding of our images
        self.fc = layer_init(nn.Linear(28800, 256))

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(memory_format=channels_last)
        x = self.ds_1(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.ds_2(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.ds_3(x)
        x = nn.functional.max_pool2d(x, 2, 2)

        for block in self.res:
            x = block(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
