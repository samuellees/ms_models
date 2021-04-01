# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import One



class DoubleConv(nn.Cell):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        init_value_0 = One()
        init_value_1 = One()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.SequentialCell(
            [nn.Conv3d(in_channels, mid_channels, kernel_size=3, has_bias=False,
                       weight_init=init_value_0),
             nn.BatchNorm3d(mid_channels, momentum=0.1),
             nn.ReLU(),
             nn.Conv3d(mid_channels, out_channels, kernel_size=3, has_bias=False,
                       weight_init=init_value_1),
             nn.BatchNorm3d(out_channels),
             nn.ReLU()
             ]
        )

        # self.tmp = nn.BatchNorm3d(mid_channels, momentum=0.1)

    def construct(self, x):
        return self.double_conv(x)
        # x = self.double_conv(x)
        # x = self.tmp(x)
        # return x


class Down(nn.Cell):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.SequentialCell(
            [nn.MaxPool3d(kernel_size=2, stride=2),
             DoubleConv(in_channels, out_channels)]
        )

    def construct(self, x):
        return self.maxpool_conv(x)


class Up(nn.Cell):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Conv3dTranspose(in_channels, in_channels // 2, kernel_size=2, stride=2, weight_init=One())
        self.relu = nn.ReLU()
        self.concat = F.Concat(axis=1)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def construct(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.relu(x1)
        x = self.concat((x1, x2))
        return self.conv(x)


class OutConv(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        init_value = One()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init=init_value)

    def construct(self, x):
        x = self.conv(x)
        return x




class UNet(nn.Cell):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def construct(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits