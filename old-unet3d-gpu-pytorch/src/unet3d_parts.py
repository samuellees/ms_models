# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.conv import MyConv3D as Conv3D, MyConv3DTranspose as Conv3DTranspose

class ResidualUnit(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), down=True, is_output=False):
        super().__init__()
        self.stride = stride
        self.down = down
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv_1 = Conv3D(in_channel, out_channel, kernel_size=(3, 3, 3), stride=self.stride, pad=1)
        self.is_output = is_output
        if not is_output:
            self.batchNormal1 = nn.BatchNorm3d(num_features=self.out_channel)
            self.relu1 = nn.PReLU()
        if self.down:
            self.down_conv_2 = Conv3D(out_channel, out_channel, kernel_size=(3, 3, 3), stride=1, pad=1)
            self.relu2 = nn.PReLU()
            if kernel_size[0] == 1:
                self.residual = Conv3D(in_channel, out_channel, kernel_size=(1, 1, 1), stride=self.stride)
            else:
                self.residual = Conv3D(in_channel, out_channel, kernel_size=(3, 3, 3), stride=self.stride, pad=1)
            self.batchNormal2 = nn.BatchNorm3d(num_features=self.out_channel)


    def construct(self, x):
        out = self.down_conv_1(x)
        if self.is_output:
            return out
        out = self.batchNormal1(out)
        out = self.relu1(out)
        if self.down:
            out = self.down_conv_2(out)
            out = self.batchNormal2(out)
            out = self.relu2(out)
            res = self.residual(x)
        else:
            res = x
        return out + res

class Down(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3)):
        super().__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv = ResidualUnit(self.in_channel, self.out_channel, stride, kernel_size)

    def construct(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Cell):
    def __init__(self, in_channel, down_in_channel, out_channel, stride=2, is_output=False):
        super().__init__()
        self.in_channel = in_channel
        self.down_in_channel = down_in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv3d_transpose = Conv3DTranspose(in_channel=self.in_channel + self.down_in_channel, \
                                                pad=1, out_channel=self.out_channel, kernel_size=(3, 3, 3), \
                                                stride=self.stride, output_padding=(1, 1, 1))

        self.conv = ResidualUnit(self.out_channel, self.out_channel, stride=1, down=False, \
                                 is_output=is_output)
        self.batchNormal1 = nn.BatchNorm3d(num_features=self.out_channel)
        self.relu = nn.PReLU()

    def construct(self, input_data, down_input):
        x = torch.cat((input_data, down_input), 1)
        x = self.conv3d_transpose(x)
        x = self.batchNormal1(x)
        x = self.relu(x)
        x = self.conv(x)
        return x
