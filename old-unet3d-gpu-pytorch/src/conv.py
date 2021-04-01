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

class MyConv3D(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 pad=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 has_bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, \
                               padding=pad, stride=stride, dilation=dilation, \
                               groups=groups, bias=has_bias)
        nn.init.constant_(self.conv.bias, 0)

    def construct(self, x):
        output = self.conv(x)
        return output

class MyConv3DTranspose(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 pad=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 output_padding=0,
                 has_bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels=in_channel, out_channels=out_channel,\
                                                  kernel_size=kernel_size, padding=pad, stride=stride, \
                                                  dilation=dilation, groups=groups, output_padding=output_padding, bias=has_bias)
        nn.init.constant_(self.conv_transpose.bias, 0)

    def construct(self, x):
        output = self.conv_transpose(x)
        return output
