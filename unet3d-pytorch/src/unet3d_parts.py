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


import time

class ResidualUnit(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3), down=True, is_output=False):
        super().__init__()
        self.stride = stride
        self.down = down
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv_1 = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), \
                                     stride=self.stride, padding=1, bias=False)
        nn.init.normal_(self.down_conv_1.weight)
        self.is_output = is_output
        if not is_output:
            self.batchNormal1 = nn.BatchNorm3d(num_features=self.out_channel, momentum=0.9)
            self.relu1 = nn.PReLU()
            nn.init.constant_(self.relu1.weight, 0.25)
        if self.down:
            self.down_conv_2 = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), \
                                         stride=1, padding=1, bias=False)
            self.relu2 = nn.PReLU()
            if kernel_size[0] == 1:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), \
                                          stride=self.stride, bias=False)
            else:
                self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), \
                                       stride=self.stride, padding=1, bias=False)
            self.batchNormal2 = nn.BatchNorm3d(num_features=self.out_channel, momentum=0.9)
            nn.init.constant_(self.relu2.weight, 0.25)
            nn.init.normal_(self.down_conv_2.weight)
            nn.init.normal_(self.residual.weight)


    def forward(self, x):
        torch.cuda.synchronize(0)
        begin = time.time()
        out = self.down_conv_1(x)
        torch.cuda.synchronize(0)
        print("conv1: ", time.time() - begin)
        if self.is_output:
            return out
        begin = time.time()
        out = self.batchNormal1(out)
        torch.cuda.synchronize(0)
        print("batchNormal1: ", time.time() - begin)
        begin = time.time()
        out = self.relu1(out)
        torch.cuda.synchronize(0)
        print("relu1: ", time.time() - begin)
        if self.down:
            begin = time.time()
            out = self.down_conv_2(out)
            torch.cuda.synchronize(0)
            print("conv2: ", time.time() - begin)

            begin = time.time()
            out = self.batchNormal2(out)
            torch.cuda.synchronize(0)
            print("batchNormal2: ", time.time() - begin)

            begin = time.time()
            out = self.relu2(out)
            torch.cuda.synchronize(0)
            print("relu2: ", time.time() - begin)

            begin = time.time()
            res = self.residual(x)
            torch.cuda.synchronize(0)
            print("conv3: ", time.time() - begin)
        else:
            res = x
        return out + res

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=(3, 3, 3)):
        super().__init__()
        self.stride = stride
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.down_conv = ResidualUnit(self.in_channel, self.out_channel, stride, kernel_size)

    def forward(self, x):
        x = self.down_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channel, down_in_channel, out_channel, stride=2, is_output=False):
        super().__init__()
        self.in_channel = in_channel
        self.down_in_channel = down_in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.conv3d_transpose = nn.ConvTranspose3d(in_channels=self.in_channel + self.down_in_channel, \
                                                   out_channels=self.out_channel, kernel_size=(3, 3, 3), \
                                                   stride=self.stride, \
                                                   output_padding=(1, 1, 1), padding=1, bias=False)

        self.conv = ResidualUnit(self.out_channel, self.out_channel, stride=1, down=False, \
                                 is_output=is_output)
        self.batchNormal1 = nn.BatchNorm3d(num_features=self.out_channel, momentum=0.9)
        self.relu = nn.PReLU()
        nn.init.constant_(self.relu.weight, 0.25)
        nn.init.normal_(self.conv3d_transpose.weight)

    def forward(self, input_data, down_input):
        begin = time.time()
        x = torch.cat((input_data, down_input), 1)
        torch.cuda.synchronize(0)
        print("cat: ", time.time() - begin)
        torch.cuda.synchronize(0)
        begin = time.time()
        x = self.conv3d_transpose(x)
        torch.cuda.synchronize(0)
        print("conv3d_transpose: ", time.time() - begin)
        torch.cuda.synchronize(0)
        begin = time.time()
        x = self.batchNormal1(x)
        torch.cuda.synchronize(0)
        print("batchNormal1: ", time.time() - begin)
        torch.cuda.synchronize(0)
        begin = time.time()
        x = self.relu(x)
        torch.cuda.synchronize(0)
        print("relu: ", time.time() - begin)
        torch.cuda.synchronize(0)
        begin = time.time()
        x = self.conv(x)
        torch.cuda.synchronize(0)
        print("conv4: ", time.time() - begin)
        return x
