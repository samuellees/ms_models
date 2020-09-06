# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore.ops import operations as P

class SqueezeNet(nn.Cell):

    def __init__(self, num_classes=1000, version='1.0'):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.version = version
        
        if self.version not in ['1.0', '1.1']:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=self.version))
        # extract features
        self.features = nn.SequentialCell(
            Conv2dBlock(3, 96, kernel_size=7, stride=2, pad_mode="valid"),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"), # without ceil_mode option
            FireBlock(96, 16, 64, 64), 
            FireBlock(128, 16, 64, 64),
            FireBlock(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
            # (32, 256, 26, 26)
            FireBlock(256, 32, 128, 128),
            FireBlock(256, 48, 192, 192),
            FireBlock(384, 48, 192, 192),
            FireBlock(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
            FireBlock(512, 64, 256, 256)
            # (32, 512, 12, 12)
        )
        # classifier
        self.classifier = nn.SequentialCell(
            nn.Dropout(keep_prob=0.5),
            Conv2dBlock(512, self.num_classes, kernel_size=1, weight_init=init.Normal(sigma=0.01)),
            # Conv2dBlock(512, self.num_classes, kernel_size=1)
        )
        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.mean(x, (2, 3))
        x = self.flatten(x)
        return x

class FireBlock(nn.Cell):
    
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
      super(FireBlock, self).__init__()
      self.squeeze = Conv2dBlock(in_channels=in_channels, out_channels=squeeze_channels, kernel_size=1)
      self.expand1x1 = Conv2dBlock(in_channels=squeeze_channels, out_channels=expand1x1_channels, kernel_size=1)
      self.expand3x3 = Conv2dBlock(in_channels=squeeze_channels, out_channels=expand3x3_channels, kernel_size=3, pad_mode='same')
      self.concat = P.Concat(axis=1)
             
    def construct(self, x):
      x = self.squeeze(x)
      expand1x1 = self.expand1x1(x)
      expand3x3 = self.expand3x3(x)
      return self.concat((expand1x1, expand3x3))

class Conv2dBlock(nn.Cell):
    """
     Basic convolutional block
     Args:
        in_channles (int): Input channel.
        out_channels (int): Output channel.
        kernel_size (int or tuple[int]): Input kernel size. Default: 1
        stride (int or tuple[int]): Stride size for the first convolutional layer. Default: 1.
        padding (int): Implicit paddings on both sides of the input. Default: 0.
        pad_mode (str): Padding mode. Optional values are "same", "valid", "pad". Default: "same".
        weight_init (str or Initializer): same with nn.Conv2d
      Returns:
          Tensor, output tensor.
    """
    
    # def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
    #             pad_mode="same", padding=0, weight_init="XavierUniform", 
    #             with_relu=True, 
    #             with_bn=False, 
    #             has_bias=True):
    #     super(Conv2dBlock, self).__init__()
    #     self.with_bn = with_bn
    #     self.with_relu = with_relu
    #     self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
    #                           kernel_size=kernel_size, stride=stride, pad_mode=pad_mode,
    #                           padding=padding, 
    #                           weight_init=weight_init, 
    #                           has_bias=has_bias, bias_init='zeros')
    #     if (with_bn):
    #         self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    #     if (with_relu):
    #         self.relu = nn.ReLU()
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                pad_mode="same", padding=0, weight_init="XavierUniform", 
                with_relu=True, 
                with_bn=True):
        super(Conv2dBlock, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, pad_mode=pad_mode,
                              padding=padding, 
                              weight_init=weight_init)
        if (with_bn):
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        if (with_relu):
            self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

