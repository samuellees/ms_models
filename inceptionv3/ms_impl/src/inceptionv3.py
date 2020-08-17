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
"""Inception v3."""
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal, XavierUniform
from mindspore.ops import operations as P

class Inceptionv3(nn.Cell):
    """
    Inception v3
    """

    def __init__(self, num_classes=1000, create_aux_logits=False):
        super(Inceptionv3, self).__init__()
        self.create_aux_logits = create_aux_logits
        # N x 3 x 299 x 299
        self.Conv2d_1a_3x3 = Conv2dBlock(in_channels=3, out_channels=32, kernel_size=3, stride=2, pad_mode="valid")
        # N x 32 x 149 x 149
        self.Conv2d_2a_3x3 = Conv2dBlock(in_channels=32, out_channels=32, kernel_size=3, pad_mode="valid")
        # N x 32 x 147 x 147
        self.Conv2d_2b_3x3 = Conv2dBlock(in_channels=32, out_channels=64, kernel_size=3, pad_mode="same")
        # N x 64 x 147 x 147
        self.MaxPool_3a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        # N x 64 x 73 x 73
        self.Conv2d_3b_1x1 = Conv2dBlock(in_channels=64, out_channels=80, kernel_size=1)
        # N x 80 x 73 x 73
        self.Conv2d_4a_3x3 = Conv2dBlock(in_channels=80, out_channels=192, kernel_size=3, pad_mode="valid")
        # N x 192 x 71 x 71
        self.MaxPool_5a_3x3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        # N x 192 x 35 x 35
        self.Mixed_5b = InceptionBlockA(in_channels=192, var_channels=32)
        # N x 256 x 35 x 35
        self.Mixed_5c = InceptionBlockA(in_channels=256, var_channels=64)
        # N x 288 x 35 x 35
        self.Mixed_5d = InceptionBlockA(in_channels=288, var_channels=64)
        # N x 288 x 35 x 35
        self.Mixed_6a = InceptionBlockB_1(in_channels=288)
        # N x 768 x 17 x 17
        self.Mixed_6b = InceptionBlockB_2(in_channels=768, var_channels=128)
        # N x 768 x 17 x 17
        self.Mixed_6c = InceptionBlockB_2(in_channels=768, var_channels=160)
        # N x 768 x 17 x 17
        self.Mixed_6d = InceptionBlockB_2(in_channels=768, var_channels=160)
        # N x 768 x 17 x 17
        self.Mixed_6e = InceptionBlockB_2(in_channels=768, var_channels=192)
        # N x 768 x 17 x 17
        if create_aux_logits:
            self.AuxLogits = InceptionBlockAux(in_channels=768, num_classes=num_classes)
        # N x 768 x 17 x 17
        self.Mixed_7a = InceptionBlockC_1(in_channels=768)
        # N x 1280 x 8 x 8
        self.Mixed_7b = InceptionBlockC_2(in_channels=1280)
        # N x 2048 x 8 x 8
        self.Mixed_7c = InceptionBlockC_2(in_channels=2048)
        # N x 2048 x 8 x 8
        self.mean = P.ReduceMean(keep_dims=True)
        # N x 2048 x 1 x 1
        self.Dropout_last = nn.Dropout(keep_prob=0.8)
        # N x 2048 x 1 x 1
        self.Conv2d_last = Conv2dBlock(in_channels=2048, out_channels=num_classes, 
                                kernel_size=1, with_relu=False, with_bn=False)
        # N x num_classes x 1 x 1
        self.fc = nn.Dense(in_channels=2048, out_channels=num_classes)
        self.flatten = nn.Flatten()
        # N x num_classes


    def construct(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.MaxPool_3a_3x3(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.MaxPool_5a_3x3(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.create_aux_logits:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.mean(x, (2, 3)) # use reduce instead of avgpool
        # N x 2048 x 1 x 1
        x = self.Dropout_last(x)
        # N x 2048 x 1 x 1

        x = self.flatten(x)
        # N x 2048
        x = self.fc(x)
        # N x num_classes

        # x = self.Conv2d_last(x)
        # N x num_classes x 1 x 1
        # x = self.flatten(x)
        # N x num_classes
        # if self.create_aux_logits:
        #     return x, aux
        return x

class InceptionBlockA(nn.Cell):
    
    def __init__(self, in_channels, var_channels):
        super(InceptionBlockA, self).__init__()
        self.branch_1 = Conv2dBlock(in_channels, out_channels=64, kernel_size=1)
        self.branch_2 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=48, kernel_size=1),
                            Conv2dBlock(in_channels=48, out_channels=64, kernel_size=5)])
        self.branch_3 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=64, kernel_size=1),
                            Conv2dBlock(in_channels=64, out_channels=96, kernel_size=3),
                            Conv2dBlock(in_channels=96, out_channels=96, kernel_size=3)])
        self.branch_4 = nn.SequentialCell([
                            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
                            Conv2dBlock(in_channels, out_channels=var_channels, kernel_size=1)])
        self.concat = P.Concat(axis=1)
             
    def construct(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)
        return self.concat((branch_1, branch_2, branch_3, branch_4))

class InceptionBlockB_1(nn.Cell):
    
    def __init__(self, in_channels):
        super(InceptionBlockB_1, self).__init__()
        self.branch_1 = Conv2dBlock(in_channels, out_channels=384, kernel_size=3, stride=2, pad_mode="valid")
        self.branch_2 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=64, kernel_size=1),
                            Conv2dBlock(in_channels=64, out_channels=96, kernel_size=3),
                            Conv2dBlock(in_channels=96, out_channels=96, kernel_size=3, stride=2, pad_mode="valid")])
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.concat = P.Concat(axis=1)
             
    def construct(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        return self.concat((branch_1, branch_2, branch_3))

class InceptionBlockB_2(nn.Cell):
    
    def __init__(self, in_channels, var_channels):
        super(InceptionBlockB_2, self).__init__()
        self.branch_1 = Conv2dBlock(in_channels, out_channels=192, kernel_size=1)
        self.branch_2 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=var_channels, kernel_size=1),
                            Conv2dBlock(in_channels=var_channels, out_channels=var_channels, kernel_size=(1,7)),
                            Conv2dBlock(in_channels=var_channels, out_channels=192, kernel_size=(7,1))])
        self.branch_3 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=var_channels, kernel_size=1),
                            Conv2dBlock(in_channels=var_channels, out_channels=var_channels, kernel_size=(7,1)),
                            Conv2dBlock(in_channels=var_channels, out_channels=var_channels, kernel_size=(1,7)),
                            Conv2dBlock(in_channels=var_channels, out_channels=var_channels, kernel_size=(7,1)),
                            Conv2dBlock(in_channels=var_channels, out_channels=192, kernel_size=(1,7))])
        self.branch_4 = nn.SequentialCell([
                            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
                            Conv2dBlock(in_channels, out_channels=192, kernel_size=1)])
        self.concat = P.Concat(axis=1)
             
    def construct(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        branch_4 = self.branch_4(x)
        return self.concat((branch_1, branch_2, branch_3, branch_4))

class InceptionBlockC_1(nn.Cell):
    
    def __init__(self, in_channels):
        super(InceptionBlockC_1, self).__init__()
        self.branch_1 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=192, kernel_size=1),
                            Conv2dBlock(in_channels=192, out_channels=320, kernel_size=3, stride=2, pad_mode="valid")])
        self.branch_2 = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=192, kernel_size=1),
                            Conv2dBlock(in_channels=192, out_channels=192, kernel_size=(1,7)),
                            Conv2dBlock(in_channels=192, out_channels=192, kernel_size=(7,1)),
                            Conv2dBlock(in_channels=192, out_channels=192, kernel_size=3, stride=2, pad_mode="valid")])
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.concat = P.Concat(axis=1)
             
    def construct(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)
        return self.concat((branch_1, branch_2, branch_3))

class InceptionBlockC_2(nn.Cell):
    
    def __init__(self, in_channels):
        super(InceptionBlockC_2, self).__init__()
        self.branch_1 = Conv2dBlock(in_channels, out_channels=320, kernel_size=1)
        self.branch_2_a = Conv2dBlock(in_channels, out_channels=384, kernel_size=1)
        self.branch_2_b_1x3 = Conv2dBlock(in_channels=384, out_channels=384, kernel_size=(1, 3))
        self.branch_2_b_3x1 = Conv2dBlock(in_channels=384, out_channels=384, kernel_size=(3, 1))
        self.branch_3_a = nn.SequentialCell([
                            Conv2dBlock(in_channels, out_channels=448, kernel_size=1),
                            Conv2dBlock(in_channels=448, out_channels=384, kernel_size=3)])
        self.branch_3_b_1x3 = Conv2dBlock(in_channels=384, out_channels=384, kernel_size=(1, 3))    # same with branch_2_b_1x3
        self.branch_3_b_3x1 = Conv2dBlock(in_channels=384, out_channels=384, kernel_size=(3, 1))    # same with branch_2_b_3x1
        self.branch_4 = nn.SequentialCell([
                            nn.AvgPool2d(kernel_size=3, stride=1, pad_mode='same'),
                            Conv2dBlock(in_channels, out_channels=192, kernel_size=1)])
        self.concat = P.Concat(axis=1)
             
    def construct(self, x):
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2_a(x)
        branch_2 = self.concat((self.branch_2_b_1x3(branch_2), self.branch_2_b_3x1(branch_2)))
        branch_3 = self.branch_3_a(x)
        branch_3 = self.concat((self.branch_3_b_1x3(branch_3), self.branch_3_b_3x1(branch_3)))
        branch_4 = self.branch_4(x)
        return self.concat((branch_1, branch_2, branch_3, branch_4))

class InceptionBlockAux(nn.Cell):
    
    def __init__(self, in_channels, num_classes):
        super(InceptionBlockAux, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3, pad_mode='valid')
        self.conv_1 = Conv2dBlock(in_channels, out_channels=128, kernel_size=1)
        # self.conv_2 = Conv2dBlock(in_channels=128, out_channels=768, kernel_size=ksize, pad_mode="valid", weight_init=TruncatedNormal(0.01))
        self.conv_2 = nn.SequentialCell([
                        Conv2dBlock(in_channels=128, out_channels=768, kernel_size=1, weight_init=TruncatedNormal(0.01)),
                        P.ReduceMean(keep_dims=True, axis=(2, 3))])
        self.conv_3 = Conv2dBlock(in_channels=768, out_channels=num_classes, kernel_size=1, weight_init=TruncatedNormal(0.001), with_relu=False)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.pool(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        return x

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

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                pad_mode="same", padding=0, weight_init="XavierUniform", with_relu=True, with_bn=True):
        super(Conv2dBlock, self).__init__()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, stride=stride, pad_mode=pad_mode,
                              padding=padding, weight_init=weight_init)
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
