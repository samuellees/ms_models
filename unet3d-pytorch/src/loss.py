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
from src.config import config

class SoftmaxCrossEntropyWithLogits(nn.Module):
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, logits, label):
        logits = logits.permute(0, 2, 3, 4, 1)
        label = label.permute(0, 2, 3, 4, 1)
        loss = self.cross_entropy_loss(torch.reshape(logits, (-1, config['num_classes'])), 
                                    torch.reshape(label, (-1, config['num_classes'])))

        return loss

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, label):
        intersection = torch.sum(torch.mul(logits, label))
        unionset = torch.sum(torch.mul(logits, logits)) + torch.sum(torch.mul(label, label))

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff

        return dice_loss