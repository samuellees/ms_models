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

import os
import argparse
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from queue import Queue
from src.dataset import create_dataset
from src.unet3d_model import UNet3d
from src.config import config as cfg
from src.lr_schedule import dynamic_lr_scheduler
from src.loss import SoftmaxCrossEntropyWithLogits, DiceLoss

torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train_net(config=None):

    network = UNet3d(config=config)
    criterion = DiceLoss()
    # criterion = SoftmaxCrossEntropyWithLogits()
    device = torch.device('cuda:0')
    network.to(device)

    optimizer = torch.optim.Adam(params=network.parameters(), lr=1)
    scheduler = dynamic_lr_scheduler(config, 877, optimizer)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : step, last_epoch=-1)
    lrs = []
    for i in range(887):
        lrs.append(scheduler.get_lr())
        optimizer.step()
        scheduler.step()
    print(lrs)
    print("============== Starting Training ==============")
    network.train()
    time_epoch = 0.0
    inputs = torch.ones((1, 1, 144, 144, 144)).type(torch.float32)
    inputs = inputs.to(device)
    # zeros the parameter gradients
    outputs = network(inputs)
    # print(outputs)

    print("============== End Training ==============")

if __name__ == '__main__':
    train_net(config=cfg)
