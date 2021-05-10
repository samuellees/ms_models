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

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet3D on images and target masks')
    parser.add_argument('--data_url', dest='data_url', type=str, default='', help='image data directory')
    parser.add_argument('--seg_url', dest='seg_url', type=str, default='', help='seg data directory')
    return parser.parse_args()

def train_net(data_dir,
              seg_dir,
              config=None):
    train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, config=config, is_training=True)
    # for item in train_dataset:
    #     print(item)
    # exit(0)

    train_data_size = len(train_dataset)
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)
    criterion = DiceLoss()
    # criterion = SoftmaxCrossEntropyWithLogits()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=1)
    scheduler = dynamic_lr_scheduler(config, train_data_size, optimizer)
    device = torch.device('cuda:0')
    network.to(device)

    print("============== Starting Training ==============")
    network.train()
    step_per_epoch = train_data_size
    for epoch_id in range(cfg.epoch_size):
        time_epoch = 0.0
        torch.cuda.synchronize(0)
        time_start = time.time()
        for batch_idx, data in enumerate(train_dataset, 0):
            inputs, labels = data
            inputs.squeeze_(0)
            labels.squeeze_(0)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zeros the parameter gradients
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # print statistics
            running_loss = loss.item()
            torch.cuda.synchronize(0)
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                        (epoch_id, cfg.epoch_size, batch_idx + 1, step_per_epoch, 
                        running_loss, time_step))
            time_start = time.time()
        print('Epoch time: %10.4f, per step time: %7.4f' % (time_epoch, time_epoch / step_per_epoch))

        ckpt_path = "./ckpt_0/Unet3d"
        ckpt_file = ('%s-%d_%d.ckpt' % (ckpt_path, epoch_id + 1, step_per_epoch))
        torch.save(network, ckpt_file)

    print("============== End Training ==============")

if __name__ == '__main__':
    args = get_args()
    print("Training setting:", args)
    train_net(data_dir=args.data_url,
              seg_dir=args.seg_url,
              config=cfg)
