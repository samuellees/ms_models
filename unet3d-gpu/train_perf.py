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
import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset, create_dataset_diy
from src.unet3d_model import UNet3d
from src.config import config as cfg
from src.lr_schedule import dynamic_lr
from src.loss import SoftmaxCrossEntropyWithLogits
import time

import numpy as np
import logging

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, \
                    device_id=device_id)
# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=True, \
#                     device_id=device_id)
mindspore.set_seed(1)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet3D on images and target masks')
    parser.add_argument('--data_url', dest='data_url', type=str, default='', help='image data directory')
    parser.add_argument('--seg_url', dest='seg_url', type=str, default='', help='seg data directory')
    parser.add_argument('--run_distribute', dest='run_distribute', type=ast.literal_eval, default=False, \
                        help='Run distribute, default: false')
    return parser.parse_args()

def train_net(data_dir,
              seg_dir,
              run_distribute,
              config=None):
    train_data_size = 5
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)

    loss = nn.DiceLoss()
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    network.set_train()

    inputs = mindspore.Tensor(np.ones((1, 1, 224, 224, 96)), mstype.float32)
    labels = mindspore.Tensor(np.ones((1, 4, 224, 224, 96)), mstype.float32)

    net = nn.wrap.WithLossCell(network, loss)
    model = nn.TrainOneStepCell(net, optimizer).set_train()
    step_per_epoch = train_data_size
    print("============== Starting Training ==============")
    # for epoch_id in range(1):
    for epoch_id in range(cfg.epoch_size):
        time_epoch = 0.0
        # for step_id in range(1):
        for step_id in range(step_per_epoch):
            time_start = time.time()
            loss = model(inputs, labels)
            # loss = network(inputs)
            loss = loss.asnumpy()
            time_end = time.time()
            time_step = time_end - time_start
            time_epoch = time_epoch + time_step
            print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
                        (epoch_id, cfg.epoch_size, step_id, step_per_epoch, 
                        loss, time_step))
        print('Epoch time: %10.4f, per step time: %7.4f' % (time_epoch, time_epoch / step_per_epoch))

    print("============== End Training ==============")

if __name__ == '__main__':
    args = get_args()
    print("Training setting:", args)
    train_net(data_dir=args.data_url,
              seg_dir=args.seg_url,
              run_distribute=args.run_distribute,
              config=cfg)
