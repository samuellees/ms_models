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
from mindspore.ops import functional as F
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

import numpy as np
import time

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False, \
                    device_id=device_id)
mindspore.set_seed(1)

class OutputTo16(nn.Cell):
    "Wrap cell for amp. Cast network output back to float16"

    def __init__(self, op):
        super(OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


def _do_keep_batchnorm_fp32(network):
    """Do keep batchnorm fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            network._cells[name] = OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _do_keep_batchnorm_fp32(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())

def _add_loss_network(network, loss_fn, cast_model_type):
    """Add loss network."""
    class WithLossCell(nn.Cell):
        "Wrap loss for amp. Cast network output back to float32"

        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    if cast_model_type == mstype.float16:
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet3D on images and target masks')
    parser.add_argument('--data_url', dest='data_url', type=str, default='', help='image data directory')
    parser.add_argument('--seg_url', dest='seg_url', type=str, default='', help='seg data directory')
    parser.add_argument('--run_distribute', dest='run_distribute', type=ast.literal_eval, default=False, \
                        help='Run distribute, default: false')
    return parser.parse_args()

def train_net__(data_dir,
              seg_dir,
              run_distribute,
              config=None):

    train_data_size = 5
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)

    loss = SoftmaxCrossEntropyWithLogits()
    # loss = nn.DiceLoss()
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    network.set_train()
    network.to_float(mstype.float16)
    _do_keep_batchnorm_fp32(network)
    network = _add_loss_network(network, loss, mstype.float16)
    loss_scale = 1.0
    loss_scale = scale_manager.get_loss_scale()
    update_cell = scale_manager.get_update_cell()
    if update_cell is not None:
        model = nn.TrainOneStepWithLossScaleCell(network, optimizer,
                                                    scale_sense=update_cell).set_train()
    else:
        model = nn.TrainOneStepCell(network, optimizer, loss_scale).set_train()

    inputs = mindspore.Tensor(np.random.rand(1, 1, 224, 224, 96), mstype.float32)
    labels = mindspore.Tensor(np.random.rand(1, 4, 224, 224, 96), mstype.float32)

    step_per_epoch = train_data_size
    print("============== Starting Training ==============")
    # for epoch_id in range(1):
    for epoch_id in range(cfg.epoch_size):
        time_epoch = 0.0
        for step_id in range(step_per_epoch):
        # for step_id in range(1):
            time_start = time.time()
            loss = model(inputs, labels)
            # loss = network(inputs, labels)
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



def train_net(data_dir,
              seg_dir,
              run_distribute,
              config=None):
    if run_distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=True)
    else:
        rank_id = 0
        rank_size = 1
    # train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, config=config, \
    #                                 rank_size=rank_size, rank_id=rank_id, is_training=True)
    train_dataset = create_dataset_diy()
    # for item in train_dataset:
    #     print(item)
    # exit(0)

    train_data_size = train_dataset.get_dataset_size()
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)

    loss = SoftmaxCrossEntropyWithLogits()
    # loss = nn.DiceLoss()
    lr = Tensor(dynamic_lr(config, train_data_size), mstype.float32)
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr)
    scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    network.set_train()

    model = Model(network, loss_fn=loss, optimizer=optimizer, loss_scale_manager=scale_manager,
                  amp_level='O3')

    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor(per_print_times=2)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(config.model),
                                 directory='./ckpt_{}/'.format(rank_size),
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list, dataset_sink_mode=False)
    print("============== End Training ==============")


if __name__ == '__main__':
    args = get_args()
    print("Training setting:", args)
    train_net(data_dir=args.data_url,
              seg_dir=args.seg_url,
              run_distribute=args.run_distribute,
              config=cfg)
