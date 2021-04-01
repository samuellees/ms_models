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
from mindspore import Model, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from src.dataset import create_dataset
from src.unet3d_model import UNet3d
from src.config import config as cfg
from src.loss import SoftmaxCrossEntropyWithLogits

device_id = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, \
                    device_id=device_id)
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
    train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, config=config, \
                                    rank_size=rank_size, rank_id=rank_id, is_training=True)
    train_data_size = train_dataset.get_dataset_size()
    print("train dataset length is:", train_data_size)

    network = UNet3d(config=config)

    loss = SoftmaxCrossEntropyWithLogits()
    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=config.lr)
    scale_manager = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    network.set_train()

    model = Model(network, loss_fn=loss, optimizer=optimizer, loss_scale_manager=scale_manager)

    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=config.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(config.model),
                                 directory='./ckpt_{}/'.format(device_id),
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks_list)
    print("============== End Training ==============")

if __name__ == '__main__':
    args = get_args()
    print("Training setting:", args)
    train_net(data_dir=args.data_url,
              seg_dir=args.seg_url,
              run_distribute=args.run_distribute,
              config=cfg)






import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from queue import Queue

from src.unet3d_model import UNet3d
from src.config import config as cfg
from src.dataset import create_dataset
from src.loss import SoftmaxCrossEntropyWithLogits


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pytorch Unet3D on CUDA')
  parser.add_argument('--data_url', type=str, default="",
                      help='path where the data is saved')
  parser.add_argument('--seg_url', type=str, default="",
                      help='path where the seg label is saved')
  parser.add_argument('--device_id', type=int, default=0, help='device id of GPU. (Default: 0)')
  args = parser.parse_args()

  device = torch.device('cuda:'+str(args.device_id))
  network = UNet3d(cfg.num_classes, config=cfg)
  network.to(device)
  criterion = SoftmaxCrossEntropyWithLogits()
  optimizer = nn.Adam(learning_rate=cfg.lr)
  
  train_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, config=config, \
                                rank_size=rank_size, rank_id=rank_id, is_training=True)
  dataloader = create_dataset_pytorch(args.data_path, is_train=True)
  step_per_epoch = len(dataloader)
  scheduler = optim.lr_scheduler.StepLR(
                            optimizer, 
                            gamma=cfg.lr_decay_rate, 
                            step_size=cfg.lr_decay_epoch*step_per_epoch)
  # scheduler = optim.lr_scheduler.ExponentialLR(
  #                             optimizer, 
  #                             gamma=cfg.lr_decay_rate)

  q_ckpt = Queue(maxsize=cfg.keep_checkpoint_max)

  global_step_id = 0
  for epoch in range(cfg.epoch_size):
    time_epoch = 0.0
    torch.cuda.synchronize()
    for i, data in enumerate(dataloader, 0):
      time_start = time.time()
      inputs, labels = data
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
      torch.cuda.synchronize()
      time_end = time.time()
      time_step = time_end - time_start
      time_epoch = time_epoch + time_step
      print('Epoch: [%3d/%3d], step: [%5d/%5d], loss: [%6.4f], time: [%.4f]' %
            (epoch + 1, cfg.epoch_size, i + 1, step_per_epoch, 
              running_loss, time_step), flush=True)
      
      # save checkpoint every epoch
      global_step_id = global_step_id + 1
      if global_step_id % cfg.save_checkpoint_steps == 0:
        if q_ckpt.full():
          last_file = q_ckpt.get()
          os.remove(last_file)
        ckpt_file = ('%s/squeezenet%s_%d-%d.ckpt' % 
                    (args.ckpt_path, version, epoch + 1, i + 1))
        q_ckpt.put(ckpt_file)
        torch.save(network, ckpt_file)

    print('Epoch time: %10.4f, per step time: %7.4f' %
          (time_epoch, time_epoch / step_per_epoch), flush=True)

  print('Finished Training', flush=True)
    
