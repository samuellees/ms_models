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

import argparse
from src.config import squeezenet_cfg as cfg
from src.dataset import create_dataset
from src.squeezenet import SqueezeNet
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.callback import SummaryCollector
# from mindspore.summary.summary_record import SummaryRecord
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.nn.dynamic_lr import exponential_decay_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore SqueezeNet Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    parser.add_argument('--summary_path', type=str, default="./summary",
                        help='path where the summary to be saved')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint",
                        help='path where the checkpoint to be saved')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='dataset_sink_mode is False or True')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU. (Default: 0)')
    args = parser.parse_args()
    if args.device_target == "CPU":
        args.dataset_sink_mode = False

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
    # create network
    version = '1.0'
    network = SqueezeNet(cfg.num_classes, version)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, 
                reduction="mean", smooth_factor=cfg.label_smoothing_eps)
    ds_train = create_dataset(args.data_path, cfg.batch_size, cfg.epoch_size)
    # set optimizer
    step_per_epoch = ds_train.get_dataset_size()
    total_step = step_per_epoch * cfg.epoch_size
    lr = exponential_decay_lr(learning_rate=cfg.lr_init, 
            decay_rate=cfg.lr_decay_rate, total_step=total_step, 
            step_per_epoch=step_per_epoch, decay_epoch=cfg.lr_decay_epoch)
    net_opt = nn.RMSProp(network.trainable_params(), learning_rate=lr, 
                decay=cfg.rmsprop_decay, momentum=cfg.rmsprop_momentum, 
                epsilon=cfg.rmsprop_epsilon)
    # net_opt = nn.RMSProp(network.trainable_params(), learning_rate=cfg.lr_init, 
    #             decay=cfg.rmsprop_decay, momentum=cfg.rmsprop_momentum, 
    #             epsilon=cfg.rmsprop_epsilon)
    
    # set callbacks
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="squeezenet"+version, 
                                 config=config_ck,
                                 directory=args.ckpt_path)
    summary_cb = SummaryCollector(args.summary_path,
                                collect_freq=1,
                                keep_default_action=False,
                                collect_specified_data={'collect_graph': True})
    # set model
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    # model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor(), summary_cb],
    #             dataset_sink_mode=args.dataset_sink_mode)
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
