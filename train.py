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
"""
######################## train inceptionv3 example ########################
train inceptionv3 and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import argparse
from src.config import inceptionv3_cfg as cfg
from src.dataset import create_dataset
from src.inceptionv3 import Inceptionv3
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Inceptionv3 Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./Data",
                        help='path where the dataset is saved')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='dataset_sink_mode is False or True')
    args = parser.parse_args()

    if args.device_target == "CPU":
        args.dataset_sink_mode = False

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(args.data_path, cfg.batch_size, cfg.epoch_size)

    network = Inceptionv3(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
    # net_opt = nn.RMSProp(network.trainable_params(), learning_rate=cfg.lr, weight_decay=cfg.decay)
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, 0.9)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_inceptionv3", config=config_ck)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
