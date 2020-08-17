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
######################## eval inceptionv3 example ########################
eval inceptionv3 according to model file:
python eval.py --data_path /YourDataPath --ckpt_path Your.ckpt
"""

import argparse
from src.config import inceptionv3_cfg as cfg
from src.dataset import create_dataset
from src.inceptionv3 import Inceptionv3
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.nn.dynamic_lr import exponential_decay_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Inceptionv3 Example')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./", help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=bool, default=True, help='dataset_sink_mode is False or True')
    parser.add_argument('--device_id', type=int, default=0, help='device id of GPU. (Default: 0)')
    args = parser.parse_args()

    if args.device_target == "CPU":
        args.dataset_sink_mode = False

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)

    network = Inceptionv3(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, 
                reduction="mean", smooth_factor=cfg.label_smoothing_eps)
    ds_eval = create_dataset(args.data_path, cfg.batch_size, status="test")
    step_per_epoch = ds_eval.get_dataset_size()
    total_step = step_per_epoch * cfg.epoch_size
    lr = exponential_decay_lr(learning_rate=cfg.lr_init, 
            decay_rate=cfg.lr_decay_rate, total_step=total_step, 
            step_per_epoch=step_per_epoch, decay_epoch=cfg.lr_decay_epoch)
    net_opt = nn.RMSProp(network.trainable_params(), learning_rate=lr, 
                decay=cfg.rmsprop_decay, momentum=cfg.rmsprop_momentum, 
                epsilon=cfg.rmsprop_epsilon)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
