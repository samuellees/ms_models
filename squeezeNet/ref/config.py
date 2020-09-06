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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

squeezenet_cfg = edict({
    'lr_init': 0.0045,
    'lr_decay_rate': 0.94,
    'lr_decay_epoch': 2,
    'rmsprop_decay': 0.9,
    'rmsprop_momentum': 0.9,
    'rmsprop_epsilon': 1.0,
    'label_smoothing_eps': 0.1,

    'epoch_size': 200,
    # 'epoch_size': 1,
    'batch_size': 32,
    'buffer_size': 1000,
    'keep_checkpoint_max': 10,
    'save_checkpoint_steps': 1562,
    'num_classes': 10,
    'image_height': 224,
    'image_width': 224,
})
