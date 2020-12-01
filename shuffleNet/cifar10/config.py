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

cfg = edict({
    'lr_init': 0.5,
    'lr_decay_rate': 0.94,
    'SGD_momentum': 0.9,
    'SGD_weight_decay': 4e-5,
    'n_workers': 4,
    'epoch_size': 800,
    'batch_size': 128,
    'buffer_size': 1000,
    'num_classes': 10,
    'image_size': 224
})
