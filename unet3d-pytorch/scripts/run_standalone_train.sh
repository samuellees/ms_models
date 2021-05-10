#!/bin/bash
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


PATH1="/userhome/datasets/LUNA16-NII/train_data"
PATH2="/userhome/datasets/LUNA16-NII/train_label"

PYTHON_EXE="/userhome/software/conda_envs/mindspore-0.7/bin/python -u"

ulimit -u unlimited
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1
# export GLOG_v=1

rm -rf ./train
mkdir ./train
mkdir ./train/ckpt_0
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
$PYTHON_EXE train.py --data_url=$PATH1 --seg_url=$PATH2 > train.log 2>&1
# $PYTHON_EXE train.py --data_url=$PATH1 --seg_url=$PATH2
cd ..
