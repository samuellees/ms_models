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

# PATH_DATA="/gpfs/share/home/1600011337/likesen/datasets/cifar-10-batches-py"
# PATH_MODEL="/gpfs/share/home/1600011337/likesen/ms_models/xception_pytorch_cifar10"

PATH_MODEL="/userhome/ms_models/shuffleNet/cifar10"
PATH_DATA="/userhome/datasets"
# PATH_DATA="/dev/shm/dataset"

PATH_TRAIN=$PATH_MODEL"/train"$(date "+%Y%m%d%H%M%S")
PATH_INFER=$PATH_MODEL"/infer"$(date "+%Y%m%d%H%M%S")

PATH_CKPT=$PATH_TRAIN"/checkpoint"

PYTHON_EXE="/userhome/software/conda_envs/mindspore-0.7/bin/python -u"

export DEVICE_ID=0

if [ -d $PATH_TRAIN ];
then
    rm -rf $PATH_TRAIN
fi
cd $PATH_MODEL
mkdir $PATH_TRAIN
cp *.py $PATH_TRAIN
cd $PATH_TRAIN || exit
mkdir $PATH_CKPT

echo "start training for device $DEVICE_ID"
$PYTHON_EXE train.py --data_path=$PATH_DATA --ckpt_path=$PATH_CKPT  --device_id=$DEVICE_ID
# $PYTHON_EXE train.py --data_path=$PATH_DATA --ckpt_path=$PATH_CKPT  --device_id=$DEVICE_ID > log.txt 2>&1 
cd ..
