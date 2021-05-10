#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

PYTHON_EXE="/userhome/software/conda_envs/mindspore-0.7/bin/python -u"

IMAGE_PATH="/userhome/datasets/LUNA16-NII/val_data"
SEG_PATH="/userhome/datasets/LUNA16-NII/val_label"
CHECKPOINT_FILE_PATH="/userhome/ms_models/unet3d-pytorch/scripts/train/ckpt_0/Unet3d-10_877.ckpt"



ulimit -u unlimited
export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export DEVICE_ID=0
export RANK_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval
echo "start eval for checkpoint file: ${CHECKPOINT_FILE_PATH}"
$PYTHON_EXE eval.py --data_url=$IMAGE_PATH --seg_url=$SEG_PATH --ckpt_path=$CHECKPOINT_FILE_PATH
# $PYTHON_EXE eval.py --data_url=$IMAGE_PATH --seg_url=$SEG_PATH --ckpt_path=$CHECKPOINT_FILE_PATH > eval.log 2>&1
echo "end eval for checkpoint file: ${CHECKPOINT_FILE_PATH}"
cd ..
