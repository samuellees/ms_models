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

PATH1="/userhome/datasets/LUNA16-NII/train_data"
PATH2="/userhome/datasets/LUNA16-NII/train_label"

PYTHON_EXE="/root/anaconda3/bin/python -u"

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
# export RANK_TABLE_FILE=$PATH0

# for((i=0; i<${DEVICE_NUM}; i++))
# do
#     export DEVICE_ID=$i
#     export RANK_ID=$i
#     rm -rf ./train_parallel$i
#     mkdir ./train_parallel$i
#     cp ../*.py ./train_parallel$i
#     cp *.sh ./train_parallel$i
#     cp -r ../src ./train_parallel$i
#     cd ./train_parallel$i || exit
#     echo "start training for rank $RANK_ID, device $DEVICE_ID"
#     env > env.log

#     if [ $i == $[${DEVICE_NUM}-1] ]; then
#       $PYTHON_EXE train.py \
#       --run_distribute=True \
#       --data_url=$PATH1 \
#       --seg_url=$PATH2 > train.log 2>&1
#       cd ../
#     else
#       $PYTHON_EXE train.py \
#       --run_distribute=True \
#       --data_url=$PATH1 \
#       --seg_url=$PATH2 > train.log 2>&1 &
#       cd ../
#     fi
# done


rm -rf ./train_parallel_half_o2
mkdir ./train_parallel_half_o2
cp ../*.py ./train_parallel_half_o2
cp *.sh ./train_parallel_half_o2
cp -r ../src ./train_parallel_half_o2
cd ./train_parallel_half_o2 || exit
echo "start distributed training with $DEVICE_NUM GPUs."
env >env.log
mpirun --allow-run-as-root -n $DEVICE_NUM $PYTHON_EXE train_half_o2.py --run_distribute=True --data_url $PATH1 --seg_url $PATH2 > train.log 2>&1 
cd ..