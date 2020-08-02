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

PATH1="../datasets/cifar-10-batches-bin"
PATH2="./train/summary"

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $PATH1)
PATH2=$(get_real_path $PATH2)

if [ ! -d "$PATH1" ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi 

export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0

# mkdir ./train and enter ./train
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp -r src ./train
cp *.py ./train
cd ./train || exit
mkdir $PATH2

echo "start training for device $DEVICE_ID"
python train.py --data_path=$PATH1 --summary_path=$PATH2 --device_target=GPU
# python train.py --data_path=$PATH1 --summary_path=$PATH2 --device_target=GPU &> log &
cd ..
