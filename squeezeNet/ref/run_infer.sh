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

PATH_DATA="../../../datasets"
PATH_CKPT="./train/checkpoint/squeezenet1.0_100-1562.ckpt"

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH_DATA=$(get_real_path $PATH_DATA)
PATH_CKPT=$(get_real_path $PATH_CKPT)


if [ ! -d $PATH_DATA ]
then 
    echo "error: DATASET_PATH=$PATH_DATA is not a directory"
exit 1
fi 

if [ ! -f $PATH_CKPT ]
then 
    echo "error: CHECKPOINT_PATH=$PATH_CKPT is not a file"
exit 1
fi 

export DEVICE_NUM=1
export DEVICE_ID=3
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

if [ -d "infer" ];
then
    rm -rf ./infer
fi
mkdir ./infer
cp *.py ./infer
cd ./infer || exit
echo "start infering for device $DEVICE_ID"
python eval.py --data_path=$PATH_DATA --ckpt_path=$PATH_CKPT --device_id=$DEVICE_ID
# python eval.py --data_path=$PATH_DATA --ckpt_path=$PATH_CKPT --device_id=$DEVICE_ID  &> log &
cd ..
