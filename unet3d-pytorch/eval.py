# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from queue import Queue
from src.dataset import create_dataset
from src.unet3d_model import UNet3d
from src.config import config
from src.lr_schedule import dynamic_lr_scheduler
from src.loss import DiceLoss
from src.utils import create_sliding_window, CalculateDice
import numpy as np


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Test the UNet3D on images and target masks')
  parser.add_argument('--data_url', dest='data_url', type=str, default='', help='image data directory')
  parser.add_argument('--seg_url', dest='seg_url', type=str, default='', help='seg data directory')
  parser.add_argument('--ckpt_path', dest='ckpt_path', type=str, default='', help='checkpoint path')
  args = parser.parse_args()

  device = torch.device('cuda:0')
  network = torch.load(args.ckpt_path)
  network.eval()
  network.to(device)

  train_dataset = create_dataset(data_path=args.data_url, seg_path=args.seg_url, config=config, is_training=False)
  with torch.no_grad():
    eval_data_size = len(train_dataset)
    index = 0
    total_dice = 0
    for batch_idx, data in enumerate(train_dataset, 0):
        image, seg = data
        image = image.numpy()
        seg = seg.numpy()
        print("current image shape is {}".format(image.shape), flush=True)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        for window, slice_ in zip(sliding_window_list, slice_list):
            window_image = torch.Tensor(window).type(torch.float32).to(device)
            pred_probs = network(window_image)
            output_image[slice_] += pred_probs.cpu().numpy()
            count_map[slice_] += importance_map
        output_image = output_image / count_map
        dice, _ = CalculateDice(output_image, seg)
        print("The {} batch dice is {}".format(index, dice), flush=True)
        total_dice += dice
        index = index + 1
    avg_dice = total_dice / eval_data_size
    print("**********************End Eval***************************************")
    print("eval average dice is {}".format(avg_dice))
