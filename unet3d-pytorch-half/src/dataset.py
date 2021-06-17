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
# =========================================================================

import os
import glob
import numpy as np
from src.config import config as cfg
from src.transform import ExpandChannel, LoadData, Orientation, ScaleIntensityRange, RandomCropSamples, OneHot

import torchvision
import torch

class TorchDataset(torch.utils.data.Dataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.

    Args:
        data: input data to load and transform to generate dataset for model.
        seg: segment data to load and transform to generate dataset for model
    """
    def __init__(self, data, seg, transform=None):
        self.data = data
        self.seg = seg
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        seg = self.seg[index]
        for trans in self.transform:
            data, seg = trans(data, seg)
        return data, seg

class ConvertLabel:
    """
    Crop at the center of image with specified ROI size.

    Args:
        roi_size: the spatial size of the crop region e.g. [224,224,128]
        If its components have non-positive values, the corresponding size of input image will be used.
    """
    def operation(self, data):
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        data[data > cfg['upper_limit']] = 0
        data = data - (cfg['lower_limit'] - 1)
        data = np.clip(data, 0, cfg['lower_limit'])
        return data

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label




def create_dataset(data_path, seg_path, config, is_training=True):
    seg_files = sorted(glob.glob(os.path.join(seg_path, "*.nii.gz")))
    train_files = [os.path.join(data_path, os.path.basename(seg)) for seg in seg_files]

    if is_training:
        transform_image = [LoadData(),
                            ExpandChannel(),
                            Orientation(),
                            ScaleIntensityRange(src_min=config.min_val, src_max=config.max_val, tgt_min=0.0, \
                                                tgt_max=1.0, is_clip=True),
                            RandomCropSamples(roi_size=config.roi_size, num_samples=2),
                            ConvertLabel(),
                            OneHot(num_classes=config.num_classes)]
    else:
        transform_image = [LoadData(),
                            ExpandChannel(),
                            Orientation(),
                            ScaleIntensityRange(src_min=config.min_val, src_max=config.max_val, tgt_min=0.0, \
                                                tgt_max=1.0, is_clip=True),
                            ConvertLabel()]

    dataset = TorchDataset(data=train_files, seg=seg_files, transform=transform_image)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
            batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=8)

    return data_loader
