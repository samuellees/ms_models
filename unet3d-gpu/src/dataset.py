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
import mindspore.dataset as ds
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.communication.management import init, get_rank, get_group_size
from src.transform import Dataset, AddChannel, LoadNifti, Orientation, ScaleIntensityRange, RandSpatialCropSamples, OneHot

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
        data[data > 5] = 0
        data = data - 2
        data = np.clip(data, 0, 3)
        return data

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label

def create_dataset(data_path, seg_path, config, is_training=True, run_distribute=False):
    seg_files = sorted(glob.glob(os.path.join(seg_path, "*.nii.gz")))
    train_files = [os.path.join(data_path, os.path.basename(seg)) for seg in seg_files]
    train_ds = Dataset(data=train_files, seg=seg_files)
    if run_distribute:
        init()
        rank_id = get_rank()
        rank_size = get_group_size()
        train_loader = ds.GeneratorDataset(train_ds, column_names=["image", "seg"], num_parallel_workers=4, \
                                           shuffle=is_training, num_shards=rank_size, shard_id=rank_id)
    else:
        train_loader = ds.GeneratorDataset(train_ds, column_names=["image", "seg"], num_parallel_workers=12, \
                                           shuffle=is_training)
    if is_training:
        transform_image = Compose([LoadNifti(),
                                   AddChannel(),
                                   Orientation(axcodes="RAS"),
                                   ScaleIntensityRange(a_min=config.min_val, a_max=config.max_val, b_min=0.0, \
                                                       b_max=1.0, clip=True),
                                   RandSpatialCropSamples(roi_size=config.roi_size, num_samples=2, random_size=False, \
                                                          is_training=is_training),
                                   ConvertLabel(),
                                   OneHot(num_classes=config.num_classes, is_training=is_training)])
    else:
        transform_image = Compose([LoadNifti(),
                                   AddChannel(),
                                   Orientation(axcodes="RAS"),
                                   ScaleIntensityRange(a_min=config.min_val, a_max=config.max_val, b_min=0.0, \
                                                       b_max=1.0, clip=True),
                                   ConvertLabel()])

    train_loader = train_loader.map(operations=transform_image, input_columns=["image", "seg"], num_parallel_workers=12,
                                    python_multiprocessing=True)
    if not is_training:
        train_loader = train_loader.batch(1)
    return train_loader
