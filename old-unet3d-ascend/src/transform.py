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

import re
import numpy as np
from mindspore import log as logger
import nibabel as nib
from src.utils import correct_nifti_header_if_necessary, fall_back_tuple, get_valid_patch_size, \
    get_random_patch
np_str_obj_array_pattern = re.compile(r'[SaUO]')

MAX_SEED = np.iinfo(np.uint32).max + 1

class Dataset:
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.

    Args:
        data: input data to load and transform to generate dataset for model.
        transform: a callable data transform on input data.
    """
    def __init__(self, data, seg, transform=None):
        self.data = data
        self.seg = seg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        seg = self.seg[index]
        return [data], [seg]


class LoadNifti:
    """
    Load Nifti format file or files from provided path.
    """
    def __init__(self, as_closest_canonical=False, dtype=np.float32):
        """
        Args:
        as_closest_canonical: if True, load the image as closest to canonical axis format.
        dtype: if not None convert the loaded image to this data type.
        """
        self.as_closest_canonical = as_closest_canonical
        self.dtype = dtype

    def operation(self, filename):
        """
        Args:
            filename: path file or file-like object or a list of files.
        """
        img_array = list()
        compatible_meta = dict()
        filename = str(filename, encoding="utf-8")
        filename = [filename]
        for name in filename:
            img = nib.load(name)
            img = correct_nifti_header_if_necessary(img)
            header = dict(img.header)
            header["filename_or_obj"] = name
            header["affine"] = img.affine
            header["original_affine"] = img.affine.copy()
            header["as_closest_canonical"] = self.as_closest_canonical
            ndim = img.header["dim"][0]
            spatial_rank = min(ndim, 3)
            header["spatial_shape"] = img.header["dim"][1 : spatial_rank + 1]
            if self.as_closest_canonical:
                img = nib.as_closest_canonical(img)
                header["affine"] = img.affine
            img_array.append(np.array(img.get_fdata(dtype=self.dtype)))
            img.uncache()
            if not compatible_meta:
                for meta_key in header:
                    meta_datum = header[meta_key]
                    if isinstance(meta_datum, np.ndarray) \
                        and np_str_obj_array_pattern.search(meta_datum.dtype.str) is not None:
                        continue
                    compatible_meta[meta_key] = meta_datum
            else:
                assert np.allclose(header["affine"], compatible_meta["affine"])

        img_array = np.stack(img_array, axis=0) if len(img_array) > 1 else img_array[0]
        return img_array

    def __call__(self, filename1, filename2):
        img_array = self.operation(filename1)
        seg_array = self.operation(filename2)
        return img_array, seg_array


class AddChannel:
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).
    """
    def operation(self, img):
        """
        Apply the transform to `img`.
        """
        return img[None]

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = self.operation(label)
        return img_array, seg_array


class Orientation:
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """
    def __init__(self, axcodes=None, as_closest_canonical=False, labels=tuple(zip("LPI", "RAS")), \
                meta_key_postfix: str = "meta_dict"):
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
        if axcodes is not None and as_closest_canonical:
            logger.warning("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels
        self.meta_key_postfix = meta_key_postfix

    def operation(self, data_array: np.ndarray, affine=None):
        """
        original orientation of `data_array` is defined by `affine`.

        Args:
            data_array: in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.

        Raises:
            ValueError: When ``data_array`` has no spatial dimensions.
            ValueError: When ``axcodes`` spatiality differs from ``data_array``.

        Returns:
            data_array (reoriented in `self.axcodes`), original axcodes, current axcodes.
        """
        sr = data_array.ndim - 1
        if sr <= 0:
            raise ValueError("data_array must have at least one spatial dimension.")
        if affine is None:
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_ = to_affine_nd(sr, affine)
        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            assert self.axcodes is not None
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"axcodes must match data_array spatially, got axcodes={len(self.axcodes)}D data_array={sr}D")
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        data_array = nib.orientations.apply_orientation(data_array, ornt)
        return data_array

    def __call__(self, img, label):
        img_array = self.operation(img)
        seg_array = self.operation(label)
        return img_array, seg_array


class ScaleIntensityRange:
    """
    Apply specific intensity scaling to the whole numpy array.
    Scaling from [a_min, a_max] to [b_min, b_max] with clip option.

    Args:
        a_min: intensity original range min.
        a_max: intensity original range max.
        b_min: intensity target range min.
        b_max: intensity target range max.
        clip: whether to perform clip after scaling.
    """
    def __init__(self, a_min: float, a_max: float, b_min: float, b_max: float, clip: bool = False) -> None:
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip

    def operation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the transform to `img`.
        """
        if self.a_max - self.a_min == 0.0:
            logger.warning("Divide by zero (a_min == a_max)")
            return img - self.a_min + self.b_min
        img = (img - self.a_min) / (self.a_max - self.a_min)
        img = img * (self.b_max - self.b_min) + self.b_min
        if self.clip:
            img = np.clip(img, self.b_min, self.b_max)
        return img

    def __call__(self, image, label):
        image = self.operation(image)
        return image, label


class RandSpatialCropSamples:
    """
    Dictionary-based version :py:class:`monai.transforms.RandSpatialCrop`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_size: if `random_size` is True, it specifies the minimum crop region.
            if `random_size` is False, it specifies the expected ROI size to crop. e.g. [224, 224, 128]
            If its components have non-positive values, the corresponding size of input image will be used.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specific size ROI.
            The actual size is sampled from `randint(roi_size, img_size)`.
    """
    def __init__(self, roi_size, random_center=True, random_size=True, num_samples=1, is_training=True):
        self.is_training = is_training
        self.roi_size = roi_size
        self.random_center = random_center
        self.random_size = random_size
        self.num_samples = num_samples
        self._slices = None
        self._size = None
        if self.is_training:
            self.set_random_state()
        else:
            self.set_random_state(0)

    def set_random_state(self, seed=None, state=None):
        """
        Set the random state locally, to control the randomness, the derived
        classes should use :py:attr:`self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed: set the random state with an integer seed.
            state: set the random state with a `np.random.RandomState` object.

        Returns:
            a Randomizable instance.
        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % MAX_SEED
            self.R = np.random.RandomState(_seed)

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
        self.R = np.random.RandomState()
        return self

    def randomize(self, img_size):
        self._size = fall_back_tuple(self.roi_size, img_size)
        if self.random_size:
            self._size = [self.R.randint(low=self._size[i], high=img_size[i] + 1) for i in range(len(img_size))]
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = (slice(None),) + get_random_patch(img_size, valid_size, self.R)

    def __call__(self, image, label):
        res_image = []
        res_label = []
        for _ in range(self.num_samples):
            self.randomize(image.shape[1:])
            assert self._size is not None
            if self.random_center:
                img = image[self._slices]
                label_crop = label[self._slices]
            else:
                cropper = CenterSpatialCrop(self._size)
                img = cropper(image)
                label_crop = label(img)
            res_image.append(img)
            res_label.append(label_crop)
        return np.array(res_image), np.array(res_label)


class OneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def one_hot(self, labels):
        N, K = labels.shape
        one_hot_encoding = np.zeros((N, self.num_classes, K), dtype=np.float32)
        for i in range(N):
            for j in range(K):
                one_hot_encoding[i, labels[i][j], j] = 1
        return one_hot_encoding

    def operation(self, labels):
        N, _, D, H, W = labels.shape
        labels = labels.astype(np.int32)
        labels = np.reshape(labels, (N, -1))
        labels = self.one_hot(labels)
        labels = np.reshape(labels, (N, self.num_classes, D, H, W))
        return labels

    def __call__(self, image, label):
        label = self.operation(label)
        return image, label
