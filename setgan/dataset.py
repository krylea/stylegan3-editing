import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, random_split, RandomSampler
import torch


import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import cv2
import math
import os
import random


# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import copy

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 1,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._base_raw_idx = copy.deepcopy(self._raw_idx)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def set_dyn_len(self, new_len):
        self._raw_idx = self._base_raw_idx[:new_len]

    def set_classes(self, cls_list):
        self._raw_labels = self._load_raw_labels()
        new_idcs = [self._raw_labels == cl for cl in cls_list]
        new_idcs = np.sum(np.vstack(new_idcs), 0)  # logical or
        new_idcs = np.where(new_idcs)  # find location
        self._raw_idx = self._base_raw_idx[new_idcs]
        assert all(sorted(cls_list) == np.unique(self._raw_labels[self._raw_idx]))
        print(f"Training on the following classes: {cls_list}")

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------


import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import imageio

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)


    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            raise Exception('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            raise Exception('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

class ImageFolderDatasetWithPreprocessing(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        
        self._base_resolution = resolution
        self.dataset_attrs=None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))

        image = self._preprocess(image, raw_idx)

        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image
    
    def _preprocess(self, image, idx):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

        transform = make_transform("center-crop", self._base_resolution, self._base_resolution)
        try:
            img = transform(image)
        except:
            raise Exception("Image %d failed." % idx)
            

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if self.dataset_attrs is None:
            self.dataset_attrs = cur_image_attrs
            width = self.dataset_attrs['width']
            height = self.dataset_attrs['height']
            if width != height:
                raise Exception(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if self.dataset_attrs['channels'] not in [1, 3]:
                raise Exception('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                raise Exception('Image width/height after scale and crop are required to be power-of-two')
        elif self.dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {self.dataset_attrs[k]}/{cur_image_attrs[k]}' for k in self.dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
            raise Exception(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        return img
    
    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels



def shard_dataset(dataset, rank, world_size, shuffle=False, seed=None):
    #   should work for any list-like object, not just datasets
    N = len(dataset)
    mindex = math.floor(rank / world_size * N)
    maxdex = math.floor((rank + 1) / world_size * N)
    if shuffle:
        rng = None
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
        indices = (torch.randperm(N, generator=rng)[mindex:maxdex]).tolist()
    else:
        indices = range(mindex, maxdex)
    
    if isinstance(dataset, Dataset):
        return Subset(dataset, indices)
    else:
        return [dataset[i] for i in indices]
    

class FixedRotationTransform():
    def __init__(self, angle, fill=0):
        self.angle = angle
        self.fill = fill

    def __call__(self, image):
        return TF.rotate(image, self.angle, fill=self.fill)

class HorizontalFlipTransform():
    def __call__(self, image):
        return TF.hflip(image)

class VerticalFlipTransform():
    def __call__(self, image):
        return TF.vflip(image)



def cycle_(iterable):
    # Creating custom cycle since itertools.cycle attempts to save all outputs in order to
    # re-cycle through them, creating amazing memory leak
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)



def wrap_dataset(dataset):
    if isinstance(dataset, IterableDataset):
        return cycle_(dataset)
    else:
        sampler = (dataset[i] for i in cycle_(RandomSampler(dataset)))
        return sampler



class ImageMultiSetGenerator():
    @classmethod
    def by_class(cls, dataset, n_classes, *args, **kwargs):
        classes = [[] for _ in range(n_classes)]
        for img, label in dataset:
            classes[label].append(img)
        return cls(classes, *args, **kwargs)

    @classmethod
    def splits_by_class(cls, dataset, n_classes, splits, *args, **kwargs):
        classes = [[] for _ in range(n_classes)]
        for img, label in dataset:
            classes[label].append(img)
        return [cls([classes[i] for i in split], *args, **kwargs) for split in splits]

    def __init__(self, datasets, flatten=False, transforms=None, data_augmentation=False, fill=None, load_all=False, rank=-1, world_size=-1):
        '''if world_size > 1:
            self.datasets = shard_dataset(datasets, rank, world_size, shuffle=True, seed=1)   #shard by class, not by example
        else:
            self.datasets = datasets
        if load_all:
            self.datasets = [[x for x in dataset] for dataset in self.datasets]'''
        self.datasets = datasets
        self.iters = [wrap_dataset(dataset) for dataset in datasets]
        self.n = len(self.datasets)
        self.min_elements = min([len(x) for x in self.datasets])
        self.flatten=flatten
        self.data_augmentation = data_augmentation
        self.fill = fill
        self.transforms=transforms
        self.rank = rank
        self.world_size = world_size

    def _augment_sets(self, image_sets, rng=None):
        transforms = []
        if torch.rand(1, generator=rng).item() < 0.5:
            if torch.rand(1, generator=rng).item() < 0.5:
                transforms.append(VerticalFlipTransform())
            else:
                transforms.append(HorizontalFlipTransform())
        if torch.rand(1, generator=rng).item() < 0.5:
            angle = torch.rand(1, generator=rng).item() * 360
            transforms.append(FixedRotationTransform(angle, fill=self.fill))
        transforms = torchvision.transforms.Compose(transforms)

        transformed_sets = [[transforms(img) for img in image_set] for image_set in image_sets]
        return transformed_sets


    def _build_sets_from_dataset(self, dataset_id, set_sizes, rng=None):
        def _get_image(data_iter):
            img = next(data_iter)
            if self.transforms is not None:
                img = self.transforms(img)
            if self.flatten:
                img = img.view(-1)
            return img

        dataset = self.datasets[dataset_id]
        data_iter = self.iters[dataset_id]
        assert sum(set_sizes) <= len(dataset), f'sum(set_sizes) = {sum(set_sizes)}, len(dataset) = {len(dataset)}'

        sets = []
        for set_size in set_sizes:
            sets_i = [_get_image(data_iter) for j in range(set_size)]
            sets.append(sets_i)
        
        if self.data_augmentation:
            sets = self._augment_sets(sets, rng=rng)

        sets = [torch.stack(x, dim=0) for x in sets]
        
        return sets
    
    def _build_sets_from_dataset2(self, dataset_id, set_sizes, rng=None):
        def _get_image(img):
            if self.transforms is not None:
                img = self.transforms(img)
            if self.flatten:
                img = img.view(-1)
            return img

        dataset = self.datasets[dataset_id]
        assert sum(set_sizes) <= len(dataset), f'sum(set_sizes) = {sum(set_sizes)}, len(dataset) = {len(dataset)}'

        p = torch.randperm(len(dataset), generator=rng)

        sets = []
        j=0
        for set_size in set_sizes:
            sets_i = [_get_image(dataset[idx]) for idx in p[j:j+set_size]]
            sets.append(sets_i)
            j += set_size
        
        if self.data_augmentation:
            sets = self._augment_sets(sets, rng=rng)

        sets = [torch.stack(x, dim=0) for x in sets]
        
        return sets
    
    def __call__(self, batch_size, set_sizes, contrastive=False, rng=None, class_id=None):
        dataset_inds = torch.randint(self.n, (batch_size,), generator=rng) if class_id is None else torch.ones(batch_size, dtype=torch.int)*class_id
        sets = []
        for i in range(batch_size):
            if rng is None:
                sets.append(self._build_sets_from_dataset(dataset_inds[i], set_sizes, rng=rng))
            else:
                sets.append(self._build_sets_from_dataset2(dataset_inds[i], set_sizes, rng=rng))

        sets_out = [torch.stack(set_i, dim=0) for set_i in zip(*sets)]

        if contrastive:
            contrastive_inds = torch.randint(self.n, (batch_size,), generator=rng)
            while contrastive_inds.eq(dataset_inds).any():
                eq_inds = contrastive_inds.eq(dataset_inds)
                contrastive_inds[eq_inds] = torch.randint(self.n, (batch_size,), generator=rng)[eq_inds]
            contrastive_sets = []
            for i in range(batch_size):
                sets.append(self._build_sets_from_dataset(contrastive_inds[i], set_sizes, rng=rng))
            
            contrastive_sets_out = [torch.stack(set_i, dim=0) for set_i in zip(*contrastive_sets)]

            return sets_out, contrastive_sets_out
        else:
            return sets_out