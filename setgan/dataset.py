import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, random_split, RandomSampler
import torch

from setgan.utils import split_dataset


import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import cv2
import math
import os
import random


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




"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

try:
    import pyspng
except ImportError:
    pyspng = None


class ImagesDataset(Dataset):
    @classmethod
    def from_folder(cls, path, resolution):
        source_paths = sorted(make_dataset(path))
        return cls(source_paths, resolution)

    @classmethod
    def from_folder_by_category(cls, path, resolution):
        source_paths = sorted(make_dataset(path))
        all_category_paths = {}
        for path in source_paths:
            cate = path.split('/')[-1].split('_')[0]
            if cate not in all_category_paths:
                all_category_paths[cate] = []
            all_category_paths[cate].append(path)
        return [cls(category_paths, resolution) for category_paths in all_category_paths.values()]
    
    def __init__(self, source_paths, resolution, store_in_memory=False):
        super().__init__()
        self.source_paths = source_paths
        self.resolution = self._base_resolution = resolution
        self.dataset_attrs=None

        self.dataset = None
        if store_in_memory:
            self.dataset = [x for x in self]

        test_img = self[0]
        self.image_shape = test_img.shape
        self.num_channels = self.image_shape[0]
            

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        if self.dataset is not None:
            im = self.dataset[index]
        else:
            path = self.source_paths[index]
            im = self._load_raw_image(path, index)

        return im
    
    def _load_raw_image(self, fname, idx):
        with open(fname, 'rb')as f:
            image = np.array(PIL.Image.open(f))

        image = self._preprocess(image, idx)

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
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
                
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