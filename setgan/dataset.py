import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, random_split, RandomSampler
import torch

from src.setgan.utils import split_dataset

from src.meta_dataset.reader import Reader, parse_record
from src.meta_dataset.dataset_spec import Split
from src.meta_dataset import dataset_spec as dataset_spec_lib
from src.meta_dataset.transform import get_transforms


import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import cv2
import math
import os
import random


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



class ImagesDataset(Dataset):
    @classmethod
    def from_folder(cls, source_root, opts, transforms=None):
        source_paths = sorted(make_dataset(source_root))
        return cls(source_paths, opts, transforms)

    @classmethod
    def from_folder_by_category(cls, source_root, opts, transforms=None):
        source_paths = sorted(make_dataset(source_root))
        all_category_paths = {}
        for path in source_paths:
            cate = path.split('/')[-1].split('_')[0]
            if cate not in all_category_paths:
                all_category_paths[cate] = []
            all_category_paths[cate].append(path)
        return [cls(category_paths, opts, transforms) for category_paths in all_category_paths.values()]
    
    def __init__(self, source_paths, opts, transforms=None, store_in_memory=False):
        super().__init__()
        self.source_paths = source_paths
        self.transforms = transforms
        #self.average_codes = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
        self.opts = opts

        self.dataset = None
        if store_in_memory:
            self.dataset = [x for x in self]
            

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        if self.dataset is not None:
            im = self.dataset[index]
        else:
            path = self.source_paths[index]
            im = Image.open(path).convert('RGB')
            if self.transforms:
                im = self.transforms(im)

        return im


class MDDataset(IterableDataset):
    @classmethod
    def build_class_datasets(cls, dataset_path, split, transforms=None, min_class_examples=-1):
        #dataset_path = os.path.join(root_dir, dataset_name)
        dataset_spec = dataset_spec_lib.load_dataset_spec(dataset_path)
        reader = Reader(dataset_spec, split, False, 0) 
        class_datasets = reader.construct_class_datasets()
        split_classes = dataset_spec.get_classes(split)
        dataset_sizes = [dataset_spec.get_total_images_per_class(split_class) for split_class in split_classes]
        dataset_args = zip(class_datasets, dataset_sizes)
        if min_class_examples > 0:
            dataset_args = [(dataset, n) for (dataset, n) in dataset_args if n >= min_class_examples]
        
        return [cls(*dataset_args_i, transforms=transforms) for dataset_args_i in dataset_args]

    def __init__(self, dataset, n, transforms=None):
        self.dataset = dataset
        self.n = n
        self.transforms=transforms

    def __iter__(self):
        for sample_dic in self.dataset:
            sample_dic = parse_record(sample_dic)
            img = sample_dic['image']
            if self.transforms is not None:
                img = self.transforms(img)
            yield img

    def __len__(self):
        return self.n


class MergeDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.n = len(datasets)
        self.lens = [len(dataset) for dataset in self.datasets]
        self.length = sum(self.lens)

    def __iter__(self):
        n_remaining = [len(dataset) for dataset in self.datasets]
        iters = [iter(dataset) for dataset in self.datasets]
        n_total = sum(n_remaining)
        for _ in range(n_total):
            classid = random.choices(range(self.n), weights=n_remaining, k=1)[0]
            n_remaining[classid] -= 1
            yield next(iters[classid])



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