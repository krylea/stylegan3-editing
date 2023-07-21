import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, random_split, RandomSampler
import torch

import zipfile

import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import cv2
import math
import os
import random

class Dataset():
    def __init__(self,
        path,                   # path to the zip file
        name,                   # Name of the dataset
        raw_labels = None,
        use_labels = True, 
        resolution      = None, # Ensure specific resolution, None = highest available.
    ):
        self._path = path
        self._name = name
        self._zipfile = None
        self._raw_labels = None
        self._use_labels = use_labels

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
    
    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if hasattr(self, '_zipfile') and self._zipfile is not None:
            return self._zipfile  # Already loaded, use existing
        else:
            self._zipfile = zipfile.ZipFile(self._path, 'r')  # Open the ZIP file
            return self._zipfile
    
    def _load_raw_labels(self):
        fname = 'images/dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[os.path.basename(fname.replace('\\', '/'))] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _get_raw_labels(self):
        if self._raw_labels is None:
            if self._use_labels == True:
                self._raw_labels = self._load_raw_labels()
            else:
                self._raw_labels = None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([len(self._image_fnames), 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            # assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = Image.open(f).convert('RGB')
        return np.array(image)

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None
    
    def splitfnames(self):
        all_labels = self._get_raw_labels()
        all_fnames = self._image_fnames
        unique_labels = np.unique(all_labels)
        num_labels = unique_labels.size
        fname_groups = []  # will contain a list of lists (second layer is for each class)
        for i in range(num_labels):
            lis = []
            fname_groups.append(lis)
        
        for j in range(len(all_fnames)):
            idx = np.where(unique_labels == all_labels[j])
            if idx[0].size == 0:
                print(f"No matching label found for {all_fnames[j]}")
            else:
                fname_groups[idx[0][0]].append(all_fnames[j])

        # next create objects for each class
        image_classes = []
        for k in range(num_labels):
            image_classes.append(ImageFolderDatasetWithPreprocessing(fname_groups[k]))
        
        return image_classes

class ImageFolderDatasetWithPreprocessing(torch.utils.data.Dataset):
    def __init__(self,
        fnames,                 # fnames for images of this class
        raw_shape = None,       # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip. is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 1,        # Random seed to use when applying max_size.
    ):
        self.fnames = fnames
        self._zipfile = None
        
        self.dataset_attrs=None
            
        raw_shape = [len(self.fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        # super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        
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
    
    def set_dyn_len(self, new_len):
        self._raw_idx = self._base_raw_idx[:new_len]

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    def __getstate__(self):
        return super().__getstate__()

    def _load_raw_image(self, raw_idx):
        fname = self.fnames[raw_idx]
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