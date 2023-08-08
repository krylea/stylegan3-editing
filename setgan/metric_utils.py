import os
import random
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
from tqdm import tqdm

# PG additions
import contextlib
from pg_modules.projector import F_RandomProj
from pathlib import Path
import dill
from models.styleganxl.torch_utils import gen_utils
import setgan.safe_dataset

from torch.utils.data import Subset




class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStatsByClass(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj


class FeatureStatsByClass():
    def __init__(self, num_classes, capture_all=False, capture_mean_cov=False, max_items=None):
        self.num_classes = num_classes
        self.stats_by_class = [FeatureStats(capture_all=capture_all, capture_mean_cov=capture_mean_cov, max_items=max_items) for _ in range(num_classes)]

    def __getitem__(self, idx):
        return self.stats_by_class[i]

    def __len__(self):
        return self.num_classes
    
    def aggregate(self, capture_all=False, capture_mean_cov=False):
        stats = FeatureStats(capture_all=capture_all, capture_mean_cov=capture_mean_cov)

        for stats_cl in self.stats_by_class:
            stats.num_items += stats_cl.num_items

            if capture_all:
                stats.all_features += stats_cl.all_features

            if capture_mean_cov:
                stats.raw_mean += stats_cl.raw_mean
                stats.raw_cov += stats_cl.raw_cov
        
        return stats

    def get_mean_cov(self):
        return zip(*[stats.get_mean_cov for stats in self.stats_by_class])

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(s.num_classes, capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj



class Split():
    def __init__(self, reference_sets, evaluation_sets, generation_size, seed):
        assert len(reference_sets) == len(evaluation_sets)
        self.reference_sets = reference_sets
        self.evaluation_sets = evaluation_sets
        self.num_classes = len(reference_sets)
        self.reference_size = len(reference_sets[0])
        self.evaluation_size = len(evaluation_sets[0])
        self.generation_size = generation_size
        self.seed = seed

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = Split(s.reference_sets, s.evaluation_sets, s.reference_size, s.evaluation_size, s.seed)
        return obj

    def _class_feature_stats_for_generator(self, stats, reference_set, opts, G, detector, detector_kwargs, batch_size=64, batch_gen=None, **stats_kwargs):
        while not stats.is_full():
            images = []
            for _i in range(batch_size // batch_gen):
                s = gen_utils.get_w_from_seed(G, batch_gen, opts.device, **opts.G_kwargs)
                img = G(reference_set, s)

                img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                images.append(img)
            images = torch.cat(images)
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            with torch.no_grad():
                if opts.feature_network is None:
                    features = detector(images.to(opts.device), **detector_kwargs)
                    if sfid:
                        features = activation['mixed6_conv'][:, :7].flatten(1)
                else:
                    images = images.to(opts.device).to(torch.float32) / 127.5 - 1
                    features = detector(images)
                    features = torch.nn.AdaptiveAvgPool2d(1)(features['3']).squeeze()

            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

    def compute_feature_stats_for_generator(self, opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, sfid=False, **stats_kwargs):
        if batch_gen is None:
            batch_gen = min(batch_size, 4)
        assert batch_size % batch_gen == 0

        # Setup generator and labels.
        G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)

        # Initialize.
        all_stats = FeatureStatsByClass(self.num_classes, max_items=self.generation_size, **stats_kwargs)
        progress = opts.progress.sub(tag='generator features', num_items=self.num_classes, rel_lo=rel_lo, rel_hi=rel_hi)

        # get detector
        if opts.feature_network is not None:
            with contextlib.redirect_stdout(None):
                detector = F_RandomProj(opts.feature_network, proj_type=1).eval().to(opts.device)
                detector.proj_type = 0
        else:
            detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

        # Main loop.
        if sfid:
            detector.layers.mixed_6.conv.register_forward_hook(getActivation('mixed6_conv'))

        for cl in range(self.num_classes):
            reference_set = reference_sets[cl]
            reference_set = torch.stack([x for x in reference_set], dim=0).to(opts.device)

            self._class_feature_stats_for_generator(
                stats           = all_stats[cl],
                reference_set   = reference_set,
                opts            = opts,
                G               = G,
                detector        = detector,
                detector_kwargs = detector_kwargs,
                batch_size      = batch_size,
                batch_gen       = batch_gen,
                **stats_kwargs
            )

            progress.update(1)

        return all_stats

    def _class_feature_stats_for_dataset(self, stats, dataset, opts, detector, detector_kwargs, batch_size=64, data_loader_kwargs=None, shuffle_size=None, **stats_kwargs):
        item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
        if shuffle_size is not None:
            random.shuffle(item_subset)
            item_subset = item_subset[:shuffle_size]

        for images, _labels in tqdm(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)):
            if images.shape[1] == 1:
                images = images.repeat([1, 3, 1, 1])

            with torch.no_grad():
                if opts.feature_network is None:
                    features = detector(images.to(opts.device), **detector_kwargs)
                    if sfid:
                        features = activation['mixed6_conv'][:, :7].flatten(1)
                else:
                    images = images.to(opts.device).to(torch.float32) / 127.5 - 1
                    features = detector(images)
                    features = torch.nn.AdaptiveAvgPool2d(1)(features['3']).squeeze()

            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

    def compute_feature_stats_for_dataset(self, opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, sfid=False, shuffle_size=None, **stats_kwargs):
        if data_loader_kwargs is None:
            data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

        # Try to lookup from cache.
        cache_file = None
        if opts.cache:
            if opts.feature_network is not None:
                det_name = opts.feature_network
            else:
                det_name = get_feature_detector_name(detector_url)

            # Choose cache file name.
            dataset_name = Path(opts.dataset_kwargs.path).stem
            args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs, sfid=sfid, shuffle_size=shuffle_size)
            md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
            cache_tag = f'{dataset_name}-{det_name}-{md5.hexdigest()}'
            cache_file = os.path.join('.', 'dnnlib', 'gan-metrics', cache_tag + '.pkl')

            # Check if the file exists (all processes must agree).
            flag = os.path.isfile(cache_file) if opts.rank == 0 else False
            if opts.num_gpus > 1:
                flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
                torch.distributed.broadcast(tensor=flag, src=0)
                flag = (float(flag.cpu()) != 0)

            # Load.
            if flag:
                return FeatureStats.load(cache_file)

        #print('Calculating the stats for this dataset the first time\n')
        #print(f'Saving them to {cache_file}')
        #dataset = safe_dataset.SafeDataset(dnnlib.util.construct_class_by_name(**opts.dataset_kwargs))

        # Initialize.
        #num_items = len(dataset)
        #if max_items is not None:
        #    num_items = min(num_items, max_items)
        all_stats = FeatureStatsByClass(self.num_classes, max_items=self.evaluation_size, **stats_kwargs)
        progress = opts.progress.sub(tag='dataset features', num_items=self.num_classes, rel_lo=rel_lo, rel_hi=rel_hi)

        # get detector
        if opts.feature_network is not None:
            with contextlib.redirect_stdout(None):
                detector = F_RandomProj(opts.feature_network, proj_type=1).eval().to(opts.device)
                detector.proj_type = 0
        else:
            detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

        # Main loop.
        if sfid:
            detector.layers.mixed_6.conv.register_forward_hook(getActivation('mixed6_conv'))

        for cl in range(self.num_classes):
            dataset = self.evaluation_sets[cl]

            self._class_feature_stats_for_dataset(
                stats               = all_stats[cl],
                dataset             = dataset,
                opts                = opts,
                detector            = detector,
                detector_kwargs     = detector_kwargs,
                batch_size          = batch_size,
                data_loader_kwargs  = data_loader_kwargs,
                shuffle_size        = shuffle_size,
                **stats_kwargs
            )

            progress.update(1)

        # Save to cache.
        if cache_file is not None and opts.rank == 0:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = cache_file + '.' + uuid.uuid4().hex
            all_stats.save(temp_file)
            os.replace(temp_file, cache_file)  # atomic
        return all_stats


class ConditionalMetrics():
    def __init__(self, class_datasets):
        self.class_datasets = class_datasets
        self.num_classes = len(class_datasets)
        self.splits = {}
        self.metrics = {}

    def add_split(self, name, reference_size, evaluation_size, generation_size, seed=None):
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)

        reference_sets = []
        evaluation_sets = []
        for cl in self.class_datasets:
            N = len(cl)
            p = torch.randperm(N, generator=rng)
            assert reference_size + evaluation_size <= N
            if evaluation_size == -1: 
                evaluation_size = N - reference_size
            reference_sets.append(Subset(cl, p[:reference_size]))
            evaluation_sets.append(Subset(cl, p[reference_size:reference_size+evaluation_size]))
        
        self.splits[name] = Split(
            reference_sets      = reference_sets,
            evaluation_sets     = evaluation_sets,
            generation_size     = generation_size,
            seed                = seed
        )

    def add_metric(self, metric):
        assert is_valid_metric(metric)
        metric_info = _metric_dict[metric]
        metric_fct = metric_info['metric_fct'](self.splits[metric_info['split_name']], **metric_info['kwargs'])
        self.metrics[metric] = metric_fct

    def calc_metric(self, metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
        assert metric in self.metrics
        opts = metric_utils.MetricOptions(**kwargs)

        # Calculate.
        start_time = time.time()
        results = self.metrics[metric](opts)
        total_time = time.time() - start_time

        # Broadcast results.
        for key, value in list(results.items()):
            if opts.num_gpus > 1:
                value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
                torch.distributed.broadcast(tensor=value, src=0)
                value = float(value.cpu())
            results[key] = value

        # Decorate with metadata.
        return dnnlib.EasyDict(
            results         = dnnlib.EasyDict(results),
            metric          = metric,
            total_time      = total_time,
            total_time_str  = dnnlib.util.format_time(total_time),
            num_gpus        = opts.num_gpus,
        )

    def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
        metric = result_dict['metric']
        assert is_valid_metric(metric)
        if run_dir is not None and snapshot_pkl is not None:
            snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

        jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
        print(jsonl_line)
        if run_dir is not None and os.path.isdir(run_dir):
            with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
                f.write(jsonl_line + '\n')