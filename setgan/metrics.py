# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils


class Metric():
    def __init__(self, name, split):
        self.name = name
        self.split = split

    def __call__(*args):
        pass


class FID(Metric):
    def __init__(self, split, sfid=False, rfid=False, aggregate=False):
        super().__init__('fid', split)
        self.sfid = sfid
        self.rfid = rfid
        self.aggregate = aggregate

    def __call__(self, opts):
        # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
        if self.rfid:
            detector_url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/feature_networks/inception_rand_full.pkl'
            detector_kwargs = {}  # random inception network returns features by default

        stats_real = self.split.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, sfid=self.sfid)

        stats_gen = self.split.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, sfid=self.sfid)

        if opts.rank != 0:
            return float('nan')

        def _fid(mu_real, sigma_real, mu_gen, sigma_gen):
            m = np.square(mu_gen - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
            return float(fid)

        if self.aggregate:
            stats_real = stats_real.aggregate()
            stats_gen = stats_gen.aggregate()
            mu_real, sigma_real = stats_real.get_mean_cov()
            mu_gen, sigma_gen = stats_gen.get_mean_cov()

            score = _fid(mu_real, sigma_real, mu_gen, sigma_gen)
        else:
            mus_real, sigmas_real = stats_real.get_mean_cov()
            mus_gen, sigmas_gen = stats_gen.get_mean_cov()
            fids = [_fid(mu_r, sigma_r, mu_g, sigma_g) for mu_r, sigma_r, mu_g, sigma_g in zip(mus_real, sigmas_real, mus_gen, sigmas_gen)]

            score = sum(fids) / len(fids)

        return {'fid': score}




#----------------------------------------------------------------------------

def compute_fid(split, opts, sfid=False, rfid=False, aggregate=False):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.
    if rfid:
        detector_url = 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/feature_networks/inception_rand_full.pkl'
        detector_kwargs = {}  # random inception network returns features by default

    stats_real = split.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, sfid=sfid)

    stats_gen = split.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, sfid=sfid)

    if opts.rank != 0:
        return float('nan')

    def _fid(mu_real, sigma_real, mu_gen, sigma_gen):
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    if aggregate:
        stats_real = stats_real.aggregate()
        stats_gen = stats_gen.aggregate()
        mu_real, sigma_real = stats_real.get_mean_cov()
        mu_gen, sigma_gen = stats_gen.get_mean_cov()

        return _fid(mu_real, sigma_real, mu_gen, sigma_gen)
    else:
        mus_real, sigmas_real = stats_real.get_mean_cov()
        mus_gen, sigmas_gen = stats_gen.get_mean_cov()
        fids = [_fid(mu_r, sigma_r, mu_g, sigma_g) for mu_r, sigma_r, mu_g, sigma_g in zip(mus_real, sigmas_real, mus_gen, sigmas_gen)]

        return sum(fids) / len(fids)


#----------------------------------------------------------------------------

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


#----------------------------------------------------------------------------

'''
_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())



@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)
'''

#----------------------------------------------------------------------------

_metric_dict = {
    'fid-agg': {
        'metric_fct': FID,
        'kwargs': {'aggregate': True},
        'split_name': 'base'
    }
}

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())