# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import dill
import psutil
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import pickle
from models.styleganxl.torch_utils import misc
from models.styleganxl.torch_utils import training_stats
from models.styleganxl.torch_utils.ops import conv2d_gradfix
from models.styleganxl.torch_utils.ops import grid_sample_gradfix

import models.styleganxl.legacy as legacy
#from metrics import metric_main

import setgan.safe_dataset as safe_dataset

from setgan.dataset import ImageMultiSetGenerator, ImagesDataset, build_datasets
from models.setgan.setgan import SetGAN

from setgan.metric_utils import ConditionalMetrics

STEP_INTERVAL=1000

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0, gw=None, gh=None):
    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = np.clip(7680 // training_set[0].image_shape[2], 7, 32)
    if gh is None:
        gh = np.clip(4320 // training_set[0].image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set[0].has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(reference_img, generated_img, fname, drange):
    lo, hi = drange

    # handle reference_img
    reference_img = np.asarray(reference_img, dtype=np.float32)
    reference_img = (reference_img- lo) * (255 / (hi - lo))
    reference_img = np.rint(reference_img).clip(0, 255).astype(np.uint8)

    N, M, C, H, W = reference_img.shape
    #reference_img = reference_img.reshape([gh, gw, C, H, W])
    reference_img = reference_img.transpose(0, 3, 1, 4, 2)
    reference_img = reference_img.reshape([N * H, M * W, C])

    # handle genearted_img
    generated_img = np.asarray(generated_img, dtype=np.float32)
    generated_img = (generated_img - lo) * (255 / (hi - lo))
    generated_img = np.rint(generated_img).clip(0, 255).astype(np.uint8)

    N, M, C, H, W = generated_img.shape
    #generated_img = generated_img.reshape([gh, gw, C, H, W])
    generated_img = generated_img.transpose(0, 3, 1, 4, 2)
    generated_img = generated_img.reshape([N * H, M * W, C])


    # Concatenate reference_img and generated_img along width axis
    final_image_grid = np.concatenate((reference_img, generated_img), axis=1)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(final_image_grid[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(final_image_grid, 'RGB').save(fname)

#----------------------------------------------------------------------------

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    dataset_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = 4,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    restart_every           = -1,       # Time interval in seconds to exit code
    reference_size          = (7,12),
    candidate_size          = (1,4),
    eval_metric             = 'fid-agg',
    step_interval           = STEP_INTERVAL
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    __RESTART__ = torch.tensor(0., device=device)       # will be broadcasted to exit loop
    __CUR_NIMG__ = torch.tensor(resume_kimg * step_interval, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __AUGMENT_P__ = torch.tensor(augment_p, dtype=torch.float, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    #training_set = safe_dataset.SafeDataset(dnnlib.util.construct_class_by_name(**dataset_kwargs)) # subclass of training.dataset.Dataset
    #training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    #training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    training_set, validation_set = build_datasets(**dataset_kwargs)
    training_set = [safe_dataset.SafeDataset(x) for x in training_set]
    validation_set = [safe_dataset.SafeDataset(x) for x in validation_set]
    training_set_generator = ImageMultiSetGenerator(training_set, rank=rank, world_size=num_gpus)
    validation_set_generator = ImageMultiSetGenerator(validation_set, rank=rank, world_size=num_gpus)
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set[0].image_shape)
        #print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=0, img_resolution=training_set[0].resolution, img_channels=training_set[0].num_channels)
    #G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G = SetGAN(G_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    with dnnlib.util.open_url(G_kwargs.stylegan_weights) as f:
        resume_data = legacy.load_network_pkl(f)
    D.load_weights(resume_data['D'])

    # Check for existing checkpoint
    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)


    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')

        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        if ckpt_pkl is not None:            # Load ticks
            __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
            __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
            __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
            __AUGMENT_P__ = resume_data['progress'].get('augment_p', torch.tensor(0.)).to(device)
            __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
            best_fid = resume_data['progress']['best_fid']       # only needed for rank == 0

    # this is relevant when you continue training a lower-res model
    # ie. train 16 model, start training 32 model but continue training 16 model
    # then restart 32 model to reload the improved 16 model
    if hasattr(G, 'reinit_stem'):
        G.reinit_stem()
        G_ema.reinit_stem()

    # Print network summary tables.
    '''
    if rank == 0:
        z = torch.empty([batch_gpu, G.decoder.z_dim], device=device)
        c = torch.empty([batch_gpu, G.decoder.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])
    '''

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, G_ema=G_ema, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    iter_dict = {'G': 1, 'D': 1}  # change here if you want to do several G/D iterations at once

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.

    
    if rank == 0:
        print('Exporting sample images...')

        reference_path = os.path.join(run_dir, "samples.pt")
        if os.path.exists(reference_path):
            samples_dict = torch.load(reference_path)
            sample_refs = samples_dict['reference_set']
            grid_s = samples_dict['s']
        else:
            # Generate reference sets (assuming set size of 5 for now)
            N = len(validation_set)
            sample_refs, = validation_set_generator(N, set_sizes=(5,), class_id=torch.arange(N))

            # Getting grid information and labels
            #grid_size, _, labels = setup_snapshot_image_grid(training_set=training_set)

            # Generate noise tensors
            grid_s = torch.randn([N, 5, G.decoder.z_dim], device=device)

            # Generate images based on reference set and noise tensors
            generated_images = [G_ema(ref_set.to(device), s).cpu() for ref_set, s in zip(sample_refs.split(batch_gpu), grid_s.split(batch_gpu))]
            generated_images = torch.cat(generated_images, dim=0)

            samples_path_init = os.path.join(run_dir, "fakes_init.png")
            save_image_grid(sample_refs, generated_images, samples_path_init, drange=[-1,1])

            torch.save({
                'reference_set': sample_refs,
                's': grid_s
            }, reference_path)
    

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)


    all_metrics = ConditionalMetrics(validation_set, dataset_name=dataset_kwargs['dataset_name'])
    all_metrics.add_split('base', reference_size=10, evaluation_size=100, generation_size=100, seed=0)

    for metric in metrics:
        all_metrics.add_metric(metric)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if num_gpus > 1:  # broadcast loaded states to all
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__AUGMENT_P__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()  # ensure all processes received this info
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // step_interval, total_kimg)
    augment_p = __AUGMENT_P__
    if augment_pipe is not None:
        augment_pipe.p.copy_(augment_p)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)
    while True:
        torch.cuda.empty_cache()
        with torch.autograd.profiler.record_function('data_fetch'):
            reference_samples = torch.randint(*reference_size, (1,))
            candidate_samples = torch.randint(*candidate_size, (1,))
            reference_set, candidate_set = training_set_generator(batch_size, set_sizes=(reference_samples, candidate_samples))

            # save reference set
            #save_image_grid(reference_set, os.path.join(run_dir, 'reference.png'), drange=[0,255], grid_size=grid_size)
            
            #phase_real_img, phase_real_c = next(training_set_iterator)
            phase_reference_set = (reference_set.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_candidate_set = (candidate_set.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            #phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_s = torch.randn([len(phases) * batch_size, candidate_samples, G.decoder.z_dim], device=device)
            all_gen_s = [phase_gen_s.split(batch_gpu) for phase_gen_s in all_gen_s.split(batch_size)]
            #all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            #all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            #all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_s in zip(phases, all_gen_s):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            ### PROJECTED GAN ADDITIONS ###
            if phase.name in ['Dmain', 'Dboth', 'Dreg'] and hasattr(phase.module, 'feature_networks'):
                phase.module.feature_networks.requires_grad_(False)

            for reference_set, candidate_set, gen_s in zip(phase_reference_set, phase_candidate_set, phase_gen_s):
                loss.accumulate_gradients(phase=phase.name, reference_set=reference_set, candidate_set=candidate_set, gen_s=gen_s, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * step_interval
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * step_interval)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * step_interval)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * step_interval):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / step_interval):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * step_interval):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Check for restart.
        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print('Restart job...')
            __RESTART__ = torch.tensor(1., device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f'Process {rank} leaving...')
            if num_gpus > 1:
                torch.distributed.barrier()

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            generated_images = [G_ema(ref_set.to(device), s).cpu() for ref_set, s in zip(sample_refs.split(batch_gpu), grid_s.split(batch_gpu))]
            generated_images = torch.cat(generated_images, dim=0)
            save_image_grid(sample_refs, generated_images, os.path.join(run_dir, f'fakes{cur_nimg//step_interval:06d}.png'), drange=[-1,1])

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    # value = copy.deepcopy(value).eval().requires_grad_(False)
                    # value = misc.spectral_to_cpu(value)
                    # if num_gpus > 1:
                    #     misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    #     for param in misc.params_and_buffers(value):
                    #         torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value #.cpu()
                del value # conserve memory

            # save for current time step (only for superres training, as we do not evaluate metrics here)
            if False:
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//step_interval:06d}.pkl')
                if rank == 0:
                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)

        # Save Checkpoint if needed
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (
                done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl = misc.get_ckpt_path(run_dir)
            # save as tensors to avoid error for multi GPU
            snapshot_data['progress'] = {
                'cur_nimg': torch.LongTensor([cur_nimg]),
                'cur_tick': torch.LongTensor([cur_tick]),
                'batch_idx': torch.LongTensor([batch_idx]),
                'best_fid': best_fid,
            }
            if augment_pipe is not None:
                snapshot_data['progress']['augment_p'] = augment_pipe.p.cpu()
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()

            with open(snapshot_pkl, 'wb') as f:
                dill.dump(snapshot_data, f)

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        
        if cur_tick and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in all_metrics.metrics:
                result_dict = all_metrics.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                                        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    all_metrics.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)

            # save best fid ckpt
            snapshot_pkl = os.path.join(run_dir, f'best_model.pkl')
            cur_nimg_txt = os.path.join(run_dir, f'best_nimg.txt')
            if rank == 0:
                if eval_metric in stats_metrics and stats_metrics[eval_metric] < best_fid:
                    best_fid = stats_metrics[eval_metric]

                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)
                    # save curr iteration number (directly saving it to pkl leads to problems with multi GPU)
                    with open(cur_nimg_txt, 'w') as f:
                        f.write(str(cur_nimg))
        

        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and \
                    not (phase.start_event.cuda_event == 0 and phase.end_event.cuda_event == 0):            # Both events were not initialized yet, can happen with restart
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / step_interval)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // step_interval, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
