# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import sys
import os
import click
import re
import json
import tempfile
import torch

sys.path.append(os.getcwd())
sys.path.append('models/styleganxl/')

import models.styleganxl.legacy as legacy

import dnnlib
import setgan.training_loop as training_loop
#from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops
from torch_utils import misc

from setgan.configs import dataset_paths, model_paths



#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------
def launch_training(c, exp_name, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    #prev_run_dirs = []
    #if os.path.isdir(outdir):
    #    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    c.run_dir = os.path.join(outdir, exp_name + "_" + str(c.G_kwargs.decoder_kwargs.img_resolution))

    '''
    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)
    '''
    
    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    #print(f'Dataset path:        {dataset_paths[c.dataset_kwargs.dataset_name]}')
    #print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.G_kwargs.decoder_kwargs.img_resolution}')
    #print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    #print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data, resolution):
    try:
        if 'imagenet' in data:
            dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDatasetWithPreprocessing", path=data, resolution=resolution, use_labels=True, max_size=None, xflip=False)
        else:
            dataset_kwargs = dnnlib.EasyDict(class_name="training.dataset.ImageFolderDataset", path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

def init_setgan_args(opts, c):
    assert opts.encoder_res >= opts.resolution or opts.encoder_res < 0

    c.dataset_kwargs = dnnlib.EasyDict(resolution=opts.resolution, dataset_name=opts.dataset_name)

    # Generator
    c.G_kwargs = dnnlib.EasyDict()
    c.G_kwargs.n_styles = opts.n_styles if opts.n_styles is not None else opts.syn_layers+2
    c.G_kwargs.latent = opts.g_latent
    c.G_kwargs.input_nc = 3 if opts.restyle_mode == 'none' else 6
    c.G_kwargs.n_heads = opts.g_attn_heads
    c.G_kwargs.attn_layers = opts.g_attn_layers
    c.G_kwargs.use_set_decoder = opts.use_set_decoder
    c.G_kwargs.disable_style_concat = opts.disable_style_concat
    c.G_kwargs.use_temperature = opts.use_temperature
    c.G_kwargs.restyle_mode = opts.restyle_mode
    c.G_kwargs.restyle_iters = opts.restyle_iters
    c.G_kwargs.freeze_encoder = opts.freeze_encoder
    c.G_kwargs.freeze_decoder = opts.freeze_decoder
    c.G_kwargs.mean_center = not opts.no_mean_center

    if opts.dataset_name == 'imagenet':
        c.G_kwargs.encoder_type='ResNetProgressiveBackboneEncoder'
        c.G_kwargs.encoder_kwargs = dnnlib.EasyDict(
            class_name='models.setgan.encoder.encoders.restyle_e4e_encoders.ResNetProgressiveBackboneEncoder',
            input_nc=c.G_kwargs.input_nc,
            n_styles=c.G_kwargs.n_styles 
        )
    else:
        c.G_kwargs.encoder_type='ProgressiveBackboneEncoder'
        c.G_kwargs.encoder_kwargs = dnnlib.EasyDict(
            class_name='models.setgan.encoder.encoders.restyle_e4e_encoders.ProgressiveBackboneEncoder',
            num_layers=50,
            mode='ir_se',
            input_nc=c.G_kwargs.input_nc,
            n_styles=c.G_kwargs.n_styles 
        )
    

    if opts.superres:
        c.G_kwargs.decoder_kwargs = dnnlib.EasyDict(
            class_name='models.styleganxl.training.networks_stylegan3_resetting.SuperresGenerator',
            path_stem=opts.path_stem,
            head_layers=opts.head_layers,
            up_factor=opts.up_factor,
        )
        c.G_kwargs.path_stem = opts.path_stem
    else:
        c.G_kwargs.decoder_kwargs = dnnlib.EasyDict(
            class_name='models.styleganxl.training.networks_stylegan3_resetting.Generator',
            channel_base=opts.cbase * 2,
            channel_max=opts.cmax*2,
            magnitude_ema_beta=0.5 ** (c.batch_size / (20 * 1e3)),
            conv_kernel=1 if opts.cfg == 'stylegan3-r' else 3,
            use_radial_filters = True if opts.cfg == 'stylegan3-r' else False
        )
        c.G_kwargs.path_stem = None
    c.G_kwargs.decoder_kwargs.w_dim = 512
    c.G_kwargs.decoder_kwargs.z_dim = 64
    c.G_kwargs.decoder_kwargs.mapping_kwargs=dnnlib.EasyDict()
    c.G_kwargs.decoder_kwargs.mapping_kwargs.rand_embedding = False
    c.G_kwargs.decoder_kwargs.num_layers = opts.syn_layers
    c.G_kwargs.decoder_kwargs.mapping_kwargs.num_layers = 2
    c.G_kwargs.decoder_kwargs.c_dim = 0
    c.G_kwargs.decoder_kwargs.img_resolution = opts.resolution
    c.G_kwargs.decoder_kwargs.img_channels = 3

    c.G_kwargs.use_pretrained = opts.use_pretrained
    if opts.use_pretrained:
        c.G_kwargs.encoder_ckpt = model_paths['stylegan_xl_%s_%d_encoder' % (opts.dataset_name, opts.resolution)]
        c.G_kwargs.decoder_ckpt = model_paths['stylegan_xl_%s_%d' % (opts.dataset_name, opts.resolution)]
    else:
        c.G_kwargs.encoder_ckpt = None
        c.G_kwargs.decoder_ckpt = None

    # Discriminator
    c.D_kwargs = dnnlib.EasyDict(
        class_name='models.setgan.discriminator.ProjectedSetDiscriminator',
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        diffaug=True,
        interp224=(c.dataset_kwargs.resolution < 224),
        backbone_res={'deit_base_distilled_patch16_224':4, 'tf_efficientnet_lite0':5},
        backbone_kwargs=dnnlib.EasyDict(),
        latent_size=opts.d_latent,
        set_kwargs=dnnlib.EasyDict(
            num_blocks=opts.d_attn_layers,
            num_heads=opts.d_attn_heads
        )
    )
    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2 
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name='setgan.loss.ProjectedSetGANLoss')
    c.loss_kwargs.blur_init_sigma = 2  # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = 300
    c.loss_kwargs.pl_weight = 2.0
    c.loss_kwargs.pl_no_weight_grad = True
    c.loss_kwargs.style_mixing_prob = 0.0
    c.loss_kwargs.cls_weight = 0.0  # use classifier guidance only for superresolution training (i.e., with pretrained stem)
    c.loss_kwargs.cls_model = 'deit_small_distilled_patch16_224'
    c.loss_kwargs.train_head_only = False
    
    if opts.encoder_res > 0:
        assert opts.encoder_res > opts.resolution
        c.downsample_res = opts.resolution
        c.G_kwargs.decoder_kwargs.img_resolution = opts.resolution
        c.dataset_kwargs.resolution = opts.encoder_res
        

def init_sgxl_args(opts, c):
    c.dataset_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, resolution=opts.resolution)
    c.dataset_kwargs = dnnlib.EasyDict(resolution=opts.resolution, dataset_name=opts.dataset_name)
    if opts.cond and not c.dataset_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.dataset_kwargs.use_labels = opts.cond
    c.dataset_kwargs.xflip = opts.mirror

    if opts.dataset_name is None:
        opts.dataset_name = dataset_name

    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.class_name = 'training.networks_stylegan3_resetting.Generator'
    c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
    c.G_kwargs.channel_base *= 2  # increase for StyleGAN-XL
    c.G_kwargs.channel_max *= 2   # increase for StyleGAN-XL
    c.G_kwargs.conv_kernel = 1 if opts.cfg == 'stylegan3-r' else 3
    c.G_kwargs.use_radial_filters = True if opts.cfg == 'stylegan3-r' else False



    if opts.cfg == 'stylegan3-r':
        c.G_kwargs.channel_base *= 2
        c.G_kwargs.channel_max *= 2

    c.D_kwargs = dnnlib.EasyDict(
        class_name='models.styleganxl.pg_modules.discriminator.ProjectedDiscriminator',
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        diffaug=True,
        latent_size=opts.latent,
        interp224=(c.dataset_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )
    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2 
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Loss
    c.loss_kwargs = dnnlib.EasyDict(class_name='setgan.loss.ProjectedSetGANLoss')
    c.loss_kwargs.blur_init_sigma = 2  # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = 300
    c.loss_kwargs.pl_weight = 2.0
    c.loss_kwargs.pl_no_weight_grad = True
    c.loss_kwargs.style_mixing_prob = 0.0
    c.loss_kwargs.cls_weight = 0.0  # use classifier guidance only for superresolution training (i.e., with pretrained stem)
    c.loss_kwargs.cls_model = 'deit_small_distilled_patch16_224'
    c.loss_kwargs.train_head_only = False








#----------------------------------------------------------------------------

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2', 'fastgan']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1))

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid-agg', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=0), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# StyleGAN-XL additions
@click.option('--restart_every',help='Time interval in seconds to restart code', metavar='INT', type=int, default=999999999, show_default=True)
@click.option('--stem',         help='Train the stem.', is_flag=True)
@click.option('--syn_layers',   help='Number of layers in the stem', type=click.IntRange(min=1), default=14, show_default=True)
@click.option('--superres',     help='Train superresolution stage. You have to provide the path to a pretrained stem.', is_flag=True)
@click.option('--path_stem',    help='Path to pretrained stem',  type=str)
@click.option('--head_layers',  help='Layers of added superresolution head.', type=click.IntRange(min=1), default=7, show_default=True)
@click.option('--cls_weight',   help='class guidance weight', type=float, default=0.0, show_default=True)
@click.option('--up_factor',    help='Up sampling factor of superres head', type=click.IntRange(min=2), default=2, show_default=True)
@click.option('--resolution',    help='Image resolution', type=click.IntRange(min=1))
@click.option('--dataset_name',    help='Dataset name', type=str)

# SetGAN
@click.option('--input_nc', type=int, default=3)
@click.option('--g_latent', type=int, default=512)
@click.option('--d_latent', type=int, default=512)
@click.option('--n_styles', type=int, default=None)
@click.option('--g_attn_heads', type=int, default=8)
@click.option('--d_attn_heads', type=int, default=8)
@click.option('--g_attn_layers', type=int, default=4)
@click.option('--d_attn_layers', type=int, default=4)
@click.option('--use_set_decoder', is_flag=True)
@click.option('--disable_style_concat', is_flag=True)
@click.option('--use_temperature', is_flag=True)
@click.option('--encoder_type', type=str, default='ResNetProgressiveBackboneEncoder')
@click.option('--resume_ckpt', type=str)
@click.option('--train_encoder', is_flag=True)
@click.option('--train_decoder', is_flag=True)
@click.option('--exp_name', type=str, required=True)
@click.option('--reference_size', type=int, nargs=2, default=(7,12))
@click.option('--candidate_size', type=int, nargs=2, default=(1,4))
@click.option('--restyle_mode', type=click.Choice(['none', 'encoder', 'resetgan', 'resetgan2']), default='encoder')
@click.option('--restyle_iters', type=int, default=3)
@click.option('--use_setgan', is_flag=True)
@click.option('--eval_metric', type=str, default='fid-agg')
@click.option('--step_interval', type=int, default=1000)
@click.option('--freeze_encoder', is_flag=True)
@click.option('--freeze_decoder', is_flag=True)
@click.option('--use_pretrained', is_flag=True)
@click.option('--no_mean_center', is_flag=True)
@click.option('--encoder_res', type=int, default=-1)
@click.option('--warmup_steps', type=int, default=-1)

def main(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)  # Command line arguments
    c = dnnlib.EasyDict()  # Main config dict.
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    #c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, resolution=opts.resolution)
    
    #if opts.cond and not c.training_set_kwargs.use_labels:
    #    raise click.ClickException('--cond=True requires labels specified in dataset.json')
    #c.training_set_kwargs.use_labels = opts.cond
    #c.training_set_kwargs.xflip = opts.mirror

    #if opts.dataset_name is None:
    #    opts.dataset_name = dataset_name

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus

    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.eval_metric = opts.eval_metric
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.eval_ticks = opts.snap
    c.network_snapshot_ticks = 1
    c.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    c.reference_size = opts.reference_size
    c.candidate_size = opts.candidate_size

    c.step_interval = opts.step_interval
    c.warmup_steps = opts.warmup_steps


    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    #if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
    #    raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        #c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.G_reg_interval = 4  # Enable lazy regularization for G.
        #c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.

    elif opts.cfg == 'fastgan':
        #c.G_kwargs = dnnlib.EasyDict(class_name='training.networks_fastgan.Generator',
        #                             cond=opts.cond, mapping_kwargs=dnnlib.EasyDict(),
        #                             synthesis_kwargs=dnnlib.EasyDict())
        #c.G_kwargs.synthesis_kwargs.lite = True
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        c.G_opt_kwargs.lr = 0.002

    #else:
        #c.G_kwargs.class_name = 'training.networks_stylegan3_resetting.Generator'
        #c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        #c.G_kwargs.channel_base *= 2  # increase for StyleGAN-XL
        #c.G_kwargs.channel_max *= 2   # increase for StyleGAN-XL
        #c.G_kwargs.conv_kernel = 1 if opts.cfg == 'stylegan3-r' else 3
        #c.G_kwargs.use_radial_filters = True if opts.cfg == 'stylegan3-r' else False

        #if opts.cfg == 'stylegan3-r':
        #    c.G_kwargs.channel_base *= 2
        #    c.G_kwargs.channel_max *= 2
        
    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100  # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.

    # Restart.
    c.restart_every = opts.restart_every

    # Performance-related toggles.
    if opts.fp32:
        #c.G_kwargs.num_fp16_res = 0
        #c.G_kwargs.conv_clamp = None
        pass
    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{opts.dataset_name:s}{opts.resolution:d}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    ##################################
    ########## StyleGAN-XL ###########
    ##################################

    if opts.use_setgan:
        init_setgan_args(opts, c)
    else:
        init_sgxl_args(opts, c)


    if opts.superres:
        assert opts.path_stem is not None, "When training superres head, provide path to stem"

        # Loss
        c.loss_kwargs.pl_weight = 0.0
        c.loss_kwargs.cls_weight = opts.cls_weight if opts.cond else 0
        c.loss_kwargs.train_head_only = True

    ##################################
    ##################################
    ##################################

    # Launch.
    launch_training(c=c, exp_name=opts.exp_name, outdir=opts.outdir, dry_run=opts.dry_run)

    # Check for restart
    last_snapshot = misc.get_ckpt_path(c.run_dir)
    if os.path.isfile(last_snapshot):
        # get current number of training images
        with dnnlib.util.open_url(last_snapshot) as f:
            cur_nimg = legacy.load_network_pkl(f)['progress']['cur_nimg'].item()
        if (cur_nimg//1000) < c.total_kimg:
            print('Restart: exit with code 3')
            exit(3)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
