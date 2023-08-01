# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch
import math

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

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

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
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

def init_dataset_kwargs(data, class_name='training.dataset.ImageFolderDataset', kwargs = {}):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name=class_name, path=data, use_labels=True, max_size=None, xflip=False, **kwargs)
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

@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--cfg',          help='Base configuration',                                      type=str, required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]',                      type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT',                    type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                         type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=True)

# Model features
@click.option('--recon-gamma',  help='R1 regularization weight', metavar='FLOAT',               type=click.FloatRange(min=0), required=False, default=None)
@click.option('--use-semantic-loss', help='Use semantic loss in discriminator',                 type=bool, required=False, default=False)
@click.option('--use-semantic-floor', help='Compare semantic segmentation against scene layout',                 type=bool, required=False, default=False)
@click.option('--remove-classes',      help='Remove classes from dataset',         type=parse_comma_separated_list, default=None, show_default=True)
@click.option('--camera-y-noise', help='Add noise to camera height',                 type=float, required=False, default=None)
@click.option('--fov-noise', help='Add noise to FOV',                 type=float, required=False, default=None)
@click.option('--optimize-camera-coords', help='Learn camera coordinates',                 type=bool, required=False, default=False)
@click.option('--dual-discrimination',         help='Use dual discriminator', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--semantic-resolution', type=int, required=False, default=32)
@click.option('--semantic-gamma', metavar='FLOAT',               type=click.FloatRange(min=0), required=False, default=1.0)
@click.option('--feature-resolution', type=int, required=False, default=128)
@click.option('--semantic-floor-type', type=str, required=False, default='floor')
@click.option('--compute-alphas', type=bool, required=False, default=False)
@click.option('--alpha-gamma', metavar='FLOAT',               type=click.FloatRange(min=0), required=False, default=1.0)
@click.option('--unconditional', type=bool, required=False, default=False)
@click.option('--n_hidden_layers_mlp', type=int, required=False, default=1)
@click.option('--hidden_decoder_mlp_dim', type=int, required=False, default=64)
@click.option('--img_pairs', type=bool, required=False, default=False)
@click.option('--flip_pair', type=bool, required=False, default=False)
@click.option('--double_discrimination', help='Use video and image discriminator', type=bool, default=False, show_default=True)
@click.option('--pairs_conv_fuse_inter_channels', type=int, required=False, default=64)
@click.option('--pairs_conv_fuse_num_layers', type=int, required=False, default=2)
@click.option('--cond_vid_fake_prob', help='', type=float, required=False, default=1.0)
@click.option('--cond_vid_const_prob_init', help='', type=float, required=False, default=1.0)
@click.option('--cond_vid_const_prob_end', help='', type=float, required=False, default=0.5)
@click.option('--cond_vid_const_anneal_kimg', help='', type=int, required=False, default=1000)
@click.option('--cond_vid_const_anneal_kimg_start', help='', type=int, required=False, default=0)
@click.option('--cond_vid_const_anneal', help='', type=bool, required=False, default=False)
@click.option('--pairs_conv_fuse_type', help='', type=click.Choice(['concat_norm', 'conv_fuse_1', 'conv_fuse_0']), required=False, default='concat_norm')
@click.option('--cond_vid_const_type', help='', type=click.Choice(['zeros', 'same']), required=False, default='zeros')
@click.option('--out_sigmoid', help='', type=bool, required=False, default=False)
@click.option('--cond_vid_const_aug', help='', type=bool, required=False, default=True)
@click.option('--use_out_extra_conv', help='', type=bool, required=False, default=False)
@click.option('--super_res_shared', help='', type=bool, required=False, default=False)
@click.option('--concat_floor_height', help='', type=bool, required=False, default=False)
@click.option('--conv_head_mod', help='', type=bool, required=False, default=False)
@click.option('--use_traj', help='', type=bool, required=False, default=False)
@click.option('--use_obj_latent', help='', type=bool, required=False, default=False)
@click.option('--encoder_mod', help='', type=bool, required=False, default=False)
@click.option('--use_layout_latent', help='', type=bool, required=False, default=False)
@click.option('--z_dim_obj', help='', type=int, required=False, default=512)
@click.option('--dataset_name', help='Name of dataset', type=click.Choice(['3dfront', 'kitti']), required=False, default="3dfront")
@click.option('--embed_semantic_layout', help='', type=bool, required=False, default=False)
@click.option('--add_floor_class', help='', type=bool, required=False, default=False)
@click.option('--add_none_class', help='', type=bool, required=False, default=False)
@click.option('--semantic_top_down_direct', help='', type=bool, required=False, default=False)

# Optional features.
@click.option('--cond',         help='Train conditional model', metavar='BOOL',                 type=bool, default=True, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode',                                       type=click.Choice(['noaug', 'ada', 'fixed', 'diff', 'xflip']), default='diff', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]',  type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT',                 type=click.IntRange(min=0), default=0, show_default=True)

# Misc hyperparameters.
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT',            type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT',             type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',                 type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT',                      type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT',                        type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate  [default: varies]', metavar='FLOAT',     type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT',                 type=click.IntRange(min=1), default=4, show_default=True)

# Misc settings.
@click.option('--desc',         help='String to include in result dir name', metavar='STR',     type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG',                 type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',             type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',            type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT',                              type=click.IntRange(min=0), default=0, show_default=True)
# @click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL',                 type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT',              type=click.IntRange(min=0), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit',                         is_flag=True)

# @click.option('--sr_module',    help='Superresolution module', metavar='STR',  type=str, required=True)
@click.option('--neural_rendering_resolution', help='Resolution to render at', metavar='INT',  type=click.IntRange(min=1), default=64, required=False)

@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT',  type=click.IntRange(min=0), required=False, default=200)
@click.option('--gen_pose_cond', help='If true, enable generator pose conditioning.', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--c-scale', help='Scale factor for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=1)
@click.option('--c-noise', help='Add noise for generator pose conditioning.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--gpc_reg_prob', help='Strength of swapping regularization. None means no generator pose conditioning, i.e. condition with zeros.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0.5)
@click.option('--gpc_reg_fade_kimg', help='Length of swapping prob fade', metavar='INT',  type=click.IntRange(min=0), required=False, default=1000)
@click.option('--disc_c_noise', help='Strength of discriminator pose conditioning regularization, in standard deviations.', metavar='FLOAT',  type=click.FloatRange(min=0), required=False, default=0)
@click.option('--sr_noise_mode', help='Type of noise for superresolution', metavar='STR',  type=click.Choice(['random', 'none']), required=False, default='none')
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL',  type=bool, required=False, default=False)
@click.option('--sr_num_fp16_res',    help='Number of fp16 layers in superresolution', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--g_num_fp16_res',    help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res',    help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=4, required=False, show_default=True)
@click.option('--sr_first_cutoff',    help='First cutoff for AF superresolution', metavar='INT', type=click.IntRange(min=2), default=2, required=False, show_default=True)
@click.option('--sr_first_stopband',    help='First cutoff for AF superresolution', metavar='FLOAT', type=click.FloatRange(min=2), default=2**2.1, required=False, show_default=True)
@click.option('--style_mixing_prob',    help='Style-mixing regularization probability for training.', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0, required=False, show_default=True)
@click.option('--sr-module',    help='Superresolution module override', metavar='STR',  type=str, required=False, default=None)
@click.option('--density_reg',    help='Density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.0, required=False, show_default=True)
@click.option('--density_reg_every',    help='lazy density reg', metavar='int', type=click.FloatRange(min=1), default=4, required=False, show_default=True)
@click.option('--density_reg_p_dist',    help='density regularization strength.', metavar='FLOAT', type=click.FloatRange(min=0), default=0.004, required=False, show_default=True)
@click.option('--reg_type', help='Type of regularization', metavar='STR',  type=click.Choice(['l1', 'l1-alt', 'monotonic-detach', 'monotonic-fixed', 'total-variation']), required=False, default='l1')
@click.option('--decoder_lr_mul',    help='decoder learning rate multiplier.', metavar='FLOAT', type=click.FloatRange(min=0), default=1, required=False, show_default=True)

def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror
    c.training_set_kwargs.remove_classes = opts.remove_classes

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.recon_gamma = opts.recon_gamma
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size:
        raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    c.G_kwargs.class_name = 'training.generator.Generator'
    c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    c.loss_kwargs.filter_mode = 'antialiased' # Filter mode for raw images ['antialiased', 'none', float [0-1]]
    # c.D_kwargs.disc_c_noise = opts.disc_c_noise # Regularization for discriminator pose conditioning

    if c.training_set_kwargs.resolution == 512:
        sr_module = 'training.superresolution.SuperresolutionHybrid8XDC'
    elif c.training_set_kwargs.resolution == 256:
        sr_module = 'training.superresolution.SuperresolutionHybrid4X'
    elif c.training_set_kwargs.resolution == 128:
        sr_module = 'training.superresolution.SuperresolutionHybrid2X'
    else:
        assert False, f"Unsupported resolution {c.training_set_kwargs.resolution}; make a new superresolution module"
    
    if opts.sr_module != None:
        sr_module = opts.sr_module
    
    # Extra features
    c.G_kwargs.camera_y_noise = opts.camera_y_noise
    c.G_kwargs.fov_noise = opts.fov_noise
    c.training_set_kwargs.optimize_camera_coords = opts.optimize_camera_coords
    c.loss_kwargs.semantic_gamma = opts.semantic_gamma
    c.G_kwargs.semantic_floor_type = opts.semantic_floor_type
    c.G_kwargs.compute_alphas = opts.compute_alphas
    c.G_kwargs.alpha_gamma = opts.alpha_gamma
    c.G_kwargs.unconditional = opts.unconditional
    c.G_kwargs.hidden_decoder_mlp_dim = opts.hidden_decoder_mlp_dim
    c.G_kwargs.n_hidden_layers_mlp = opts.n_hidden_layers_mlp
    c.G_kwargs.img_pairs = opts.img_pairs
    c.D_kwargs.img_pairs = opts.img_pairs
    c.loss_kwargs.img_pairs = opts.img_pairs
    c.training_set_kwargs.img_pairs = opts.img_pairs
    c.training_set_kwargs.flip_pair = opts.flip_pair
    c.loss_kwargs.double_discrimination = opts.double_discrimination
    c.G_kwargs.pairs_conv_fuse_inter_channels = opts.pairs_conv_fuse_inter_channels
    c.G_kwargs.pairs_conv_fuse_num_layers = opts.pairs_conv_fuse_num_layers
    c.loss_kwargs.cond_vid_fake_prob = opts.cond_vid_fake_prob
    c.loss_kwargs.cond_vid_const_anneal = opts.cond_vid_const_anneal
    c.loss_kwargs.cond_vid_const_prob_init = opts.cond_vid_const_prob_init
    c.loss_kwargs.cond_vid_const_prob_end = opts.cond_vid_const_prob_end
    c.loss_kwargs.cond_vid_const_anneal_kimg = opts.cond_vid_const_anneal_kimg
    c.loss_kwargs.cond_vid_const_anneal_kimg_start = opts.cond_vid_const_anneal_kimg_start
    c.G_kwargs.pairs_conv_fuse_type = opts.pairs_conv_fuse_type
    c.G_kwargs.out_sigmoid = opts.out_sigmoid
    c.loss_kwargs.cond_vid_const_type = opts.cond_vid_const_type
    c.G_kwargs.cond_vid_const_type = opts.cond_vid_const_type
    c.loss_kwargs.cond_vid_const_aug = opts.cond_vid_const_aug
    c.G_kwargs.use_out_extra_conv = opts.use_out_extra_conv
    c.G_kwargs.super_res_shared = opts.super_res_shared
    c.G_kwargs.concat_floor_height = opts.concat_floor_height
    c.G_kwargs.conv_head_mod = opts.conv_head_mod
    c.training_set_kwargs.use_traj = opts.use_traj
    c.G_kwargs.use_obj_latent = opts.use_obj_latent
    c.G_kwargs.encoder_mod = opts.encoder_mod
    c.G_kwargs.use_layout_latent = opts.use_layout_latent
    c.G_kwargs.z_dim_obj = opts.z_dim_obj
    c.G_kwargs.add_floor_class = opts.add_floor_class
    c.training_set_kwargs.add_floor_class = opts.add_floor_class
    c.G_kwargs.add_none_class = opts.add_none_class
    c.training_set_kwargs.add_none_class = opts.add_none_class
    c.G_kwargs.semantic_top_down_direct = opts.semantic_top_down_direct
    
    rendering_options = {
        'image_resolution': c.training_set_kwargs.resolution,
        'disparity_space_sampling': False,
        'clamp_mode': 'softplus',
        'superresolution_module': sr_module,
        'c_gen_conditioning_zero': not opts.gen_pose_cond, # if true, fill generator pose conditioning label with dummy zero vector
        'gpc_reg_prob': opts.gpc_reg_prob if opts.gen_pose_cond else None,
        'c_scale': opts.c_scale, # mutliplier for generator pose conditioning label
        'superresolution_noise_mode': opts.sr_noise_mode, # [random or none], whether to inject pixel noise into super-resolution layers
        'density_reg': opts.density_reg, # strength of density regularization
        'density_reg_p_dist': opts.density_reg_p_dist, # distance at which to sample perturbed points for density regularization
        'reg_type': opts.reg_type, # for experimenting with variations on density regularization
        'decoder_lr_mul': opts.decoder_lr_mul, # learning rate multiplier for decoder
        'sr_antialias': True,
    }
    if opts.dual_discrimination:
        if opts.double_discrimination:
            c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminatorVideo'
        else:
            c.D_kwargs.class_name = 'training.dual_discriminator.DualDiscriminator'
    else:
        if opts.double_discrimination:
            raise NotImplementedError
        else:
            c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
    c.training_set_kwargs.top_data = False
    c.training_set_kwargs.dataset_name = opts.dataset_name
    c.training_set_kwargs.num_classes = None
    if opts.dataset_name == '3dfront':
        c.G_kwargs.cond_content = ['class_one_hot_embed', 'local_coords_canonical', 'room_layout']
    elif opts.dataset_name == 'kitti':
        c.training_set_kwargs.num_classes = 59
        if opts.embed_semantic_layout:
            c.G_kwargs.cond_content = ['semantic_layout_embed']
        else:
            c.G_kwargs.cond_content = ['semantic_layout']
    if opts.cfg == '3dfront':
        # c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
        c.G_kwargs.grid_size = 32
        c.G_kwargs.neural_rendering_resolution = opts.neural_rendering_resolution
        c.G_kwargs.feature_resolution = opts.feature_resolution
        c.G_kwargs.model_type = '3D'
        c.G_kwargs.feature_type = 'volume'
        c.G_kwargs.voxel_feat_dim = 32
        c.G_kwargs.decoder_output_dim = 32
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 0.0, # near point along each ray to start taking samples.
            'ray_end': math.sqrt(2)*2, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'white_back': True,
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == '3dfront_2d':
        # c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
        c.training_set_kwargs.top_data = True
        c.G_kwargs.grid_size = opts.feature_resolution
        c.G_kwargs.feature_resolution = opts.feature_resolution
        c.G_kwargs.neural_rendering_resolution = None
        c.G_kwargs.model_type = '2D'
        c.G_kwargs.feature_type = None
        c.G_kwargs.voxel_feat_dim = 3
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 'auto', # near point along each ray to start taking samples.
            'ray_end': 'auto', # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'white_back': True,
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == '3dfront_2d_volume':
        # c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
        c.G_kwargs.grid_size = opts.feature_resolution
        c.G_kwargs.neural_rendering_resolution = opts.neural_rendering_resolution
        c.G_kwargs.feature_resolution = opts.feature_resolution
        c.G_kwargs.model_type = '2D_volume'
        c.G_kwargs.feature_type = 'volume'
        c.G_kwargs.voxel_feat_dim = opts.feature_resolution
        c.G_kwargs.decoder_head_dim = 32
        c.G_kwargs.decoder_output_dim = 32
        c.G_kwargs.num_semantic_y = 8
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 0.0, # near point along each ray to start taking samples.
            'ray_end': math.sqrt(2)*2, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'white_back': True,
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == '3dfront_2d_floorplan':
        # c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
        c.G_kwargs.grid_size = opts.feature_resolution
        c.G_kwargs.neural_rendering_resolution = opts.neural_rendering_resolution
        c.G_kwargs.feature_resolution = opts.feature_resolution
        c.G_kwargs.feature_type = 'floor_plan'
        c.G_kwargs.model_type = '2D'
        c.G_kwargs.voxel_feat_dim = opts.feature_resolution #c.G_kwargs.grid_size  * 32
        c.G_kwargs.decoder_head_dim = 32
        c.G_kwargs.decoder_output_dim = 32
        c.G_kwargs.num_semantic_y = 1
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 0.0, # near point along each ray to start taking samples.
            'ray_end': math.sqrt(2)*2, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'white_back': True,
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    elif opts.cfg == '3dfront_2d_triplane':
        # c.D_kwargs.class_name = 'training.dual_discriminator.SingleDiscriminator'
        c.G_kwargs.grid_size = opts.feature_resolution
        c.G_kwargs.neural_rendering_resolution = opts.neural_rendering_resolution
        c.G_kwargs.feature_resolution = opts.feature_resolution
        c.G_kwargs.feature_type = 'planes'
        c.G_kwargs.model_type = '2D'
        c.G_kwargs.voxel_feat_dim = 96 #c.G_kwargs.grid_size  * 32
        c.G_kwargs.decoder_head_dim = 32
        c.G_kwargs.decoder_output_dim = 32
        rendering_options.update({
            'depth_resolution': 48, # number of uniform samples to take per ray.
            'depth_resolution_importance': 48, # number of importance samples to take per ray.
            'ray_start': 0.0, # near point along each ray to start taking samples.
            'ray_end': math.sqrt(2)*2, # far point along each ray to stop taking samples. 
            'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
            'white_back': True,
            'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
            'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
        })
    # elif opts.cfg == 'ffhq':
    #     rendering_options.update({
    #         'depth_resolution': 48, # number of uniform samples to take per ray.
    #         'depth_resolution_importance': 48, # number of importance samples to take per ray.
    #         'ray_start': 2.25, # near point along each ray to start taking samples.
    #         'ray_end': 3.3, # far point along each ray to stop taking samples. 
    #         'box_warp': 1, # the side-length of the bounding box spanned by the tri-planes; box_warp=1 means [-0.5, -0.5, -0.5] -> [0.5, 0.5, 0.5].
    #         'avg_camera_radius': 2.7, # used only in the visualizer to specify camera orbit radius.
    #         'avg_camera_pivot': [0, 0, 0.2], # used only in the visualizer to control center of camera rotation.
    #     })
    # elif opts.cfg == 'afhq':
    #     rendering_options.update({
    #         'depth_resolution': 48,
    #         'depth_resolution_importance': 48,
    #         'ray_start': 2.25,
    #         'ray_end': 3.3,
    #         'box_warp': 1,
    #         'avg_camera_radius': 2.7,
    #         'avg_camera_pivot': [0, 0, -0.06],
    #     })
    # elif opts.cfg == 'shapenet':
    #     rendering_options.update({
    #         'depth_resolution': 64,
    #         'depth_resolution_importance': 64,
    #         'ray_start': 0.1,
    #         'ray_end': 2.6,
    #         'box_warp': 1.6,
    #         'white_back': True,
    #         'avg_camera_radius': 1.7,
    #         'avg_camera_pivot': [0, 0, 0],
    #     })
    else:
        assert False, "Need to specify config"
    c.loss_kwargs.neural_rendering_resolution = opts.neural_rendering_resolution
    c.training_set_kwargs.model_type = c.G_kwargs.model_type
    # Use semantic loss
    c.G_kwargs.use_semantic_loss = opts.use_semantic_loss
    c.G_kwargs.use_semantic_floor = opts.use_semantic_floor
    c.D_kwargs.use_semantic_loss = opts.use_semantic_loss
    c.D_kwargs.use_semantic_floor = opts.use_semantic_floor
    c.loss_kwargs.use_semantic_loss = opts.use_semantic_loss
    if opts.use_semantic_floor:
        # semantic_resolution = c.G_kwargs.feature_resolution
        semantic_resolution = opts.semantic_resolution
        c.D_kwargs.feature_resolution = c.G_kwargs.feature_resolution
        if c.G_kwargs.semantic_floor_type == 'top_down':
            if opts.semantic_top_down_direct:
                c.D_kwargs.in_channels_semantic = c.G_kwargs.decoder_output_dim
            else:
                c.D_kwargs.in_channels_semantic = 3
        else:
            if opts.cfg == '3dfront_2d_volume':
                c.D_kwargs.in_channels_semantic = c.G_kwargs.num_semantic_y * c.G_kwargs.decoder_head_dim
            elif opts.cfg == '3dfront_2d_floorplan':
                c.D_kwargs.in_channels_semantic = c.G_kwargs.num_semantic_y * c.G_kwargs.voxel_feat_dim
            elif opts.cfg == '3dfront_2d':
                c.D_kwargs.in_channels_semantic = 3
    else:
        semantic_resolution = c.G_kwargs.neural_rendering_resolution
    c.D_kwargs.semantic_resolution = semantic_resolution
    c.G_kwargs.semantic_resolution = semantic_resolution


    if opts.density_reg > 0:
        c.G_reg_interval = opts.density_reg_every
    c.G_kwargs.rendering_kwargs = rendering_options
    c.G_kwargs.num_fp16_res = 0
    c.loss_kwargs.blur_init_sigma = 10 # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32 # Fade out the blur during the first N kimg.

    c.loss_kwargs.gpc_reg_prob = opts.gpc_reg_prob if opts.gen_pose_cond else None
    c.loss_kwargs.gpc_reg_fade_kimg = opts.gpc_reg_fade_kimg
    c.loss_kwargs.dual_discrimination = opts.dual_discrimination
    c.G_kwargs.sr_num_fp16_res = opts.sr_num_fp16_res

    c.G_kwargs.sr_kwargs = dnnlib.EasyDict(channel_base=opts.cbase, channel_max=opts.cmax, fused_modconv_default='inference_only')

    c.loss_kwargs.style_mixing_prob = opts.style_mixing_prob

    # Augmentation.
    if opts.aug != 'noaug':
        if opts.aug == 'diff':
            c.loss_kwargs.diffaugment='color,translation'
        else:
            c.loss_kwargs.diffaugment=None
            if opts.aug == 'xflip':
                c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1)
                c.augment_p = 0.5
            else:
                c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
                if opts.aug == 'ada':
                    c.ada_target = opts.target
                if opts.aug == 'fixed':
                    c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100 # Make ADA react faster at the beginning.
        c.ema_rampup = None # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            c.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
