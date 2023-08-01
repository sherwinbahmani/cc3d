# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section
from training.utils import create_voxel_grid
import torchvision
from copy import deepcopy

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = training_set.get_details(idx).raw_label
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
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

def save_image_grid(img, fname, drange):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    C, H, W = img.shape
    img = img.swapaxes(0, 1).swapaxes(1,2)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def norm_data(x, drange, drange_out=[-1, 1]):
    lo, hi = drange
    lo_out, hi_out = drange_out
    diff_out = hi_out - lo_out
    x = (x - lo) / (hi - lo) * diff_out + lo_out
    return x

def save_video_grid(img, fname, drange, fps):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    N, C, H, W = img.shape
    img = torch.from_numpy(img).permute(0, 2, 3, 1)
    if C == 1:
        img = img.repeat(1, 1, 1, 3)
    torchvision.io.write_video(fname, img, fps=fps)

def stack_video(video, key):
    return np.concatenate([np.stack([o_i[key].cpu().numpy().squeeze(0) for o_i in o]) for o in video], axis=3)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
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
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
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
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    # common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    common_kwargs = dict(c_dim=0, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G_kwargs.camera_coords_params = training_set.camera_coords_params
    G_kwargs.num_classes = training_set.num_classes
    G_kwargs.intrinsics = training_set.intrinsics
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    # G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs, num_classes=training_set.num_classes).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    # if rank == 0:
    #     z = torch.empty([batch_gpu, G.z_dim], device=device)
    #     c = torch.empty([batch_gpu, G.c_dim], device=device)
    #     img = misc.print_module_summary(G, [z, c])
    #     misc.print_module_summary(D, [img, c])

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
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
    # Export sample images.
    num_eval_seeds = 10
    num_coords_seed = min(training_set.num_samples_per_scene + 1, 20)
    fps = 10
    grid_size = None
    eval_z = None
    eval_c = None
    eval_3d_outputs = not (G_kwargs.model_type == '2D' and G_kwargs.feature_type == None)
    room_layouts = []
    if rank == 0:
        print('Exporting sample images...')
        img_dir = os.path.join(run_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        eval_z = torch.randn([num_eval_seeds, G.z_dim], device=device)
        num_scenes = training_set.num_labels
        if training_set.img_per_scene_ratio < num_coords_seed:
            num_scenes -= num_coords_seed
        if len(training_set) > num_coords_seed:
            eval_coords_indices = [np.arange(training_set.num_samples_per_scene)[:num_coords_seed] for _ in range(num_eval_seeds)]
        else:
            eval_coords_indices = [np.zeros(1).repeat(num_coords_seed).astype(int) for _ in range(num_eval_seeds)]
        if training_set.num_samples_per_scene == 1:
            eval_coords_indices = [np.repeat(i, 2) for i in eval_coords_indices]
        eval_scene_indices = [np.floor(np.random.randint(num_scenes)*training_set.img_per_scene_ratio).astype(int) for _ in range(num_eval_seeds)]
        eval_indices = [[(eval_scene_indices[ind] + i, i) for i in eval_coords_indices[ind]] for ind in range(num_eval_seeds)]
        eval_real_imgs = [[training_set.get_image(i, only_first=True) for (i, _) in coords] for coords in eval_indices]
        eval_c = [[training_set.get_label(np.floor(coords[0][0]/training_set.img_per_scene_ratio).astype(int), j) for (i, j) in coords] for coords in eval_indices]
        eval_c = [[misc.dict_to_device(c, device) for c in eval_c_i] for eval_c_i in eval_c]
        eval_c = [[misc.add_batch_dim_dict(c) for c in eval_c_i] for eval_c_i in eval_c]
        if eval_3d_outputs:
            # eval_c_videos = deepcopy(eval_c)
            eval_c_videos = [[training_set.get_label(np.floor(coords[0][0]/training_set.img_per_scene_ratio).astype(int), j, traj=True) for (i, j) in coords] for coords in eval_indices]
            eval_c_videos = [[misc.dict_to_device(c, device) for c in eval_c_i] for eval_c_i in eval_c_videos]
            eval_c_videos = [[misc.add_batch_dim_dict(c) for c in eval_c_i] for eval_c_i in eval_c_videos]
            eval_real_videos = np.concatenate([np.stack(img_seed) for img_seed in eval_real_imgs], axis=3)
            eval_real_videos = torch.from_numpy(eval_real_videos).permute(0, 2, 3, 1)
            torchvision.io.write_video(os.path.join(img_dir, "reals.mp4"), eval_real_videos, fps=fps)
        if training_set.dataset_name == "kitti":
            s_cols = {}
            for s_i in range(training_set.num_classes):
                s_cols[s_i] = torch.rand((3, 1, 1), device=device)
        for eval_c_i, eval_real_imgs_i in zip(eval_c, eval_real_imgs):
            # Set first eval index to top down view
            eval_c_i[0]["camera_coords"][:,[0,2]]= 0
            eval_c_i[0]["target_coords"][:,[0,2]]= 0
            eval_c_i[0]["target_coords"][:,1] = -1
            eval_c_i[0]["camera_coords"][:,1] = 1
            # Set first real image to scene layout
            if training_set.dataset_name == "3dfront":
                eval_labels_voxel_masks = {k:v.squeeze(0) for k, v in eval_c_i[0].items() if k not in ['camera_coords', 'target_coords', 'label_idx', 'coords_idx']}
                eval_voxel_masks = create_voxel_grid(**eval_labels_voxel_masks, colored=True, dims=(training_set.resolution, training_set.resolution, training_set.resolution))
                eval_real_imgs_i[0] = (eval_voxel_masks[:, :, 0, :]*255).cpu().numpy().astype(eval_real_imgs_i[0].dtype)
                if "room_layout" in eval_labels_voxel_masks:
                    room_layouts.append(eval_labels_voxel_masks["room_layout"].repeat(3, 1, 1).cpu().numpy().astype(eval_real_imgs_i[0].dtype)*255)
            elif training_set.dataset_name == "kitti":
                semantic_layout = eval_c_i[0]["semantic_layout"].squeeze(0)
                semantic_layout_col = torch.zeros((3, *semantic_layout.shape[1:]), device=device)
                for s_ind in torch.unique(semantic_layout):
                    s_mask = (semantic_layout == s_ind).repeat(3,1,1)
                    s_col = s_cols[s_ind.item()].repeat(1, *semantic_layout.shape[1:])
                    semantic_layout_col[s_mask] = s_col[s_mask]
                eval_real_imgs_i[0] = (semantic_layout_col*255).cpu().numpy().astype(eval_real_imgs_i[0].dtype)
        eval_real_imgs = np.concatenate([np.concatenate(img_seed, axis=2) for img_seed in eval_real_imgs], axis=1)
        if training_set.dataset_name == "3dfront":
            room_layouts = np.concatenate(room_layouts, axis=1)
            PIL.Image.fromarray(room_layouts.swapaxes(0, 1).swapaxes(1,2), 'RGB').save(os.path.join(img_dir, "layouts.png"))
        PIL.Image.fromarray(eval_real_imgs.swapaxes(0, 1).swapaxes(1,2), 'RGB').save(os.path.join(img_dir, "reals.png"))
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

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            # TODO: Use same all_gen_c labels as used for phase_real_c
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = misc.split_dict_gpus(phase_real_c, device, batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(training_set.num_labels), np.random.randint(training_set.num_samples_per_scene)) for _ in range(len(phases) * batch_size)]
            all_gen_c = [misc.dict_to_device(all_gen_c_i, device) for all_gen_c_i in all_gen_c]
            all_gen_c = misc.create_phase_dicts(all_gen_c, len(phases) * batch_size, batch_size, batch_gpu)

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
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
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        loss.progressive_update(cur_nimg / 1000)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
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

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                # Manually set first index to fixed camera view (top-down)
                out = [[G_ema(z=z.unsqueeze(0), c=c, use_camera=i==0, noise_mode='const', rand_render=False) for i, c in enumerate(eval_c_i)] for z, eval_c_i in zip(eval_z, eval_c)]
                fake_imgs = np.concatenate([np.concatenate([o_i['image'].cpu().numpy().squeeze(0) for o_i in o], axis=2) for o in out], axis=1)
                save_image_grid(fake_imgs, os.path.join(img_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1])
                if eval_3d_outputs:
                    fake_depths = []
                    for o in out:
                        min_depths = []
                        max_depths = []
                        fake_depths_o = []
                        for o_i in o:
                            fake_depths_o_i = o_i['image_depth'].cpu().numpy().squeeze(0)
                            fake_depths_o.append(fake_depths_o_i)
                            min_depths.append(fake_depths_o_i.min().item())
                            max_depths.append(fake_depths_o_i.max().item())
                        # For top down rendering normalize not based on min/max, separate first and other coordinates
                        min_depth = [min_depths[0]] + [min(min_depths[1:])]*(len(min_depths) - 1)
                        max_depth = [max_depths[0]] + [max(max_depths[1:])]*(len(max_depths) - 1)
                        fake_depths_o = [norm_data(fake_depths_o_i, (min_depth[i], max_depth[i])) for i, fake_depths_o_i in enumerate(fake_depths_o)]
                        fake_depths_o = np.concatenate(fake_depths_o, axis=2)
                        fake_depths.append(fake_depths_o)
                    fake_imgs_raw = np.concatenate([np.concatenate([o_i['image_raw'].cpu().numpy().squeeze(0) for o_i in o], axis=2) for o in out], axis=1)
                    fake_depths = np.concatenate(fake_depths, axis=1)
                    save_image_grid(fake_imgs_raw, os.path.join(img_dir, f'fakes_raw{cur_nimg//1000:06d}.png'), drange=[-1,1])
                    save_image_grid(fake_depths, os.path.join(img_dir, f'depths{cur_nimg//1000:06d}.png'), drange=[-1,1])
                    for rand_render_k, rand_render in [('true', True), ('false', False)]:
                        for noise_type in ['const', 'none']:
                            out = []
                            for z, eval_c_i in zip(eval_z, eval_c_videos):
                                out_z = []
                                min_depths = []
                                max_depths = []
                                for i, c in enumerate(eval_c_i):
                                    # Condition either with constant image (zeros) or with previous frame
                                    if i == 0:
                                        cond_img_i = None
                                        const_cond = True
                                    else:
                                        cond_img_i = out_z_i['image']
                                        const_cond = False
                                    out_z_i = G_ema(z=z.unsqueeze(0), c=c, noise_mode=noise_type, rand_render=rand_render, cond_img=cond_img_i, const_cond=const_cond)
                                    out_z.append(out_z_i)
                                    fake_depths_o_i = out_z_i['image_depth']
                                    min_depths.append(fake_depths_o_i.min().item())
                                    max_depths.append(fake_depths_o_i.max().item())
                                min_depth = min(min_depths)
                                max_depth = max(max_depths)
                                for out_z_i in out_z:
                                    out_z_i['image_depth'] = norm_data(out_z_i['image_depth'], (min_depth, max_depth))
                                out.append(out_z)
                            for out_type, drange in [('image', [-1,1]), ('image_raw', [-1,1]), ('image_depth', [-1,1])]:
                                out_videos_type = stack_video(out, out_type)        
                                save_video_grid(out_videos_type, os.path.join(img_dir, f'fakes_{out_type}_{noise_type}_rand_render_{rand_render_k}{cur_nimg//1000:06d}.mp4'), drange=drange, fps=fps)
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print('Evaluating metrics...')
        #         for metric in metrics:
        #             result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #                 dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #             if rank == 0:
        #                 metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #             stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
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
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

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
