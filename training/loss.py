# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from einops import rearrange
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.diffaugment import DiffAugment
from training.utils import linear_schedule
import torch.nn.functional as F

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
                 pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 r1_gamma_init=0, r1_gamma_fade_kimg=0, gpc_reg_fade_kimg=1000,
                 gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased', diffaugment=None,
                 recon_gamma=None, use_semantic_loss=False, semantic_gamma=1.0, semantic_ignore_index=255,
                 img_pairs = False, double_discrimination=False, cond_vid_fake_prob=1.0,
                 cond_vid_const_prob_init=1.0, cond_vid_const_prob_end=0.5, cond_vid_const_anneal_kimg=1000,
                 cond_vid_const_anneal_kimg_start=0, cond_vid_const_anneal=False, cond_vid_const_type = 'zeros',
                 neural_rendering_resolution = 64, cond_vid_const_aug=True):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.diffaugment = diffaugment
        self.recon_gamma = recon_gamma
        self.use_semantic_loss = use_semantic_loss
        self.semantic_gamma = semantic_gamma
        self.semantic_ignore_index = semantic_ignore_index
        self.semantic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=semantic_ignore_index)
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.img_pairs = img_pairs
        self.double_discrimination = double_discrimination
        self.cond_vid_fake_prob = cond_vid_fake_prob
        self.cond_vid_const_anneal = cond_vid_const_anneal
        self.cond_vid_const_prob_init = cond_vid_const_prob_init
        self.cond_vid_const_prob_end = cond_vid_const_prob_end
        self.cond_vid_const_anneal_kimg = cond_vid_const_anneal_kimg
        self.cond_vid_const_anneal_kimg_start = cond_vid_const_anneal_kimg_start
        self.cond_vid_const_prob = self.cond_vid_const_prob_init
        self.cond_vid_const_type = cond_vid_const_type
        self.cond_vid_const_aug = cond_vid_const_aug
        self.neural_rendering_resolution = neural_rendering_resolution
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.num_disc_frames = 1
        self.num_dual_frames = 2
        if self.img_pairs:
            self.num_frames = 2
        else:
            self.num_frames = 1
        self.num_disc_frames *= self.num_frames * self.num_dual_frames
        self.progressive_update(0)

    def progressive_update(self, cur_kimg):
        cur_kimg_offset = max(0, cur_kimg - self.cond_vid_const_anneal_kimg_start)
        if self.img_pairs and self.cond_vid_const_anneal:
            self.cond_vid_const_prob = linear_schedule(cur_kimg_offset, self.cond_vid_const_prob_init, self.cond_vid_const_prob_end, self.cond_vid_const_anneal_kimg)

    def run_G(self, z, c, neural_rendering_resolution, update_emas=False, cond_real_img=None, const_cond=False):
        # Take only first of image pair (first 3 channels)
        if self.img_pairs and cond_real_img is not None:
            cond_real_img_red = {}
            for k in ['image', 'image_raw']:
                cond_real_img_red[k] = cond_real_img[k][:, :3]
            cond_real_img_red_super = cond_real_img_red['image']
        else:
            cond_real_img_red = None
            cond_real_img_red_super = None
        gen_output = self.G(z, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, cond_img=cond_real_img_red_super, const_cond=const_cond)
        N = z.shape[0]
        if self.img_pairs:
            # Add real image raw to fake image raw output
            if cond_real_img_red_super is not None:
                gen_output['image_raw'] = torch.cat((cond_real_img_red['image_raw'], gen_output['image_raw']), 0)
            elif cond_real_img_red_super is None and const_cond:
                gen_output['image_raw'] = torch.cat((gen_output['image_raw'], gen_output['image_raw']), 0)
            for k in ['image', 'image_raw']:
                gen_output[k] = torch.cat((gen_output[k][:N], gen_output[k][N:]), 1)
        return gen_output

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, decode=False, compute_semantic=False, cond_vid_const_aug=True):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())
        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)
        if self.diffaugment is not None:
            if self.img_pairs:
                for k in ['image', 'image_raw']:
                    # Stack frame 0 and frame 1 in batch dimension
                    img[k] = rearrange([img[k][:, :3], img[k][:, 3:]], 't b c h w -> (b t) c h w')
            augmented_pair = DiffAugment(
                    rearrange(
                        [img['image'],
                        torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                        't b c h w -> (b t) c h w'),
                    policy=self.diffaugment,
                    num_disc_frames=self.num_disc_frames,
                    num_dual_frames=self.num_dual_frames,
                    )
            img_aug = augmented_pair[0]
            img_raw_aug = torch.nn.functional.interpolate(augmented_pair[1], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)
            img_aug_dict = {'image': img_aug, 'image_raw': img_raw_aug}
            if self.img_pairs:
                for k in ['image', 'image_raw']:
                    # Revert back frame and frame 1 to channel dimension
                    img_aug_dict[k] = torch.cat([img_aug_dict[k][i::self.num_frames] for i in range(self.num_frames)], 1)
                    if not cond_vid_const_aug:
                        img_aug_dict[k][:, :3] = 0
                    img[k] = img_aug_dict[k]
            else:
                for k in ['image', 'image_raw']:
                    img[k] = img_aug_dict[k]
        out = self.D(img, c, update_emas=update_emas, decode=decode, compute_semantic=compute_semantic)
        if isinstance(out, tuple):
            logits, recon = out
        else:
            logits = out['x']
            recon = None
        if self.use_semantic_loss:
            logits_semantic = out['x_semantic']
        else:
            logits_semantic = None
        if self.double_discrimination:
            logits_video = out['x_video']
        else:
            logits_video = None
        return logits, recon, logits_semantic, logits_video
    
    def calc_semantic_loss(self, gen_img_semantics, gen_semantics):
        ignore_mask = gen_img_semantics.sum(1) == 0
        label = gen_img_semantics.argmax(1)
        label[ignore_mask] = self.semantic_ignore_index 
        loss_semantic = self.semantic_loss_fn(gen_semantics, label) * self.semantic_gamma
        return loss_semantic

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        gen_c_cond = torch.zeros((gen_z.shape[0], 0), device=gen_z.device, dtype=gen_z.dtype) # TODO: Actually use gen_c cleanly
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        # phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None
        neural_rendering_resolution = self.neural_rendering_resolution
        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}
        # Augment constant conditional input
        cond_vid_const_aug = True
        if self.img_pairs:
            # Condition on constant (zeros) image with given probability
            if self.cond_vid_const_prob > np.random.rand():
                # Also change conditional frame of real images to constant (zeros)
                for k in ['image', 'image_raw']:
                    if self.cond_vid_const_type == 'zeros':
                        real_img[k][:, :3] = torch.zeros_like(real_img[k][:, :3])
                        cond_real_img = real_img
                        # Don't augment constant input if specified
                        cond_vid_const_aug = self.cond_vid_const_aug
                    # Condition on frame 1 and set both frames to frame 1
                    elif self.cond_vid_const_type == 'same':
                        real_img[k][:, :3] = real_img[k][:, 3:]
                        cond_real_img = None
                    const_cond = True
            # Condition image pair training on fake images with given probability
            elif self.cond_vid_fake_prob > np.random.rand():
                cond_real_img = None
                const_cond = False
            # Else condition on given real images
            else:
                cond_real_img = real_img
                const_cond = False
        else:
            cond_real_img = None
            const_cond = False


        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c, neural_rendering_resolution=neural_rendering_resolution, cond_real_img=cond_real_img, const_cond=const_cond)
                gen_img_semantics = gen_img['image_semantic']
                gen_img_alphas = gen_img['alphas']
                gen_logits, _ , gen_semantics, gen_logits_video = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, compute_semantic=True, cond_vid_const_aug=cond_vid_const_aug)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                if self.double_discrimination:
                    loss_Gmain_video = torch.nn.functional.softplus(-gen_logits_video)
                    training_stats.report('Loss/G_video/loss', loss_Gmain_video)
                    loss_Gmain = loss_Gmain + loss_Gmain_video.repeat(2, 1)
                if gen_semantics is not None:
                    loss_Gmain_semantic = self.calc_semantic_loss(gen_img_semantics, gen_semantics)
                    training_stats.report('Loss/G_semantic/loss', loss_Gmain_semantic)
                    loss_Gmain = loss_Gmain + loss_Gmain_semantic
                if gen_img_alphas is not None:
                    training_stats.report('Loss/G_alphas/loss', gen_img_alphas)
                    loss_Gmain = loss_Gmain + gen_img_alphas
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c_cond.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_cond.device) < swapping_prob, c_swapped, gen_c_cond)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c_cond)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(gen_c, all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c_cond.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_cond.device) < swapping_prob, c_swapped, gen_c_cond)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c_cond)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(gen_c, all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c_cond.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_cond.device) < swapping_prob, c_swapped, gen_c_cond)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c_cond)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(gen_c, all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c_cond.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_cond.device) < swapping_prob, c_swapped, gen_c_cond)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c_cond)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(gen_c, all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c_cond.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c_cond.device) < swapping_prob, c_swapped, gen_c_cond)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c_cond)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            # if self.style_mixing_prob > 0:
            #     with torch.autograd.profiler.record_function('style_mixing'):
            #         cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            #         cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            #         ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(gen_c, all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_c, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, cond_real_img=cond_real_img, const_cond=const_cond)
                gen_img_semantics = gen_img['image_semantic']
                gen_img_alphas = gen_img['alphas']
                gen_logits, _ , gen_semantics, gen_logits_video = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, compute_semantic=True, cond_vid_const_aug=cond_vid_const_aug)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                if self.double_discrimination:
                    loss_Dgen_video = torch.nn.functional.softplus(gen_logits_video)
                    training_stats.report('Loss/G_video/loss', loss_Dgen_video)
                    loss_Dgen = loss_Dgen + loss_Dgen_video.repeat(2, 1)
                if gen_semantics is not None:
                    loss_Dgen_semantic = self.calc_semantic_loss(gen_img_semantics, gen_semantics)
                    loss_Dgen = loss_Dgen + loss_Dgen_semantic
                    training_stats.report('Loss/G_semantic/loss', loss_Dgen_semantic)
                if gen_img_alphas is not None:
                    training_stats.report('Loss/G_alphas/loss', gen_img_alphas)
                    loss_Dgen = loss_Dgen + gen_img_alphas
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}
                real_logits, real_recon, _, real_logits_video = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, decode=self.recon_gamma is not None, cond_vid_const_aug=cond_vid_const_aug)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    if self.double_discrimination:
                        loss_Dreal_video = torch.nn.functional.softplus(-real_logits_video)
                        loss_Dreal = loss_Dreal + loss_Dreal_video.repeat(2, 1)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                        # TODO: Check this
                        if self.double_discrimination:
                            with torch.autograd.profiler.record_function('r1_grads_video'), conv2d_gradfix.no_weight_gradients():
                                r1_grads_video = torch.autograd.grad(outputs=[real_logits_video.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                                r1_grads_image_video = r1_grads_video[0]
                                r1_grads_image_raw_video = r1_grads_video[1]
                            r1_penalty_video = r1_grads_image_video.square().sum([1,2,3]) + r1_grads_image_raw_video.square().sum([1,2,3])
                            # Divide by two as video + image r1
                            r1_penalty = (r1_penalty + r1_penalty_video) / 2 
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                # Reconstruction loss, TODO: Check if every iteration needed
                if self.recon_gamma is not None:
                    real_image_raw_recon = filtered_resizing(real_img['image_raw'], size=real_img['image'].shape[-1], f=self.resample_filter)
                    real_img_recon = torch.cat([real_img['image'], real_image_raw_recon], 1)
                    loss_recon = F.mse_loss(real_img_recon, real_recon) * self.recon_gamma
                else:
                    loss_recon = 0
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1 + loss_recon).mean().mul(gain).backward()

#----------------------------------------------------------------------------
