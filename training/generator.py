# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
import math
import numpy as np
from torch_utils import persistence

from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_3dgrid
from training.volumetric_rendering.ray_sampler import RaySampler
from training.unet3d import UNet3DEncoder
from training.unet2d import UNet2DEncoder, ConvStage, ConvBlock, ConvBlockNorm, UNet2DEncoderMod, ConvRelu
from training.networks import FullyConnectedLayer, MappingNetwork, SynthesisNetwork, normalize_2nd_moment, ConvFuseBlock, Conv2DNormMod
from training.networks_3d import SynthesisNetwork3D
from training.superresolution import SuperResolution, SuperResolutionShared
from training.utils import get_rotation_matrix, voxelize, voxelize2d

from camera_utils import FOV_to_intrinsics, CameraTargetSampler

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim = 512,
        c_dim = 0,
        w_dim = 512,
        img_resolution = 256,
        grid_size = 256,
        feature_type = None, # [None, 'volume', 'planes', 'floor_plan']
        cond_content=['class_one_hot_embed', 'local_coords_canonical', 'room_layout'], # 'binary', 'class', 'class_one_hot', 'class_one_hot_embed', 'latent', 'local_coords', 'local_coords_canonical', 'global_coords', 'room_layout'  
        encoder_out_resolution = 4,
        neural_rendering_resolution = 64,
        feature_resolution = 256,
        decoder_output_dim = 32,
        voxel_feat_dim = 3, # Number of features for neural rendering
        decoder_head_dim = None, # Number of channels used only for model_type = '2D_volume'
        num_classes = 21, # TODO: Depending on dataset, 23 with lamps
        class_embed_dim = 16,
        z_dim_obj = 512,
        z_obj_embed_dim = 16,
        sr_num_fp16_res = 0,
        rendering_kwargs = {},
        mapping_kwargs = {},
        model_type = '2D', # '2D', '3D', '2D_volume'
        sr_kwargs = {},
        encoder_norm_type = None, # None, 'instance', '2nd_moment'
        fov = 70.,
        unconditional = False,
        use_semantic_loss = False,
        use_semantic_floor = False,
        semantic_floor_type = 'floor', # 'floor', 'boxes', 'top_down'
        num_semantic_y = 8,
        semantic_resolution = None,
        concat_occupancy = False,
        camera_y_noise = None,
        fov_noise = None,
        camera_coords_params = None,
        compute_alphas = False,
        alpha_gamma = 1.0,
        alpha_noise_factor = 0.11,
        hidden_decoder_mlp_dim = 64,
        n_hidden_layers_mlp = 1,
        img_pairs = False,
        img_out_channels = 3,
        img_out_channels_pair_1 = 3,
        img_out_channels_pair_0 = 3,
        pairs_conv_fuse_inter_channels = 64,
        pairs_conv_fuse_num_layers = 2,
        pairs_conv_fuse_type = 'concat_norm',
        out_sigmoid = False,
        cond_vid_const_type = 'zeros',
        use_out_extra_conv = False,
        super_res_shared=False,
        concat_floor_height=False,
        conv_head_mod=False,
        use_obj_latent=False,
        encoder_mod=False,
        use_layout_latent=False,
        intrinsics=None,
        add_floor_class=False,
        add_none_class=False,
        semantic_top_down_direct=False,
        **synthesis_kwargs, # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        if use_obj_latent:
            cond_content += ['latent']
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.z_dim_obj = z_dim_obj
        self.img_resolution = img_resolution
        self.feature_type = feature_type
        self.neural_rendering_resolution = neural_rendering_resolution
        self.feature_resolution = feature_resolution
        self.rendering_kwargs = rendering_kwargs
        self.cond_content = cond_content
        self.num_classes = num_classes
        self.class_embed_dim = class_embed_dim
        self.z_obj_embed_dim = z_obj_embed_dim
        self.model_type = model_type
        self.grid_size = grid_size
        self.fov = fov
        self.decoder_head_dim = decoder_head_dim
        self.unconditional = unconditional
        self.use_semantic_loss = use_semantic_loss
        self.use_semantic_floor = use_semantic_floor
        self.semantic_floor_type = semantic_floor_type
        self.semantic_resolution = semantic_resolution
        self.num_semantic_y = num_semantic_y
        self.concat_occupancy = concat_occupancy
        self.camera_y_noise = camera_y_noise
        self.fov_noise = fov_noise
        self.compute_alphas = compute_alphas
        self.alpha_gamma = alpha_gamma
        self.alpha_noise_factor = alpha_noise_factor if compute_alphas else None
        self.upsample = self.img_resolution != self.neural_rendering_resolution
        self.img_pairs = img_pairs
        self.pairs_conv_fuse_type = pairs_conv_fuse_type
        self.out_sigmoid = out_sigmoid
        self.cond_vid_const_type = cond_vid_const_type
        self.use_out_extra_conv = use_out_extra_conv
        self.super_res_shared = super_res_shared
        self.concat_floor_height = concat_floor_height
        self.conv_head_mod = conv_head_mod
        self.encoder_mod = encoder_mod
        self.use_layout_latent = use_layout_latent
        self.add_floor_class = add_floor_class
        self.add_none_class = add_none_class
        self.semantic_top_down_direct = semantic_top_down_direct
        self.layout_conv = None
        self.num_ws = 0
        if self.concat_floor_height:
            floor_coords_dim = 1
        else:
            floor_coords_dim = 0
        if self.img_pairs:
            img_out_channels = img_out_channels_pair_1
        if not self.upsample:
            decoder_output_dim = 3
        if self.concat_occupancy:
            occ_channels = 1
        else:
            occ_channels = 0
        if self.model_type in ['2D', '2D_volume']:
            self.dims = (self.grid_size, 1, self.grid_size)
        elif self.model_type == '3D':
            self.dims = (self.grid_size, self.grid_size, self.grid_size)
        if not self.unconditional:
            self.num_feats_voxel_in = 0
            if 'semantic_layout' in cond_content:
                room_layout_dim = 0
                self.num_feats_voxel_in += self.num_classes
                self.obj_embedding = None
                self.class_embedding = None
                if 'latent' in cond_content:
                    self.num_feats_voxel_in += self.z_dim_obj
                # if 'latent_embed' in cond_content:
                #     self.num_feats_voxel_in += self.z_obj_embed_dim
                #     self.obj_embedding = EmbeddingNetwork(in_dim=self.z_dim_obj, out_dim=self.z_obj_embed_dim)
                self.layout_conv = ConvRelu(self.num_feats_voxel_in, self.num_feats_voxel_in)
            else:
                if 'binary' in cond_content:
                    self.num_feats_voxel_in += 1
                if 'class' in cond_content:
                    self.num_feats_voxel_in += 1
                if 'class_one_hot' in cond_content:
                    extra_class = -1 if add_floor_class else 0
                    if self.add_none_class:
                        extra_class -= 1
                    self.num_feats_voxel_in += self.num_classes + extra_class
                if 'class_one_hot_embed' in cond_content:
                    self.num_feats_voxel_in += self.class_embed_dim
                    extra_class = -1 if add_floor_class else 0
                    if self.add_none_class:
                        extra_class -= 1
                    self.class_embedding = EmbeddingNetwork(in_dim=self.num_classes + extra_class, out_dim=self.class_embed_dim)
                else:
                    self.class_embedding = None
                if 'latent' in cond_content:
                    self.num_feats_voxel_in += self.z_dim_obj
                if 'latent_embed' in cond_content:
                    self.num_feats_voxel_in += self.z_obj_embed_dim
                    self.obj_embedding = EmbeddingNetwork(in_dim=self.z_dim_obj, out_dim=self.z_obj_embed_dim)
                else:
                    self.obj_embedding = None
                if 'local_coords' in cond_content:
                    self.num_feats_voxel_in += 3
                if 'local_coords_canonical' in cond_content:
                    self.num_feats_voxel_in += 3
                if 'global_coords' in cond_content:
                    self.num_feats_voxel_in += 3
                if 'room_layout' in cond_content:
                    room_layout_dim = 1
                else:
                    room_layout_dim = 0
                if self.use_layout_latent:
                    room_layout_dim += self.z_dim_obj
            # Encoding Network for voxelized bounding boxes
            num_levels = int(math.log2(self.feature_resolution) - math.log2(encoder_out_resolution) + 1)
            if self.model_type == '3D':
                self.box_encoder = BoxEncoder(
                    feature_type=feature_type,
                    in_channels=self.num_feats_voxel_in + room_layout_dim,
                    num_levels=num_levels,
                    encoder_mod=encoder_mod,
                    init_resolution=self.feature_resolution,
                    w_dim=w_dim,
                    **synthesis_kwargs
                    )
            elif self.model_type in ['2D', '2D_volume']:
                self.box_encoder = BoxEncoder(
                    feature_type='planes',
                    in_channels=self.num_feats_voxel_in + room_layout_dim,
                    num_levels=num_levels,
                    encoder_mod=encoder_mod,
                    init_resolution=self.feature_resolution,
                    w_dim=w_dim,
                    **synthesis_kwargs
                    )
            self.num_ws += self.box_encoder.num_ws
            if encoder_norm_type == '2nd_moment':
                self.encoder_norm = None
            elif encoder_norm_type == 'instance':
                res_ch_dict = {self.feature_resolution // 2**i: ch for i, ch in enumerate(self.box_encoder.num_out_channels)}
                if self.model_type == '3D':
                    norm_class = torch.nn.InstanceNorm3d
                elif self.model_type in ['2D', '2D_volume']:
                    norm_class = torch.nn.InstanceNorm2d
                self.encoder_norm = {k: norm_class(v) for k, v in res_ch_dict.items()}
            else:
                self.encoder_norm = None
            self.encoder_norm_type = encoder_norm_type
            enc_feats = {2**i*encoder_out_resolution: dim for i, dim in enumerate(self.box_encoder.num_out_channels[::-1])}
        else:
            enc_feats = None
        self.renderer = ImportanceRenderer(feature_type=feature_type, concat_floor_height=self.concat_floor_height)
        self.ray_sampler = RaySampler()
        if self.model_type == '3D':
            synthesis_kwargs['img_channels'] = voxel_feat_dim
            self.synthesis = SynthesisNetwork3D(w_dim=w_dim, img_resolution=self.feature_resolution,
                                            enc_feats = enc_feats, **synthesis_kwargs)
            self.decoder_head = None
            self.decoder = OSGDecoder(
                decoder_output_dim=decoder_output_dim,
                n_features=self.synthesis.img_channels + occ_channels + floor_coords_dim,
                hidden_dim=hidden_decoder_mlp_dim,
                n_hidden_layers=n_hidden_layers_mlp
                )
        elif self.model_type == '2D' and self.feature_type == None:
            synthesis_kwargs['img_channels'] = voxel_feat_dim
            self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=self.feature_resolution,
                                            enc_feats = enc_feats, **synthesis_kwargs)
            self.decoder_head = None
        elif self.model_type == '2D_volume':
            synthesis_kwargs['img_channels'] = voxel_feat_dim
            self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=self.feature_resolution,
                                            enc_feats = enc_feats, **synthesis_kwargs)
            if self.conv_head_mod:
                self.decoder_head = Conv2DNormMod(
                    synthesis_kwargs['img_channels'], self.feature_resolution * decoder_head_dim,
                    w_dim=w_dim, resolution=self.feature_resolution, fused_modconv_default=synthesis_kwargs['fused_modconv_default'])
                self.num_ws += self.decoder_head.num_ws
            else:
                self.decoder_head = ConvBlockNorm(synthesis_kwargs['img_channels'], self.feature_resolution * decoder_head_dim)
            self.decoder = OSGDecoder(
                decoder_output_dim=decoder_output_dim,
                n_features=decoder_head_dim + occ_channels + floor_coords_dim,
                hidden_dim=hidden_decoder_mlp_dim,
                n_hidden_layers=n_hidden_layers_mlp
                )
        elif self.model_type == '2D' and self.feature_type in ['floor_plan']:
            synthesis_kwargs['img_channels'] = voxel_feat_dim
            self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=self.feature_resolution,
                                            enc_feats = enc_feats, **synthesis_kwargs)
            self.decoder = OSGDecoder(
                decoder_output_dim=decoder_output_dim,
                n_features=self.synthesis.img_channels + occ_channels + floor_coords_dim,
                hidden_dim=hidden_decoder_mlp_dim,
                n_hidden_layers=n_hidden_layers_mlp
                )
        elif self.model_type == '2D' and self.feature_type in ['planes']:
            synthesis_kwargs['img_channels'] = voxel_feat_dim
            self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=self.feature_resolution,
                                            enc_feats = enc_feats, **synthesis_kwargs)
            self.decoder = OSGDecoder(
                decoder_output_dim=decoder_output_dim,
                n_features=self.synthesis.img_channels//3 + occ_channels + floor_coords_dim,
                hidden_dim=hidden_decoder_mlp_dim,
                n_hidden_layers=n_hidden_layers_mlp
                )
        self.num_ws += self.synthesis.num_ws
        if self.model_type in ['3D', '2D_volume'] or (self.model_type == '2D' and self.feature_type in ['planes', 'floor_plan']):
            if self.upsample:
                if self.super_res_shared:
                    self.superresolution = SuperResolutionShared(
                    in_channels=decoder_output_dim, img_resolution=img_resolution,
                    input_resolution=self.neural_rendering_resolution, w_dim=w_dim,
                    sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], 
                    img_channels=img_out_channels, **sr_kwargs
                    )
                else:
                    self.superresolution = SuperResolution(
                        in_channels=decoder_output_dim, img_resolution=img_resolution,
                        input_resolution=self.neural_rendering_resolution, w_dim=w_dim,
                        sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], 
                        img_channels=img_out_channels, **sr_kwargs
                        )
                self.num_ws += self.superresolution.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        if intrinsics is None:
            self.intrinsics = FOV_to_intrinsics(torch.tensor(fov).unsqueeze(0))
        else:
            self.intrinsics = torch.tensor(intrinsics)[:3, :3].unsqueeze(0).float()
        self.extrinsics_sampler = CameraTargetSampler()
        if camera_coords_params is not None:
            # Add noise to camera height
            if self.camera_y_noise is not None:
                camera_coords_params[:, :, 1] += torch.normal(0.0, self.camera_y_noise, camera_coords_params.shape[:2])
            self.camera_coords_params = torch.nn.Parameter(camera_coords_params)
        else:
            self.camera_coords_params = None
        if self.img_pairs:
            if self.pairs_conv_fuse_type == 'concat_norm':
                self.pairs_conv_fuse = ConvBlockNorm(
                    img_out_channels+img_out_channels_pair_0, img_out_channels,
                    num_layers=pairs_conv_fuse_num_layers, inter_channels=pairs_conv_fuse_inter_channels,
                    )
            elif self.pairs_conv_fuse_type in ['conv_fuse_1', 'conv_fuse_0']:
                self.pairs_conv_fuse = ConvFuseBlock(
                    img_out_channels+img_out_channels_pair_0, img_out_channels,
                    inter_channels=pairs_conv_fuse_inter_channels, resolution=img_resolution,
                    w_dim=w_dim, use_noise=False
                )
        else:
            self.pairs_conv_fuse = None
        
        if self.use_out_extra_conv:
            self.out_extra_conv = ConvBlockNorm(
                img_out_channels, img_out_channels,
                num_layers=pairs_conv_fuse_num_layers, inter_channels=pairs_conv_fuse_inter_channels,
                )
        else:
            self.out_extra_conv = None

    def forward(self, z, c, neural_rendering_resolution = None, use_camera=False, update_emas=False, norm_depth=False, rand_render=True, cond_img=None, const_cond=False, obj_latents=None, render_depth=False, random_latents=False, **synthesis_kwargs):
        N = z.shape[0]
        img_pairs_train = True if self.img_pairs and self.training else False
        img_pairs_eval = True if self.img_pairs and not self.training else False
        # Either take frozen camera coordinates or learnable
        if self.camera_coords_params is None or use_camera:
            camera_coords = c['camera_coords']
        else:
            camera_coords = self.camera_coords_params[c['label_idx'].squeeze(1), c['coords_idx'].squeeze(1)]
        target_coords = c['target_coords']
        if self.fov_noise is not None:
            fov_noise = torch.normal(0.0, self.fov_noise, (N,))
            intrinsics = FOV_to_intrinsics(self.fov + fov_noise, device=fov_noise.device)
        else:
            intrinsics = self.intrinsics.repeat(N, 1, 1)
        intrinsics = intrinsics.to(z.device)
        is_pairs_coords = img_pairs_train and not const_cond and cond_img is None
        is_single_coords = (img_pairs_train and const_cond) or (img_pairs_train and not const_cond and cond_img is not None)
        if is_pairs_coords:
            camera_coords = torch.cat((camera_coords[:, :3], camera_coords[:, 3:]), 0)
            target_coords = target_coords.repeat(2, 1)
            intrinsics = intrinsics.repeat(2, 1, 1)
        elif is_single_coords:
            # Generate frame 1 when conditioned on frame 0
            camera_coords = camera_coords[:, 3:]
        elif img_pairs_eval:
            camera_coords = camera_coords[:, :3]
        else:
            camera_coords = camera_coords[:, :3]
        if self.semantic_floor_type == 'top_down' or self.compute_alphas:
            # Add top down coordinates
            camera_coords, target_coords, intrinsics = self.add_top_down_coords(camera_coords, target_coords, intrinsics)
        cam2world_matrix = self.extrinsics_sampler.sample(camera_coords, target_coords)
        boxes = {k:v for k, v in c.items() if k not in ['camera_coords', 'target_coords', 'label_idx', 'coords_idx']}
        
        # # Map latent code to intermediate representation
        ws = self.mapping(z, torch.zeros((N, 0), device=z.device, dtype=z.dtype), update_emas=update_emas)
        if not self.unconditional:
            # Generate bounding boxes and convert to feature grid
            features, masks, semantics, occupancy_grids = self.create_feature_grid(random_latents=random_latents, **boxes)
            if not self.use_semantic_loss:
                semantics = None

            if self.model_type in ['2D', '2D_volume']:
                features = features[:, :, :, 0, :]
                masks = masks[:, :, :, 0, :]
            if self.encoder_mod:
                ws_encoder = ws[:, :self.box_encoder.num_ws]
                ws = ws[:, self.box_encoder.num_ws:]
            else:
                ws_encoder = None
            
            if self.layout_conv is not None:
                features = self.layout_conv(features)
            features = self.box_encoder(features, ws_encoder, **synthesis_kwargs)

            if self.encoder_norm_type is not None:
                for k, v in features.items():
                    if self.encoder_norm_type == 'instance':
                        features[k] = self.encoder_norm[k](v)
                    elif self.encoder_norm_type == '2nd_moment':
                        features[k] = normalize_2nd_moment(v)
        else:
            features = None
            masks = None
            semantics = None
            occupancy_grids = None
        ws_synthesis = ws[:, :self.synthesis.num_ws]
        if self.conv_head_mod:
            ws_decoder_head = ws[:, self.synthesis.num_ws:self.synthesis.num_ws+self.decoder_head.num_ws]
        else:
            ws_decoder_head = None
        if self.super_res_shared:
            ws_super = ws
        else:
            if self.conv_head_mod:
                ws_super = ws[:, self.synthesis.num_ws+self.decoder_head.num_ws:]
            else:
                ws_super = ws[:, self.synthesis.num_ws:]
        # ws_super = ws[:, -1:]
        features = self.synthesis(ws_synthesis, features, update_emas=update_emas, **synthesis_kwargs)
        if self.model_type == '2D' and self.feature_type == None:
            if semantics is not None:
                semantics_gt = semantics[:, :, :, 0, :]
            else:
                semantics_gt = None
            return {'image': features, 'image_raw': features, 'image_depth': None, 'image_semantic': semantics_gt, 'image_semantic_sampled': features, 'alphas': None}
        if self.model_type == '2D_volume':
            # Create y dimension by reshaping channels
            if self.conv_head_mod:
                features = self.decoder_head(features, ws_decoder_head, **synthesis_kwargs)
            else:
                features = self.decoder_head(features)
            features = features.reshape(features.shape[0], self.decoder_head_dim, -1, *features.shape[2:]).transpose(2, 3) 
        if self.concat_occupancy:
            features = torch.cat((features, occupancy_grids), dim=1)

        # Create a batch of rays for volume rendering
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, alpha_noise_factor=self.alpha_noise_factor)

        # Reshape output into three planes
        if self.feature_type == 'planes':
            features = features.view(len(features), 3, features.shape[1]//3, features.shape[-2], features.shape[-1])
        elif self.feature_type == 'floor_plan':
            features = features.unsqueeze(1)
        
        if self.use_semantic_floor and self.use_semantic_loss:
            semantic_sampled, semantic_image = self.sample_semantic_grid(features, semantics)
            semantics = None
        else:
            semantic_sampled = None
        # Reuse features for top down renderings
        if self.semantic_floor_type == 'top_down' or self.compute_alphas:
            features = torch.cat((features, features), dim=0)
            if not self.semantic_top_down_direct:
                ws_super = torch.cat((ws_super, ws_super), dim=0)
            N *= 2

        # Perform volume rendering
        if is_pairs_coords:
            features = features.repeat(2, 1, 1, 1, 1)
            ws_super = ws_super.repeat(2, 1, 1)
            N = 2 * N
        feature_samples, depth_samples, weights_samples, semantic_samples, alphas = self.renderer(features, self.decoder, ray_origins, ray_directions, self.rendering_kwargs, semantics, self.compute_alphas, rand_render=rand_render) # channels last

        # Compute norm of alpha values to minimize them as a loss
        if self.compute_alphas:
            alphas = alphas[N//2:]
            alphas = alphas.mean() * self.alpha_gamma

        # Reshape into 'raw' neural-rendered image
        H = W = neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        if render_depth:
            depth_image = self.out_depth(depth_image, norm_depth)
            return {'image_depth': depth_image}
        if not self.use_semantic_floor:
            if semantic_samples is not None:
                semantic_image = semantic_samples.permute(0, 2, 1).reshape(N, semantic_samples.shape[-1], H, W).contiguous()
            else:
                semantic_image = None

        # Run superresolution to get final image
        if self.semantic_floor_type == 'top_down':
            if self.semantic_top_down_direct:
                semantic_sampled = feature_image[N//2:]
                feature_image = feature_image[:N//2]
        rgb_image = feature_image[:, :3]
        if self.upsample:
            sr_image = self.superresolution(rgb_image, feature_image, ws_super, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})
        else:
            sr_image = rgb_image
        # Use top down rendering as semantic label
        if self.semantic_floor_type == 'top_down':
            if not self.semantic_top_down_direct:
                semantic_sampled = sr_image[N//2:]
            out_semantic_shape = (self.feature_resolution,)*2
            if semantic_sampled.shape[2:] != out_semantic_shape:
                semantic_sampled = torch.nn.functional.interpolate(semantic_sampled, size=out_semantic_shape, mode='bilinear', align_corners=False, antialias=True)
        if self.semantic_floor_type == 'top_down' or self.compute_alphas:
            sr_image = sr_image[:N//2]
            rgb_image = rgb_image[:N//2]
            depth_image = depth_image[:N//2]
        if self.img_pairs:
            # Split frames into pairs of two timesteps and process timestep 1
            if img_pairs_train:
                if cond_img is not None:
                    sr_image_0 = cond_img
                    sr_image_1 = sr_image
                elif cond_img is None and const_cond:
                    if self.cond_vid_const_type == 'zeros':
                        sr_image_0 = sr_image[:N//2]
                        sr_image_1 = sr_image[N//2:]
                    elif self.cond_vid_const_type == 'same':
                        sr_image_0 = sr_image
                        sr_image_1 = sr_image
                else:
                    sr_image_0 = sr_image[:N//2]
                    sr_image_1 = sr_image[N//2:]
            # Use same timestep for both frames
            elif img_pairs_eval:
                if cond_img is not None:
                    sr_image_0 = cond_img
                else:
                    if self.cond_vid_const_type == 'zeros':
                        sr_image_0 = torch.zeros_like(sr_image)
                    elif self.cond_vid_const_type == 'same':
                        sr_image_0 = sr_image
                sr_image_1 = sr_image
            if self.pairs_conv_fuse_type == 'concat_norm':
                sr_image_1 = torch.cat((sr_image_0, sr_image_1), 1)
                sr_image_1 = self.pairs_conv_fuse(sr_image_1)
            elif self.pairs_conv_fuse_type == 'conv_fuse_1':
                sr_image_both = torch.cat((sr_image_0, sr_image_1), 1)
                sr_image_1 = self.pairs_conv_fuse(sr_image_both, sr_image_1)
            elif self.pairs_conv_fuse_type == 'conv_fuse_0':
                sr_image_both = torch.cat((sr_image_0, sr_image_1), 1)
                sr_image_1 = self.pairs_conv_fuse(sr_image_both, sr_image_0)
            if img_pairs_train:
                sr_image = torch.cat((sr_image_0, sr_image_1), 0)
            # Use second timestep as prediction for eval, TODO: Check
            elif img_pairs_eval:
                sr_image = sr_image_1
        
        if self.use_out_extra_conv:
            sr_image = self.out_extra_conv(sr_image)
        
        depth_image = self.out_depth(depth_image, norm_depth)
        if self.out_sigmoid:
            sr_image = self.sigmoid_clamp(sr_image)
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_semantic': semantic_image, 'image_semantic_sampled': semantic_sampled, 'alphas': alphas}
    
    def out_depth(self, depth_image, norm_depth):
        depth_image = -depth_image
        if norm_depth:
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()) * 2 - 1
        return depth_image

    def sample_mixed(self, c, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        boxes = {k:v for k, v in c.items() if k not in ['camera_coords', 'target_coords', 'label_idx', 'coords_idx']}
        # # Map latent code to intermediate representation
        if not self.unconditional:
            # Generate bounding boxes and convert to feature grid
            features, masks, semantics, occupancy_grids = self.create_feature_grid(**boxes)

            if self.model_type in ['2D', '2D_volume']:
                features = features[:, :, :, 0, :]
                masks = masks[:, :, :, 0, :]
            if self.encoder_mod:
                ws_encoder = ws[:, :self.box_encoder.num_ws]
                ws = ws[:, self.box_encoder.num_ws:]
            else:
                ws_encoder = None
            if self.layout_conv is not None:
                features = self.layout_conv(features)
            features = self.box_encoder(features, ws_encoder)

            if self.encoder_norm_type is not None:
                for k, v in features.items():
                    if self.encoder_norm_type == 'instance':
                        features[k] = self.encoder_norm[k](v)
                    elif self.encoder_norm_type == '2nd_moment':
                        features[k] = normalize_2nd_moment(v)
        else:
            features = None
            masks = None
            occupancy_grids = None
        semantics = None
        ws_synthesis = ws[:, :self.synthesis.num_ws]
        if self.conv_head_mod:
            ws_decoder_head = ws[:, self.synthesis.num_ws:self.synthesis.num_ws+self.decoder_head.num_ws]
        else:
            ws_decoder_head = None
        # ws_super = ws[:, -1:]
        features = self.synthesis(ws_synthesis, features, update_emas=update_emas, **synthesis_kwargs)
        if self.model_type == '2D_volume':
            # Create y dimension by reshaping channels
            if self.conv_head_mod:
                features = self.decoder_head(features, ws_decoder_head, **synthesis_kwargs)
            else:
                features = self.decoder_head(features)
            features = features.reshape(features.shape[0], self.decoder_head_dim, -1, *features.shape[2:]).transpose(2, 3) 
        if self.concat_occupancy:
            features = torch.cat((features, occupancy_grids), dim=1)
        
        # Reshape output into three planes
        if self.feature_type == 'planes':
            features = features.view(len(features), 3, features.shape[1]//3, features.shape[-2], features.shape[-1])
        elif self.feature_type == 'floor_plan':
            features = features.unsqueeze(1)
        return self.renderer.run_model(features, self.decoder, coordinates, directions, self.rendering_kwargs, semantics) # channels last
    
    def add_top_down_coords(self, camera_coords, target_coords, intrinsics):
        device = camera_coords.device
        N = camera_coords.shape[0]
        camera_coords_top_down = torch.tensor([0., 1., 0.], device=device).unsqueeze(0).repeat(N, 1)
        target_coords_top_down = torch.tensor([0., -1., 0.], device=device).unsqueeze(0).repeat(N, 1)
        camera_coords = torch.cat((camera_coords, camera_coords_top_down), dim=0)
        target_coords = torch.cat((target_coords, target_coords_top_down), dim=0)
        intrinsics = torch.cat((intrinsics, intrinsics), dim=0)
        return camera_coords, target_coords, intrinsics
    
    def sigmoid_clamp(self, x):
        x = torch.sigmoid(x)*(1 + 2*0.001) - 0.001
        x = x * 2 - 1
        return x
    
    def sample_semantic_grid(self, features, semantics, noise_factor = 0.2):
        device = features.device
        if self.semantic_floor_type == 'top_down':
            semantic_image = None
            semantic_gt = semantics[:, :, :, 0, :]
        else:
            if self.feature_type == 'planes':
                raise NotImplementedError
            elif self.feature_type == 'floor_plan':
                semantic_gt = semantics[:, :, :, 0, :]
                semantic_image = features.squeeze(1)
            elif self.feature_type == 'volume':
                N, C, H, W, D = features.shape
                # Set up given number of equidistant height coordinates
                y_coords = torch.linspace(0, 1, self.num_semantic_y + 1, device=device)[:-1].unsqueeze(0).repeat(N, 1)
                y_coords_offset = torch.rand((N, 1), device=device)
                y_coords = y_coords + y_coords_offset
                y_coords = torch.fmod(y_coords, 1)
                y_coords = y_coords.sort()[0]
                y_coords = y_coords * 2 - 1
                # Sample center points in image plane with random noise
                x_noise = torch.normal(0.0, 1/H*noise_factor, (N, 1), device=device)
                z_noise = torch.normal(0.0, 1/D*noise_factor, (N, 1), device=device)
                x_coords = torch.linspace(-1+1/H, 1-1/H, H, device=device).unsqueeze(0).repeat(N, 1) + x_noise
                z_coords = torch.linspace(-1+1/D, 1-1/D, D, device=device).unsqueeze(0).repeat(N, 1) + z_noise
                coords = torch.stack([torch.stack(torch.meshgrid(x,y,z, indexing='ij')).view(3, -1)
                                    for x, y, z in zip(x_coords,y_coords,z_coords)])
                coords = coords.transpose(1, 2)
                # Sample from feature volume
                semantic_samples = sample_from_3dgrid(features, coords)
                semantic_image = semantic_samples.permute(0, 2, 1).reshape(N, C, H, self.num_semantic_y, D).contiguous()
                semantic_image = semantic_image.permute(0, 1, 3, 4, 2)
                semantic_image = semantic_image.reshape(N, -1, H, D)

                if self.semantic_floor_type == 'floor':
                    semantic_gt = semantics[:, :, :, 0, :]
                elif self.semantic_floor_type == 'boxes':
                    N_sem, C_sem, H_sem, W_sem, D_sem = semantics.shape
                    x_coords = torch.linspace(-1+1/H_sem, 1-1/H_sem, H_sem, device=device).unsqueeze(0).repeat(N_sem, 1)
                    z_coords = torch.linspace(-1+1/D_sem, 1-1/D_sem, D_sem, device=device).unsqueeze(0).repeat(N_sem, 1)
                    coords = torch.stack([torch.stack(torch.meshgrid(x,y,z, indexing='ij')).view(3, -1)
                                        for x, y, z in zip(x_coords,y_coords,z_coords)])
                    coords = coords.transpose(1, 2)
                    semantic_gt_samples = sample_from_3dgrid(semantics, coords, mode='nearest')
                    semantic_gt = semantic_gt_samples.permute(0, 2, 1).reshape(N_sem, C_sem, H_sem, self.num_semantic_y, D_sem).contiguous()
                    semantic_gt = semantic_gt.permute(0, 1, 3, 4, 2)
                    semantic_gt = semantic_gt.reshape(N_sem, -1, H_sem, D_sem)
                else:
                    semantic_gt = None
        return semantic_image, semantic_gt
    
    def create_feature_grid(self, class_labels, translations, sizes, angles, room_layout, semantic_layout=None, obj_latents=None, random_latents=False):
        """
        Sizes and translations need to be in [-0.5, 0.5]
        Args:
            class_labels: (N, OBJ, NUM_C)
            translations: (N, OBJ, 3=(x,y,z))
            sizes: (N, OBJ, 3=(x,y,z))
            angles: (N, OBJ, 1=(theta_z))
            room_layout: (N, 1, H, W)
        Returns:
            feature_grid: (N, C, H, W, D)
        """
        if "semantic_layout" in self.cond_content:
            if semantic_layout.shape[-1] != self.grid_size or semantic_layout.shape[-2] != self.grid_size:
                semantic_layout = torch.nn.functional.interpolate(semantic_layout,
                                                                    size=(self.grid_size, self.grid_size),
                                                                    mode='nearest')
            N, C, H, W = semantic_layout.shape
            semantic_grid = torch.nn.functional.one_hot(semantic_layout.long().squeeze(1), num_classes=self.num_classes).permute(0, 3, 1, 2)
            feature_grid = semantic_grid.unsqueeze(3).float()
            mask_grid = torch.ones((N, 1, H, 1, W), device=semantic_layout.device)
            occupancy_grid = torch.ones((N, 1, H, 1, W), device=semantic_layout.device)
            if self.semantic_resolution != self.feature_resolution:
                semantic_grid = torch.nn.functional.interpolate(
                    semantic_grid.float(),
                    size=(self.semantic_resolution, self.semantic_resolution),
                    mode='nearest'
                    )
            semantic_grid = semantic_grid.unsqueeze(3)
            if 'latent' in self.cond_content:
                if obj_latents is None:
                    latent_grid = torch.zeros((N, self.z_dim_obj, *semantic_layout.shape[-2:]), device=semantic_layout.device)
                    for latent_grid_batch, semantic_layout_batch in zip(latent_grid, semantic_layout):
                        unique_labels = torch.unique(semantic_layout_batch)
                        for label in unique_labels:
                            semantic_mask = (semantic_layout_batch == label).expand(latent_grid_batch.shape)
                            if self.training or random_latents:
                                obj_latent = torch.randn(self.z_dim_obj, device=semantic_layout.device, dtype=latent_grid_batch.dtype)
                            else:
                                obj_latent = torch.from_numpy(np.random.RandomState(0).randn(self.z_dim_obj)).type(latent_grid_batch.dtype).to(semantic_layout.device)
                            obj_latent = obj_latent.unsqueeze(-1).unsqueeze(-1).expand(latent_grid_batch.shape)
                            latent_grid_batch[semantic_mask] = obj_latent[semantic_mask]
                else:
                    latent_grid = obj_latents
                latent_grid = latent_grid.unsqueeze(3)
                feature_grid = torch.cat((feature_grid, latent_grid), 1)
            return feature_grid, mask_grid, semantic_grid, occupancy_grid
        num_vertices = self.grid_size
        num_classes = self.num_classes
        is_voxelize_2d = self.model_type in ['2D', '2D_volume'] and not (self.use_semantic_loss and not self.use_semantic_floor) and self.semantic_floor_type in ['floor', 'top_down'] and not self.concat_occupancy
        feature_grid = []
        if self.model_type in ['2D', '2D_volume']:
            mask_grid = []
        if self.use_semantic_loss:
            semantic_grid = []
        if self.concat_occupancy:
            occupancy_grid = []
        if is_voxelize_2d:
            semantic_dims = (self.semantic_resolution,1,self.semantic_resolution)
        else:
            semantic_dims = (self.semantic_resolution,)*3
        # Per scene
        for i in range(sizes.shape[0]):
            # Per object
            scene_grid = torch.zeros((self.num_feats_voxel_in, *self.dims), device=sizes.device, dtype=sizes.dtype)
            voxel_mask = torch.zeros((1, *self.dims), device=sizes.device, dtype=sizes.dtype)
            voxel_mask_col = torch.ones((3, *self.dims), device=sizes.device, dtype=sizes.dtype)
            voxel_masks = torch.zeros((1, *self.dims), device=sizes.device, dtype=sizes.dtype)
            if self.use_semantic_loss:
                voxel_masks_semantic = torch.zeros((num_classes, *semantic_dims), device=sizes.device, dtype=sizes.dtype)
            if self.concat_occupancy:
                occupancy_batch = torch.zeros((1, *((self.grid_size,)*3)), device=sizes.device, dtype=sizes.dtype)
            # Sort by largest objects
            obj_indices = torch.prod(sizes[i], 1).sort(descending=True)[1]
            for j in obj_indices:
                size = sizes[i,j]
                if torch.count_nonzero(size) == 0 or self.cond_content == ['room_layout']:
                    break
                translation = translations[i,j]
                R = get_rotation_matrix(angles[i,j])
                xp = torch.linspace(- size[0], size[0], num_vertices, device=sizes.device)
                if is_voxelize_2d:
                    yp = size[[1]]
                else:
                    yp = torch.linspace(- size[1], size[1], num_vertices, device=sizes.device)
                zp = torch.linspace(- size[2], size[2], num_vertices, device=sizes.device)
                coords = torch.stack(torch.meshgrid(xp,yp,zp, indexing='ij')).view(3, -1)
                coords = torch.mm(R[0].T, coords) + translation.unsqueeze(-1)
                # Start y axis from bottom of the voxel grid to set floorplan at the bottom
                coords[1] *= -1
                # Clamp because of numerical precision
                coords = coords.clamp(-0.5, 0.5)
                if self.model_type in ['2D', '2D_volume']:
                    occ_grid = voxelize2d(coords.transpose(1,0).unsqueeze(0), self.grid_size).long()
                else:
                    occ_grid = voxelize(coords.transpose(1,0).unsqueeze(0), self.grid_size).long()
                if self.use_semantic_loss:
                    if self.semantic_resolution != self.feature_resolution:
                        if is_voxelize_2d:
                            occ_grid_semantic = voxelize2d(coords.transpose(1,0).unsqueeze(0), self.semantic_resolution).long()
                        else:
                            occ_grid_semantic = voxelize(coords.transpose(1,0).unsqueeze(0), self.semantic_resolution).long()
                    else:
                        occ_grid_semantic = occ_grid.clone()
                if self.concat_occupancy:
                    occ_grid_object = occ_grid.clone()
                if self.model_type in ['2D', '2D_volume']:
                    occ_grid = occ_grid[:,:,[0],:]
                voxel_mask = torch.logical_and(voxel_masks == 0, occ_grid != 0)
                if voxel_mask.sum() == 0:
                    continue
                voxel_masks += voxel_mask
                if self.use_semantic_loss:
                    voxel_mask_semantic = torch.logical_and(voxel_masks_semantic.sum(0, keepdim=True) == 0, occ_grid_semantic != 0)
                    class_label = torch.where(class_labels[i,j])[0][0]
                    voxel_masks_semantic[class_label, voxel_mask_semantic.squeeze(0)] = 1
                if self.concat_occupancy:
                    occupancy_mask = torch.logical_and(occupancy_batch == 0, occ_grid_object != 0)
                    occupancy_batch[occupancy_mask] = 1
                voxel_mask_col[:, voxel_mask[0]] = torch.rand((3,1,1,1), device=voxel_mask.device).expand(-1,*self.dims)[:, voxel_mask[0]]
                obj_grid = []
                if 'class' in self.cond_content:
                    class_labels_grid = occ_grid * torch.where(class_labels[i,j])[0][0]
                    class_labels_grid[voxel_mask.repeat(class_label_embed.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(class_labels_grid)
                if 'binary' in self.cond_content:
                    obj_grid.append(voxel_mask.long())
                if 'class_one_hot' in self.cond_content:
                    class_labels_grid = class_labels[i,j].view(class_labels[i,j].shape[0], 1, 1, 1).repeat(1, *self.dims)
                    class_labels_grid[voxel_mask.repeat(class_labels_grid.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(class_labels_grid)
                if 'class_one_hot_embed' in self.cond_content:
                    class_label_embed = self.class_embedding(class_labels[i,j].unsqueeze(0)).squeeze(0)
                    class_label_embed = class_label_embed.view(class_label_embed.shape[0], 1, 1, 1).repeat(1, *self.dims)
                    class_label_embed[voxel_mask.repeat(class_label_embed.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(class_label_embed)
                if 'latent' in self.cond_content:
                    if obj_latents is None:
                        if self.training or random_latents:
                            obj_latent = torch.randn(self.z_dim_obj, device=occ_grid.device)
                        else:
                            obj_latent = torch.from_numpy(np.random.RandomState(0).randn(self.z_dim_obj)).to(occ_grid.device)
                            
                    else:
                        obj_latent = obj_latents[i,j].unsqueeze(0)
                    obj_latent = obj_latent.reshape(self.z_dim_obj, 1, 1, 1).repeat(1, *voxel_mask.shape[1:])
                    obj_latent[voxel_mask.repeat(obj_latent.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(obj_latent)
                if 'latent_embed' in self.cond_content:
                    if obj_latents is None:
                        obj_latent = torch.randn(1, self.z_dim_obj, device=occ_grid.device)
                    else:
                        obj_latent = obj_latents[i,j].unsqueeze(0)
                    obj_latent = self.obj_embedding(obj_latent).squeeze(0)
                    obj_latent = obj_latent.view(obj_latent.shape[0], 1, 1, 1).repeat(1, *voxel_mask.shape[1:])
                    obj_latent[voxel_mask.repeat(obj_latent.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(obj_latent)
                if 'global_coords' in self.cond_content:
                    # TODO: Check if coords[1] *= -1
                    grid_vec = torch.arange(0, self.grid_size, 1, device=occ_grid.device)
                    if self.model_type in ['2D', '2D_volume']:
                        grid_vec_y = torch.ones(1, device=grid_vec.device, dtype=grid_vec.dtype) * (self.grid_size - 1)
                        global_coords = torch.stack(torch.meshgrid(grid_vec, grid_vec_y, grid_vec, indexing='ij'))
                    else:
                        global_coords = torch.stack(torch.meshgrid(grid_vec, grid_vec, grid_vec, indexing='ij'))
                    # Normalize [-0.5, 0.5]
                    global_coords = global_coords / (self.grid_size - 1) - 0.5
                    # Set background voxels to zero, TODO: Maybe leave global coords everywhere
                    global_coords[voxel_mask.repeat(global_coords.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(global_coords)
                if 'local_coords' in self.cond_content:
                    # TODO: Check if coords[1] *= -1
                    unique_ind = torch.stack(torch.where(voxel_mask[0]))
                    min_coords = unique_ind.min(dim=1)[0]
                    max_coords = unique_ind.max(dim=1)[0]
                    voxel_diff = max_coords - min_coords
                    grid_vec = torch.arange(0, self.grid_size, 1, device=occ_grid.device)
                    if self.model_type in ['2D', '2D_volume']:
                        grid_vec_y = torch.ones(1, device=grid_vec.device, dtype=grid_vec.dtype)
                        voxel_diff[1] = 1 # avoid division by zero
                        global_coords = torch.stack(torch.meshgrid(grid_vec, grid_vec_y, grid_vec, indexing='ij'))
                    else:
                        global_coords = torch.stack(torch.meshgrid(grid_vec, grid_vec, grid_vec, indexing='ij'))
                    # Refer coordinates to first local index
                    local_coords = global_coords - min_coords.view(3, 1, 1, 1)
                    # Normalize [-0.5, 0.5]
                    local_coords = local_coords / voxel_diff.view(3, 1, 1, 1) - 0.5
                    # Set background voxels to zero
                    local_coords[voxel_mask.repeat(local_coords.shape[0], 1, 1, 1) == 0] = 0
                    obj_grid.append(local_coords)

                if 'local_coords_canonical' in self.cond_content:
                    local_coords_canonical = torch.zeros(voxel_mask.shape, device=voxel_mask.device).repeat(3, 1, 1, 1)
                    # Scale mask indices between [0.0, 1.0]
                    mask_ind = torch.where(voxel_mask[0])
                    max_ind = torch.tensor(voxel_mask.shape[1:], device=local_coords_canonical.device).unsqueeze(1) - 1
                    # Avoid division by 0,  e.g., when y is only one dimensionals
                    max_ind[max_ind==0] = 1
                    unique_ind = torch.stack(mask_ind)/max_ind
                    # Compute mean of object and translate object by mean object coordinates
                    mean_ind = unique_ind.mean(1, keepdim=True)
                    unique_ind_shifted = unique_ind - mean_ind
                    # Rotate object back
                    R = get_rotation_matrix(-angles[i,j])
                    unique_coords_rot = torch.mm(R[0].T, unique_ind_shifted)

                    # Scale values in [0.0, 1.0]
                    min_coords = unique_coords_rot.min(dim=1, keepdim=True)[0]
                    max_coords = unique_coords_rot.max(dim=1, keepdim=True)[0]
                    shift_vector = torch.tensor([0.5]*3, device=unique_ind.device).unsqueeze(1)
                    diff_coords = max_coords - min_coords
                    # Handle cases where coordinate differences is zero, e.g., when y is only one dimensional
                    is_diff_coords_zero = diff_coords==0
                    max_coords[is_diff_coords_zero] = 1
                    shift_vector[is_diff_coords_zero] = 0
                    diff_coords[is_diff_coords_zero] = 1
                    unique_coords_scaled = ((unique_coords_rot - min_coords)/diff_coords)
                    # Shift values to [-0.5, 0.5]
                    unique_coords_scaled_shifted = unique_coords_scaled - shift_vector
                    local_coords_canonical.permute(1, 2, 3, 0)[mask_ind] = unique_coords_scaled_shifted.transpose(1, 0)
                    # Clamp for numerical inprecision
                    local_coords_canonical = local_coords_canonical.clamp(-0.5, 0.5)
                    obj_grid.append(local_coords_canonical)

                obj_grid = torch.cat(obj_grid)
                scene_grid = scene_grid + obj_grid
            if 'room_layout' in self.cond_content:
                room_layout_res = torch.nn.functional.interpolate(room_layout[i].unsqueeze(0),
                                                                  size=(self.grid_size, self.grid_size),
                                                                  mode='nearest').squeeze(0)
                if self.model_type in ['2D', '2D_volume']:
                    room_layout_res = room_layout_res.unsqueeze(2)
                else:
                    room_layout_res = room_layout_res.unsqueeze(2).repeat(1, 1, self.grid_size, 1)
                if self.add_floor_class and self.use_semantic_loss:
                    floor_only_mask = torch.logical_and(voxel_masks.bool() == False, room_layout_res.bool())
                    if self.semantic_resolution != self.grid_size:
                        floor_only_mask = torch.nn.functional.interpolate(floor_only_mask.transpose(1,2).float(),
                                                                        size=(self.semantic_resolution, self.semantic_resolution),
                                                                        mode='nearest').transpose(1,2).bool()
                    voxel_masks_semantic[-1][floor_only_mask.squeeze(0)] = 1
                if self.add_none_class and self.use_semantic_loss:
                    none_mask = torch.logical_and(voxel_masks.bool() == False, room_layout_res.bool() == False)
                    if self.semantic_resolution != self.grid_size:
                        none_mask = torch.nn.functional.interpolate(none_mask.transpose(1,2).float(),
                                                                        size=(self.semantic_resolution, self.semantic_resolution),
                                                                        mode='nearest').transpose(1,2).bool()
                    voxel_masks_semantic[-2][none_mask.squeeze(0)] = 1

                scene_grid = torch.cat((scene_grid, room_layout_res), dim=0)
            if self.use_layout_latent:
                layout_latent = torch.randn(self.z_dim_obj, 1, 1, 1, device=occ_grid.device).repeat(1, *room_layout_res.shape[1:])
                layout_latent[room_layout_res.repeat(layout_latent.shape[0], 1, 1, 1) == 0] = 0
                scene_grid = torch.cat((scene_grid, layout_latent), dim=0)
            feature_grid.append(scene_grid)
            if self.model_type in ['2D', '2D_volume']:
                voxel_masks_layout = torch.where(torch.logical_or(voxel_masks.bool(), room_layout_res.bool()), voxel_mask_col, torch.zeros_like(voxel_mask_col))
                mask_grid.append(voxel_masks_layout)
            if self.use_semantic_loss:
                semantic_grid.append(voxel_masks_semantic)
            if self.concat_occupancy:
                occupancy_grid.append(occupancy_batch)
        feature_grid = torch.stack(feature_grid)
        if self.model_type in ['2D', '2D_volume']:
            mask_grid = torch.stack(mask_grid)
        else:
            mask_grid = None
        if self.use_semantic_loss:
            semantic_grid = torch.stack(semantic_grid)
        else:
            semantic_grid = None
        if self.concat_occupancy:
            occupancy_grid = torch.stack(occupancy_grid)
        else:
            occupancy_grid = None
        return feature_grid, mask_grid, semantic_grid, occupancy_grid

@persistence.persistent_class
class BoxEncoder(torch.nn.Module):
    def __init__(
        self,
        feature_type,
        in_channels = 1,
        f_maps = 32,
        num_levels = 5,
        encoder_mod=False,
        **synthesis_kwargs,
    ):
        super().__init__()
        self.num_ws = 0
        if feature_type in ['planes', 'floor_plan']:
            if encoder_mod:
                self.model = UNet2DEncoderMod(
                    in_channels, f_maps=f_maps, num_levels=num_levels,
                    architecture='orig',
                    **synthesis_kwargs)
                self.num_ws += self.model.num_ws
            else:
                self.model = UNet2DEncoder(in_channels, f_maps=f_maps, num_levels=num_levels)
        elif feature_type == 'volume':
            self.model = UNet3DEncoder(in_channels, f_maps=f_maps, num_levels=num_levels, layer_order='cr')
        else:
            raise ValueError(f'Unknown feature type {feature_type}')
        self.num_out_channels = self.model.num_out_channels
        self.encoder_mod = encoder_mod
    
    def forward(self, x, ws=None, **synthesis_kwargs):
        if self.encoder_mod:
            out = self.model(x, ws, **synthesis_kwargs)
        else:
            out = self.model(x)
        out = {feat.shape[-1]:feat for feat in out}
        return out

class OSGDecoder(torch.nn.Module):
    def __init__(
        self,
        n_features = 32,
        hidden_dim = 64,
        decoder_output_dim = 32,
        n_hidden_layers = 1,
        decoder_lr_mul = 1
            ):
        super().__init__()
        assert n_hidden_layers >= 1
        self.hidden_dim = hidden_dim
        self.out_dim = 1 + decoder_output_dim
        layers = []
        layers.append(FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=decoder_lr_mul))
        layers.append(torch.nn.Softplus())
        for _ in range(n_hidden_layers):
            layers.append(FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=decoder_lr_mul))
            layers.append(torch.nn.Softplus())
        layers.append(FullyConnectedLayer(self.hidden_dim, self.out_dim, lr_multiplier=decoder_lr_mul))
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, sampled_features, ray_directions):
        """
        sampled_features: (N, n_planes, M, C) or (N, M, C)
        """
        # Aggregate features if feature planes
        if sampled_features.ndim == 4:
            sampled_features = sampled_features.mean(1)
        x = sampled_features.contiguous()

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

class EmbeddingNetwork(torch.nn.Module):
    def __init__(
        self,
        in_dim = 23,
        hidden_dim = 32,
        out_dim = 16,
        n_hidden_layers = 1,
        decoder_lr_mul = 1
            ):
        super().__init__()
        assert n_hidden_layers >= 1
        self.hidden_dim = hidden_dim
        layers = []
        layers.append(FullyConnectedLayer(in_dim, self.hidden_dim, lr_multiplier=decoder_lr_mul))
        layers.append(torch.nn.LeakyReLU())
        for _ in range(n_hidden_layers):
            layers.append(FullyConnectedLayer(self.hidden_dim, self.hidden_dim, lr_multiplier=decoder_lr_mul))
            layers.append(torch.nn.LeakyReLU())
        layers.append(FullyConnectedLayer(self.hidden_dim, out_dim, lr_multiplier=decoder_lr_mul))
        self.net = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.net(x)
        return x