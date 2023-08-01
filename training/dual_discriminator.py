# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Discriminator architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import math
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from einops import rearrange
from training.networks import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue, Conv2dLayer
from training.unet2d import UNet2D, UNet2DMod

@persistence.persistent_class
class DualDiscriminatorVideo(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 use_semantic_loss,
                 **kwargs,
                 ):
        super().__init__()
        assert img_channels % 2 == 0
        self.img_channels = img_channels
        # Only compute semantic in video discriminator
        self.video_disc = DualDiscriminator(c_dim, img_resolution, img_channels, use_semantic_loss=use_semantic_loss, **kwargs)
        self.image_disc = DualDiscriminator(c_dim, img_resolution, img_channels//2, use_semantic_loss=False, **kwargs)

    def forward(self, vid_dict, c, compute_semantic=False, **block_kwargs):
        x_vid = self.video_disc(vid_dict, c, compute_semantic=compute_semantic, **block_kwargs)
        img_dict = {}
        for img_type in ['image', 'image_raw']:
            x = vid_dict[img_type]
            img_dict[img_type] = torch.cat((x[:, :self.img_channels//2], x[:, self.img_channels//2:]), 0)
        if isinstance(c, torch.Tensor):
            c = torch.cat((c, c), 0)
        x_img = self.image_disc(img_dict, c, compute_semantic=False, **block_kwargs)
        return {"x": x_img['x'], "x_semantic": x_vid['x_semantic'], "x_video": x_vid['x']}

@persistence.persistent_class
class SingleDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        sr_upsample_factor  = 1,        # Ignored for SingleDiscriminator
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        use_semantic_loss   = False,
        use_semantic_floor   = False,
        semantic_resolution = None,
        feature_resolution = None,
        num_semantic_layers = 3,
        semantic_start_channels=16,
        in_channels_semantic = None,
        num_classes         = None,
        channels_last       = False,
        img_pairs = False,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.use_semantic_loss = use_semantic_loss
        self.use_semantic_floor = use_semantic_floor
        self.semantic_resolution = semantic_resolution
        self.feature_resolution = feature_resolution
        self.img_pairs = img_pairs
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        if self.use_semantic_loss:
            if self.use_semantic_floor:
                res_rate = self.feature_resolution // self.semantic_resolution
                self.semantic_layers = UNet2D(in_channels_semantic, num_classes, num_levels=num_semantic_layers, res_rate=res_rate, f_maps=semantic_start_channels)
            else:
                in_channels_semantic = channels_dict[self.semantic_resolution // 2]
                self.semantic_layers = torch.nn.Sequential()
                for i in range(num_semantic_layers):
                    if i == 0:
                        in_channels = in_channels_semantic
                        out_channels = num_classes
                    else:
                        in_channels = num_classes
                        out_channels = num_classes
                    layer = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation='lrelu',
                        trainable=True, conv_clamp=conv_clamp, channels_last=channels_last)
                    self.semantic_layers.append(layer)

    def forward(self, img, c, update_emas=False, decode=False, compute_semantic=False, **block_kwargs):
        if compute_semantic:
            semantic_img = img['image_semantic_sampled']
        img = img['image']
        _ = update_emas # unused
        x = None
        x_semantic = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.semantic_resolution:
                if self.use_semantic_loss and not self.use_semantic_floor and compute_semantic:
                    x_semantic = self.semantic_layers(x)
                else:
                    x_semantic = None
            x, img = block(x, img, **block_kwargs)
        
        if self.use_semantic_loss and self.use_semantic_floor and compute_semantic:
            x_semantic = self.semantic_layers(semantic_img)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return {"x": x, "x_semantic": x_semantic}

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f=None, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered
        
    return ada_filtered_64

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        use_semantic_loss   = False,
        use_semantic_floor   = False,
        semantic_resolution = None,
        feature_resolution = None,
        num_semantic_layers = 3,
        semantic_start_channels=16,
        in_channels_semantic = None,
        num_classes         = None,
        channels_last       = False,
        img_pairs = False,
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.use_semantic_loss = use_semantic_loss
        self.use_semantic_floor = use_semantic_floor
        self.semantic_resolution = semantic_resolution
        self.feature_resolution = feature_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.img_pairs = img_pairs
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise
        if self.use_semantic_loss:
            if self.use_semantic_floor:
                res_rate = self.feature_resolution // self.semantic_resolution
                self.semantic_layers = UNet2D(in_channels_semantic, num_classes, num_levels=num_semantic_layers, res_rate=res_rate, f_maps=semantic_start_channels)
            else:
                in_channels_semantic = channels_dict[self.semantic_resolution // 2]
                self.semantic_layers = torch.nn.Sequential()
                for i in range(num_semantic_layers):
                    if i == 0:
                        in_channels = in_channels_semantic
                        out_channels = num_classes
                    else:
                        in_channels = num_classes
                        out_channels = num_classes
                    layer = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation='lrelu',
                        trainable=True, conv_clamp=conv_clamp, channels_last=channels_last)
                    self.semantic_layers.append(layer)

    def forward(self, img, c, update_emas=False, decode=False, compute_semantic=False, **block_kwargs):
        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
        if compute_semantic:
            semantic_img = img['image_semantic_sampled']
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        x_semantic = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            if res == self.semantic_resolution:
                if self.use_semantic_loss and not self.use_semantic_floor and compute_semantic:
                    x_semantic = self.semantic_layers(x)
                    
            x, img = block(x, img, **block_kwargs)
        
        if self.use_semantic_loss and self.use_semantic_floor and compute_semantic:
            x_semantic = self.semantic_layers(semantic_img)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return {"x": x, "x_semantic": x_semantic}

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualDiscriminatorRecon(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        use_semantic_loss   = False,
        semantic_resolution = None,
        num_semantic_layers = 1,
        num_classes         = None,
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise
        self.decoder = ConvDecoder(in_channel=in_channels, out_channel=img_channels, in_res=4, out_res=self.img_resolution)

    def forward(self, img, c, update_emas=False, decode=False, **block_kwargs):
        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter)
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        if decode:
            recon = self.decoder(x)
        else:
            recon = None
        x = self.b4(x, img, cmap)
        return x, recon

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
@persistence.persistent_class
class ConvDecoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel, in_res, out_res, activation='lrelu'):
        super().__init__()

        log_size_in = int(math.log(in_res, 2))
        log_size_out = int(math.log(out_res, 2))

        self.layers = []
        in_ch = in_channel
        for i in range(log_size_in, log_size_out):
            out_ch = in_ch // 2
            self.layers.append(
                Conv2dLayer(
                    in_channels=in_ch, out_channels=out_ch, kernel_size=3, up=2, bias=True, activation=activation
                )
            )
            in_ch = out_ch

        self.layers.append(
            Conv2dLayer(in_channels=in_ch, out_channels=out_channel, kernel_size=3, bias=True, activation='linear')
        )
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

#----------------------------------------------------------------------------

@persistence.persistent_class
class DummyDualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        img_channels *= 2

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

        self.raw_fade = 1

    def forward(self, img, c, update_emas=False, **block_kwargs):
        self.raw_fade = max(0, self.raw_fade - 1/(500000/32))

        image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter) * self.raw_fade
        img = torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
