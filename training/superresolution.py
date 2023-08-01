# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import torch
from torch_utils.ops import upfirdn2d
from torch_utils import persistence
from torch_utils import misc

from training.networks import SynthesisBlock
import numpy as np

#----------------------------------------------------------------------------
@persistence.persistent_class
class SuperResolutionShared(torch.nn.Module):
    def __init__(self, in_channels, w_dim, input_resolution, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=32768, channel_max=512, img_channels=3,
                **block_kwargs):
        super().__init__()

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = input_resolution
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.input_resolution_log2 = int(np.log2(input_resolution))
        self.sr_antialias = sr_antialias
        self.num_ws = 0
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.block_resolutions = [2 ** i for i in range(self.input_resolution_log2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        self.blocks = torch.nn.ModuleList()
        for res in self.block_resolutions:
            not_first = res != self.input_resolution
            in_channels = channels_dict[res // 2] if not_first else in_channels
            out_channels = channels_dict[res]
            up = 2 if not_first else 1
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),
                up=up, **block_kwargs)
            self.blocks.append(block)

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        for i, block in enumerate(self.blocks):
            up_sample = False if i == 0 else True
            x, rgb = block(x, rgb, ws, up_sample=up_sample, **block_kwargs)
        return rgb

#----------------------------------------------------------------------------
@persistence.persistent_class
class SuperResolution(torch.nn.Module):
    def __init__(self, in_channels, w_dim, input_resolution, img_resolution, sr_num_fp16_res, sr_antialias,
                num_fp16_res=4, conv_clamp=None, channel_base=32768, channel_max=512, img_channels=3,
                **block_kwargs):
        super().__init__()

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = input_resolution
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.input_resolution_log2 = int(np.log2(input_resolution))
        self.sr_antialias = sr_antialias
        self.num_ws = 0
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.block_resolutions = [2 ** i for i in range(self.input_resolution_log2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        for res in self.block_resolutions:
            is_last = (res == self.img_resolution)
            not_first = res != self.input_resolution
            in_channels = channels_dict[res // 2] if not_first else in_channels
            out_channels = channels_dict[res]
            up = 2 if not_first else 1
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None),
                up=up, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, rgb, x, ws, **block_kwargs):
        block_ws = []
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        for i, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            block = getattr(self, f'b{res}')
            up_sample = False if i == 0 else True
            x, rgb = block(x, rgb, cur_ws, up_sample=up_sample, **block_kwargs)
        return rgb