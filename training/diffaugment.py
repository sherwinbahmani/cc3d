# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def DiffAugment(x, policy='color,translation,rotate', channels_first=True, num_disc_frames=1, num_dual_frames=2):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x, num_disc_frames)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    # Split into image and image_raw
    x = rearrange(x, '(b t) c h w -> t b c h w', t=num_dual_frames)
    return x


def rand_brightness(x, num_frames):
    x = x + (torch.rand(x.size(0)// num_frames, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,num_frames,1,1,1).view(-1,1,1,1) - 0.5)
    return x


def rand_saturation(x, num_frames):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0) // num_frames, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,num_frames,1,1,1).view(-1,1,1,1) * 2) + x_mean
    return x


def rand_contrast(x, num_frames):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0) // num_frames, 1, 1, 1, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1,num_frames,1,1,1).view(-1,1,1,1) + 0.5) + x_mean
    return x


def rand_translation(x, num_frames, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0) // num_frames, 1, 1], device=x.device).unsqueeze(1).repeat(1,num_frames,1,1).view(-1,1,1)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0)// num_frames, 1, 1], device=x.device).unsqueeze(1).repeat(1,num_frames,1,1).view(-1,1,1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        (torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device)),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, num_frames, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0) // num_frames, 1, 1], device=x.device).unsqueeze(1).repeat(1,num_frames,1,1).view(-1,1,1)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0) // num_frames, 1, 1], device=x.device).unsqueeze(1).repeat(1,num_frames,1,1).view(-1,1,1)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def rotate(x, num_frames, angles = [0, 90, 180, 270]):
    angles = torch.tensor(angles, device=x.device)
    rand_angles = angles[torch.randint(0, len(angles), size=[x.shape[0]//num_frames])].unsqueeze(1).repeat(1,num_frames).view(-1)
    x_out = torch.empty_like(x)
    for i, angle in enumerate(rand_angles):
        if angle == 0:
            x_out[i] = x[i]
        elif angle == 90:
            x_out[i] = x[i].transpose(1, 2)
        elif angle == 180:
            x_out[i] = x[i].flip(1)
        elif angle == 270:
            x_out[i] = x[i].transpose(1, 2).flip(2)
    return x_out


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'rotate': [rotate],
}