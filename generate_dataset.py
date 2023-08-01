import torch
import random
import urllib.request
import imageio as imio
from einops import rearrange
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import PIL.Image

import dnnlib
import legacy
from training.generator import Generator
from torch_utils import misc
from renderer import Renderer
from train import init_dataset_kwargs

from generate import get_eval_labels

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

def generate_sample_videos(
    outdir: str,
    data: str,
    network_pkl: str,
    rand_coord: bool,
    num_z_seeds: int = 1,
    num_layout_indices: int = 1,
    num_coords_seed=40,
    dataset_name='3dfront',
    start_coords_idx=0,
    num_images=50000,
    random_latents=False,
):
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        D = network['D'].to(device)

    # avoid persistent classes... 
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        G = G2

    # Training set
    training_set_kwargs, _ = init_dataset_kwargs(data=data)
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.dataset_name = dataset_name
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    max_coords = training_set.max_coords
    count_images = 0
    total_idx = 0
    while True:
        for seq_idx in range(num_layout_indices):
            for seed_idx in range(num_z_seeds):
                z = torch.randn(1, G.z_dim, device=device)
                if start_coords_idx is None:
                    start_coords_idx = random.randint(0, training_set.num_samples_per_scene - num_coords_seed - 1)
                for coords_idx in range(start_coords_idx, start_coords_idx + num_coords_seed):
                    if rand_coord:
                        coords_idx = random.randint(start_coords_idx, training_set.num_samples_per_scene - num_coords_seed - 1)
                    c, _, seq_name, = get_eval_labels(training_set, layout_idx=seq_idx, coords_idx=coords_idx, num_eval_seeds=1, device=device, out_image=False)
                    c = c[0][0]
                    seq_name = seq_name[0][0]
                    seq_name = seq_name.split("/")[-2]
                    with torch.no_grad():
                        img = G(z, c, noise_mode='const', rand_render=False, random_latents=random_latents)['image']
                    img = img.squeeze(0).detach().cpu().numpy()
                    out_path_seq = os.path.join(outdir, f"{seq_name}_{total_idx}_{seed_idx}")
                    out_path_file = os.path.join(out_path_seq, f"{(coords_idx-start_coords_idx):04d}.png")
                    os.makedirs(out_path_seq, exist_ok=True)
                    save_image_grid(img, out_path_file, drange=[-1, 1])
                    count_images += 1
                    if count_images == num_images:
                        exit()
        total_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', help='Network pickle filename', required=True)
    parser.add_argument('--num_z_seeds', type=int, help='Num seeds', default=1)
    parser.add_argument('--num_layout_indices', type=int, default=2)
    parser.add_argument('--num_coords_seed', type=int, default=40)
    parser.add_argument('--num_images', type=int, default=50000)
    parser.add_argument('--start_coords_idx', type=int, default=None)
    parser.add_argument('--rand_coord', action="store_true")
    parser.add_argument('--random_latents', action="store_true")
    parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
    parser.add_argument('--data', type=str)
    parser.add_argument('--dataset_name', type=str, default="3dfront")
    args = parser.parse_args()
    generate_sample_videos(**vars(args))
