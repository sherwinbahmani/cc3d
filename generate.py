
import os
import re
import glob
from typing import List, Optional
import argparse
import dnnlib
import numpy as np
import torch
import PIL
import legacy
from training.generator import Generator
from torch_utils import misc
from renderer import Renderer
from training.training_loop import save_image_grid
from train import init_dataset_kwargs
from training.utils import create_voxel_grid
from tqdm import tqdm


#----------------------------------------------------------------------------

def get_eval_labels(training_set, layout_idx=None, coords_idx=None, num_eval_seeds=1, num_coords_seed=1, device='cuda', out_image=False, out_video=False, use_coords_traj=False):
    num_scenes = training_set.num_labels
    if layout_idx is None:
        layout_idx = np.random.randint(num_scenes)
    if coords_idx is None:
        if use_coords_traj or len(training_set) > num_coords_seed:
            eval_coords_indices = [np.arange(training_set.num_samples_per_scene)[:num_coords_seed] for _ in range(num_eval_seeds)]
        else:
            eval_coords_indices = [np.zeros(1).repeat(num_coords_seed).astype(int) for _ in range(num_eval_seeds)]
    else:
        eval_coords_indices = [[coords_idx] for _ in range(num_eval_seeds)]
    label_ratio = training_set.img_per_scene_ratio / training_set.num_samples_per_scene
    eval_scene_indices = [np.floor(layout_idx*training_set.img_per_scene_ratio).astype(int) for _ in range(num_eval_seeds)]
    eval_indices = [[(eval_scene_indices[ind] + int(np.floor(i*label_ratio)), i) for i in eval_coords_indices[ind]] for ind in range(num_eval_seeds)]
    if out_image:
        eval_real_imgs = [[training_set.get_image(i) for (i, _) in coords] for coords in eval_indices]
    else:
        eval_real_imgs = None
    eval_c = [[training_set.get_label(np.floor(coords[0][0]/training_set.img_per_scene_ratio).astype(int), j) for (i, j) in coords] for coords in eval_indices]
    eval_label_names = [[training_set._label_fnames[np.floor(coords[0][0]/training_set.img_per_scene_ratio).astype(int)] for (i, j) in coords] for coords in eval_indices]
    eval_c = [[misc.dict_to_device(c, device) for c in eval_c_i] for eval_c_i in eval_c]
    eval_c = [[misc.add_batch_dim_dict(c) for c in eval_c_i] for eval_c_i in eval_c]
    return eval_c, eval_real_imgs, eval_label_names

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def generate_sample_videos(
    outdir: str,
    data: str,
    network_pkl: str = None,
    num_z_seeds: int = 1,
    z_seed: int = None,
    num_layout_indices: int = 1,
    rand_layout=False,
    render_program=None,
    num_coords_seed=1,
    rand_seed=False,
    G=None,
    D=None,
    device=None,
    out_video=False,
    use_coords_traj=False,
    dataset_name='3dfront'
):
    if device is None:
        device = torch.device('cuda')
    if network_pkl is not None:
        if os.path.isdir(network_pkl):
            network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
        
        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device) # type: ignore
            D = network['D'].to(device)
    else:
        assert G is not None and D is not None, f"Need to specify either network pkl or G and D"

    # avoid persistent classes... 
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, D, program=render_program)

    # Training set
    training_set_kwargs, _ = init_dataset_kwargs(data=data)
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.dataset_name = dataset_name
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    max_coords = training_set.max_coords
    out_dir_program = os.path.join(outdir, render_program)
    out_dir_real = os.path.join(outdir, render_program)
    os.makedirs(out_dir_program, exist_ok=True)
    if rand_layout:
        layout_indices = list(np.random.randint(0, len(training_set._label_fnames), size=num_layout_indices))
    else:
        layout_indices = list(range(num_layout_indices))
    G2.set_random_seed(0)
    if render_program == "fake_top_down":
        num_coords_seed = 1
    if z_seed is None:
        if rand_seed:
            z_seeds = [None]*num_z_seeds
        else:
            z_seeds = list(range(num_z_seeds))
    else:
        z_seeds = [z_seed]

    for layout_idx in layout_indices:
        for z_seed in z_seeds:
            if z_seed is None:
                z_seed = torch.randint(0, 10000000, (1,)).item()
                # Set manual seed
                # z_seed = 0
            z = torch.from_numpy(np.random.RandomState(z_seed).randn(1, G.z_dim)).to(device)
            c, imgs, label_names = get_eval_labels(training_set, device=device, layout_idx=layout_idx, num_coords_seed=num_coords_seed, out_video=out_video, use_coords_traj=use_coords_traj)
            with torch.no_grad():
                G2(z=z, c=c, layout_idx=layout_idx, out_path=out_dir_program, img=imgs, max_coords=max_coords, training_set=training_set, label_names=label_names,z_seed=z_seed, out_video=out_video)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_pkl', help='Network pickle filename', required=True)
    parser.add_argument('--z_seed', type=int, help='Random seed', default=None)
    parser.add_argument('--num_z_seeds', type=int, help='Num seeds', default=1)
    parser.add_argument('--num_layout_indices', type=int, default=1)
    parser.add_argument('--num_coords_seed', type=int, default=1)
    parser.add_argument('--rand_seed', action='store_true')
    parser.add_argument('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
    parser.add_argument('--data', type=str)
    parser.add_argument('--render_program', default=None)
    parser.add_argument('--rand_layout', action='store_true')
    parser.add_argument('--out_video', action='store_true')
    parser.add_argument('--dataset_name', type=str, default="3dfront")
    parser.add_argument('--use_coords_traj', action='store_true')
    args = parser.parse_args()
    generate_sample_videos(**vars(args))

#----------------------------------------------------------------------------