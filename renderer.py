import torch
import torch.nn.functional as F
import numpy as np
from training.utils import create_voxel_grid
from training.training_loop import save_image_grid, save_video_grid
from generate_utils import get_camera_target_coords
import PIL
import os
import torchvision

from visu.kitti import KITTI_LABELS


class Renderer(object):

    def __init__(self, generator, discriminator=None, program=None):
        self.generator = generator
        self.discriminator = discriminator
        self.program = program
        self.seed = 0
        self.program = program.split(':')[0]

    def set_random_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def __call__(self, *args, **kwargs):
        self.generator.eval()  # eval mode...
        with torch.no_grad():
            outputs = getattr(self, f"render_{self.program}")(*args, **kwargs)
        return outputs
    
    def set_top_down_camera(self, c):
        c["camera_coords"][:,[0,2]]= 0
        c["target_coords"][:,[0,2]]= 0
        c["target_coords"][:,1] = -1
        c["camera_coords"][:,1] = 1
        return c

    def norm_image(self, img, drange=[-1, 1]):
        lo, hi = drange
        img = (img - lo) * (255 / (hi - lo))
        img = torch.round(img).clip(0, 255).to(torch.uint8)
        return img
    
    def concat_arrs(self, arr, axis=2):
        return np.concatenate(arr, axis=axis)
    
    def get_labels(self, c):
        labels = {k:v.squeeze(0)for k, v in c.items() if k not in ['camera_coords', 'target_coords', 'label_idx', 'coords_idx']}
        return labels
    
    def compute_voxel_mask(self, c, out_scale):
        labels = self.get_labels(c)
        voxel_masks = create_voxel_grid(**labels, colored=True, out_scale=out_scale)[:, :, 0, :]
        return voxel_masks
    
    def compute_floor_object_mask(self, c, out_scale=255.):
        voxel_masks = self.compute_voxel_mask(c, out_scale=out_scale)
        return voxel_masks
    
    def render_real(self, img, *args, **kwargs):
        out_path = os.path.join(kwargs['out_path'], f"{kwargs['layout_idx']}.png")
        imgs = self.concat_arrs([self.concat_arrs(img_seed, axis=2) for img_seed in img], axis=1)
        save_image_grid(imgs, out_path, drange=[0, 255])
    
    def render_real_top_down(self, z, c, label_names, floor_col=0.5, out_scale=255., *args, **kwargs):
        out_path = os.path.join(kwargs['out_path'], f"{kwargs['layout_idx']}.png")
        imgs = []
        for c_j, label_names_j in zip(c, label_names):
            for c_i, label_names_i in zip(c_j, label_names_j):
                img = self.compute_floor_object_mask(c_i, out_scale=out_scale)
                out_dir = out_path = os.path.join(kwargs['out_path'], label_names_i.split("/")[-2])
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{kwargs['layout_idx']}.png")
                img = img.cpu().numpy()
                imgs.append(img)
                break
        imgs = self.concat_arrs(imgs)
        save_image_grid(imgs, out_path, drange=[0, 1])
    
    def render_real_kitti_layout(self, z, c, label_names, floor_col=0.5, *args, **kwargs):
        s_cols = {}
        for s_i in range(59):
            for kitti_label in KITTI_LABELS:
                if kitti_label.id == s_i:
                    s_cols[s_i] = torch.tensor(kitti_label.color, device=z.device).view(3, 1, 1).float()
        imgs = []
        for c_j, label_names_j in zip(c, label_names):
            for c_i, label_names_i in zip(c_j, label_names_j):
                semantic_layout = c_i["semantic_layout"].squeeze(0)
                semantic_layout_col = torch.zeros((3, *semantic_layout.shape[1:]), device=semantic_layout.device)
                for s_ind in torch.unique(semantic_layout):
                    s_mask = (semantic_layout == s_ind).repeat(3,1,1)
                    s_col = s_cols[s_ind.item()].repeat(1, *semantic_layout.shape[1:])
                    semantic_layout_col[s_mask] = s_col[s_mask]
                img = (semantic_layout_col).cpu().to(torch.uint8).numpy()
                out_dir = out_path = os.path.join(kwargs['out_path'], label_names_i.split("/")[-2])
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{kwargs['layout_idx']}.png")
                imgs.append(img)
                break
        imgs = self.concat_arrs(imgs)
        save_image_grid(imgs, out_path, drange=[0, 255])
    
    def render_real_layout(self, z, c, label_names, floor_col=0.5, *args, **kwargs):
        imgs = []
        for c_j, label_names_j in zip(c, label_names):
            for c_i, label_names_i in zip(c_j, label_names_j):
                img = c_i['room_layout'].squeeze(1).repeat(3, 1, 1)
                out_dir = out_path = os.path.join(kwargs['out_path'], label_names_i.split("/")[-2])
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{kwargs['layout_idx']}.png")
                img = img.cpu().numpy()
                imgs.append(img)
                break
        imgs = self.concat_arrs(imgs)
        save_image_grid(imgs, out_path, drange=[0, 1])
    
    def render_fake(self, z, c, label_names, z_seed, outs=['image', 'image_raw', 'image_depth'],
                    top_down=False, video_dataset=False, image_dataset=False, render_depth=False,
                    neural_rendering_resolution=None, out_video=False, *args, **kwargs):
        label_names_coords = label_names[0]
        imgs = {k: [] for k in outs}
        for eval_c_i in c:
            for c_i in eval_c_i:
                if top_down:
                    c_i = self.set_top_down_camera(c_i)
                img = self.generator(z, c_i, noise_mode='none', rand_render=False, norm_depth=True,
                                     render_depth=render_depth, neural_rendering_resolution=neural_rendering_resolution)
                for k, v in imgs.items():
                    v.append(img[k].squeeze(0).cpu().numpy())
        for k, v in imgs.items():
            for i, arr in enumerate(v):
                label_name = label_names_coords[i].split('/')[-2]
                if video_dataset:
                    out_path_dir = os.path.join(kwargs['out_path'], f'{label_name}_{z_seed}')
                elif image_dataset:
                    out_path_file = os.path.join(kwargs['out_path'], f'{label_name}_{z_seed}_{i:04d}.png')
                else:
                    out_path_dir = os.path.join(kwargs['out_path'], k, label_name, str(z_seed))
                if not image_dataset:
                    os.makedirs(out_path_dir, exist_ok=True)
                    out_path_file = os.path.join(out_path_dir, f"{i:04d}.png")
                if not out_video:
                    save_image_grid(arr, out_path_file, drange=[-1, 1])
            if out_video:
                out_path_dir = os.path.join(kwargs['out_path'], "video", k, label_name)
                os.makedirs(out_path_dir, exist_ok=True)
                out_path_vid = os.path.join(out_path_dir, f"{z_seed}.mp4")
                save_video_grid(v, out_path_vid, drange=[-1, 1], fps=8)
    
    def render_fake_single(self, *args, **kwargs):
        self.render_fake(*args, **kwargs)
    
    def render_fake_top_down(self, *args, **kwargs):
        self.render_fake(*args, **kwargs, top_down=True)

    def render_fake_video_dataset(self, video_dataset=True, *args, **kwargs):
        kwargs["outs"] = ['image']
        self.render_fake(video_dataset=video_dataset, *args, **kwargs)
    
    def render_fake_image_dataset(self, image_dataset=True, *args, **kwargs):
        kwargs["outs"] = ['image']
        self.render_fake(image_dataset=image_dataset, *args, **kwargs)
    
    def render_fake_single_depth(self, render_depth=True, neural_rendering_resolution=256, outs=['image_depth'], *args, **kwargs):
        self.render_fake(outs=outs, render_depth=render_depth, neural_rendering_resolution=neural_rendering_resolution, *args, **kwargs)

