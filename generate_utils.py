import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import scipy.ndimage
import scipy.spatial

CLASSES_3DFRONT_WITH_LAMPS = {
    0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chair', 5: 'children_cabinet', 6: 'coffee_table',
    7: 'desk', 8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed', 12: 'night_stand',
    13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed', 16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe',
    21: 'start', 22: 'end'
}

CLASSES_3DFRONT_WITHOUT_LAMPS = {
    0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'chair', 4: 'children_cabinet', 5: 'coffee_table',
    6: 'desk', 7: 'double_bed', 8: 'dressing_chair', 9: 'dressing_table', 10: 'kids_bed', 11: 'night_stand',
    12: 'shelf', 13: 'single_bed', 14: 'sofa', 15: 'stool', 16: 'table', 17: 'tv_stand', 18: 'wardrobe',
    19: 'start', 20: 'end'
}

CLASSES_3DFRONT = {23: CLASSES_3DFRONT_WITH_LAMPS, 21: CLASSES_3DFRONT_WITHOUT_LAMPS}

# import open3d as o3d
bed_class_labels = {
    21: [7, 10, 13], # without lamps
    23: [8, 11, 15], # with lamps
    }

def get_camera_target_coords(labels, num_samples_scene=10, max_coords = [6.0, 4.0, 6.0], bed_size=1.0):
    scene_grid, bed_grid, room_layout, bed_center, valid_mask = create_voxel_grid(**labels, bed_size=bed_size)
    camera_coords, target_coords = sample_camera_positions(scene_grid, bed_grid, room_layout, bed_center, max_coords, num_samples_scene)
    return camera_coords, target_coords
        

def sample_camera_positions(scene_grid, bed_grid, room_layout, bed_center, max_coords, num_samples_scene=10,
                            camera_height=1.7, seed=0, top_perc=0.5, dist_perc=0.10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    scene_grid = scene_grid.squeeze(0)
    bed_grid = bed_grid.squeeze(0)
    room_layout = room_layout.squeeze(0)
    H, D, W = scene_grid.shape
    scene_grid = torch.flip(scene_grid, [2])
    bed_grid = torch.flip(bed_grid, [2])
    room_layout = room_layout.flip(dims=[1])
    bed_center[2] *= -1
    scene_layout = scene_grid.sum(dim=1) > 0
    scene_layout_dist = torch.from_numpy(bwdist_manhattan(scene_layout.cpu().numpy())).to(scene_layout.device)
    dist_thres = int(dist_perc * scene_layout.shape[0])
    unique_dist = torch.unique(scene_layout_dist)
    num_valid_sampels = 0
    i = 0
    while num_valid_sampels < num_samples_scene:
        dist_thres = unique_dist[unique_dist >= dist_thres][i]
        scene_layout_mask = scene_layout_dist == dist_thres
        valid_layout = torch.logical_and(room_layout, scene_layout_mask)
        num_valid_sampels = valid_layout.sum()
        i += 1
    valid_indices = torch.stack(torch.where(valid_layout), dim=1)
    # Convert indices back to coordinates
    valid_coords = valid_indices / (H - 1)
    # Shift x and z coordinate to be centered at 0 [-H/2,+H/2], [-W/2,+W/2]
    valid_coords[:,0] -= 0.5
    valid_coords[:,1] -= 0.5
    # Multiply H and W scaling back
    valid_coords[:, 0] *= max_coords[0]
    valid_coords[:, 1] *= max_coords[2]
    # Add camera height
    camera_height_coords = torch.ones((valid_coords.shape[0], 1), device=valid_coords.device) * camera_height
    valid_coords = torch.cat((valid_coords[:,[0]], camera_height_coords, valid_coords[:,[1]]), dim=1)
    # Flip grid to revert back indexing from bottom floorplan to normal coordinates
    bed_grid_flipped = torch.flip(bed_grid, [1])
    # Find points inside bed bounding box
    target_indices = torch.stack(torch.where(bed_grid_flipped), dim=1)
    target_coords = target_indices / (H - 1)
    # Shift x and z coordinate to be centered at 0 [-H/2,+H/2], [-W/2,+W/2]
    target_coords[:,0] -= 0.5
    target_coords[:,2] -= 0.5

    # Scale coordinates back to absolute values
    target_coords[:, 0] *= max_coords[0]
    target_coords[:, 1] *= max_coords[1]
    target_coords[:, 2] *= max_coords[2]

    # Adjust bed center to [0, 1] y coordinate and then absolute coordinates
    bed_center[1] += 0.5
    bed_center *= torch.tensor(max_coords, device=bed_center.device)

    # Get largest l2 distances
    l2_dist = (valid_coords - bed_center).square().sum(dim=1).sqrt()
    l2_dist_indices = l2_dist.sort(descending=True)[1]
    # Take the top x %
    num_random_set = int(l2_dist_indices.shape[0]*top_perc)
    if num_random_set < num_samples_scene:
        num_random_set = num_samples_scene
    valid_coords_selected = valid_coords
    target_coords_selected = target_coords
    return valid_coords_selected, target_coords_selected

def create_voxel_grid(class_labels, translations, sizes, angles, room_layout, bed_size=1.0, dims=(256, 256, 256), num_vertices = 256):
    """
    Sizes and translations need to be in [-0.5, 0.5]
    Args:
        class_labels: (OBJ, NUM_C)
        translations: (OBJ, 3=(x,y,z))
        sizes: (OBJ, 3=(x,y,z))
        angles: (OBJ, 1=(theta_z))
        room_layout: (1, H, W)
    Returns:
        scene_grid: (1, H, D, W)
        bed_grid: (1, H, D, W)
        room_layout: (1, H, W)
        bed_center: (3=(x, y, z))
    """
    translations = translations.flip(1)
    sizes = sizes.flip(1)
    translations[..., 1] *= -1
    room_layout = room_layout.permute(0, 2, 1)
    translations[:,2] *= -1
    room_layout = room_layout.flip(dims=[2])


    voxel_mask = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    voxel_masks = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    # Sort by largest objects
    obj_indices = torch.prod(sizes, 1).sort(descending=True)[1]
    # Find bed index
    bed_indices_mask = class_labels[:,bed_class_labels[class_labels.shape[-1]]]
    bed_indices_mask_objs = bed_indices_mask.sum(dim=1)
    # Pick largest bed if there are multiple
    bed_index = obj_indices[(bed_indices_mask_objs==1)[obj_indices]][0]
    bed_grid = None
    bed_center = None
    valid_mask = []
    for j in obj_indices:
        size = sizes[j]
        if j == bed_index and bed_size != 1.0:
            size = size * bed_size
        if torch.count_nonzero(size) == 0:
            break
        translation = translations[j]
        R = get_rotation_matrix(angles[j])
        xp = torch.linspace(- size[0], size[0], num_vertices, device=sizes.device)
        yp = torch.linspace(- size[1], size[1], num_vertices, device=sizes.device)
        zp = torch.linspace(- size[2], size[2], num_vertices, device=sizes.device)
        coords = torch.stack(torch.meshgrid(xp,yp,zp, indexing='ij')).view(3, -1)
        coords = torch.mm(R[0].T, coords) + translation.unsqueeze(-1)
        # Start y axis from bottom of the voxel grid to set floorplan at the bottom
        coords[1] *= -1
        # Clamp because of numerical precision
        coords = coords.clamp(-0.5, 0.5)
        occ_grid = voxelize(coords.transpose(1,0).unsqueeze(0), dims[0]).long()
        voxel_mask = torch.logical_and(voxel_masks == 0, occ_grid != 0)
        voxel_masks += voxel_mask

        # Valid object if it is on the floor
        if voxel_mask[:, :, -1, :].sum() > 0:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

        # Check if object is bed
        if j == bed_index:
            bed_grid = voxel_mask
            bed_center = translation.clone()

    room_layout = torch.nn.functional.interpolate(room_layout.unsqueeze(0),
                                                            size=(dims[0], dims[2]),
                                                            mode='nearest').squeeze(0)
    return voxel_masks, bed_grid, room_layout, bed_center, valid_mask

def get_rotation_matrix(theta):
    R = torch.zeros((1, 3, 3), device=theta.device)
    R[:, 0, 0] = torch.cos(theta)
    R[:, 0, 2] = torch.sin(theta)
    R[:, 2, 0] = -torch.sin(theta)
    R[:, 2, 2] = torch.cos(theta)
    R[:, 1, 1] = 1.
    return R

def voxelize(pc: torch.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    b, n, _ = pc.shape
    half_size = grid_size / 2.
    valid = (pc >= -half_size) & (pc <= half_size)
    valid = torch.all(valid, 2)
    pc_grid = (pc + half_size) * (voxel_size - 1.) / grid_size
    indices_floor = torch.floor(pc_grid)
    indices = indices_floor.long()
    batch_indices = torch.arange(b, device=pc.device)
    batch_indices = shape_padright(batch_indices)
    batch_indices = torch.tile(batch_indices, (1, n))
    batch_indices = shape_padright(batch_indices)
    indices = torch.cat((batch_indices, indices), 2)
    indices = torch.reshape(indices, (-1, 4))
    
    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]
    
    if valid.sum() == 0:
        return torch.zeros((b, grid_size, grid_size, grid_size), device=pc.device, dtype=torch.bool)

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = torch.tensor([[0] + pos], device=pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 3
        out = torch.zeros(*out_shape, device=pc.device).flatten()
        rav_ind = ravel_index(indices_loc.t(), out_shape, pc.device).long()
        rav_ind = rav_ind.clamp(0, voxel_size**3 - 1)
        voxels = out.scatter_add_(-1, rav_ind, updates).view(*out_shape)
        return voxels

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    voxels = sum(voxels)
    voxels = torch.clamp(voxels, 0., 1.)
    voxels = voxels > 0.5
    return voxels

# Source: neuralnet_pytorch
def ravel_index(indices, shape, device):
    assert len(indices) == len(shape), 'Indices and shape must have the same length'
    shape = torch.tensor(shape, device=device, dtype=torch.long)
    return sum([indices[i] * torch.prod(shape[i + 1:]) for i in range(len(shape))])

def shape_padright(x, n_ones=1):
    pattern = tuple(range(x.ndimension())) + ('x',) * n_ones
    return dimshuffle(x, pattern)

def dimshuffle(x, pattern):
    assert isinstance(pattern, (list, tuple)), 'pattern must be a list/tuple'
    no_expand_pattern = [x for x in pattern if x != 'x']
    y = x.permute(*no_expand_pattern)
    shape = list(y.shape)
    for idx, e in enumerate(pattern):
        if e == 'x':
            shape.insert(idx, 1)
    return y.view(*shape)

def bwdist_manhattan(a, seedval=1):
    seed_mask = a==seedval
    z = np.argwhere(seed_mask)
    nz = np.argwhere(~seed_mask)

    out = np.zeros(a.shape, dtype=int)
    out[tuple(nz.T)] = scipy.spatial.distance.cdist(z, nz, 'cityblock').min(0).astype(int)
    return out