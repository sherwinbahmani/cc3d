import torch
from visu.threedfront import THREEDFRONT_LABELS

def linear_schedule(step, val_start, val_end, period):
    if step >= period:
        return val_end
    elif step <= 0:
        return val_start
    else:
        return val_start + (val_end - val_start) * step / period

def get_rotation_matrix(theta):
    R = torch.zeros((1, 3, 3), device=theta.device)
    R[:, 0, 0] = torch.cos(theta)
    R[:, 0, 2] = torch.sin(theta)
    R[:, 2, 0] = -torch.sin(theta)
    R[:, 2, 2] = torch.cos(theta)
    R[:, 1, 1] = 1.
    return R

def create_voxel_grid(class_labels, translations, sizes, angles, room_layout, semantic_layout=None, dims=(None, None, None), num_vertices = None, colored=False, out_scale=1.):
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
    if dims == (None, None, None):
        dims = (room_layout.shape[-1],) * len(dims)
    if num_vertices is None:
        num_vertices = dims[0]
    voxel_mask = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    voxel_masks = torch.zeros((1, *dims), device=sizes.device, dtype=sizes.dtype)
    voxel_mask_col = torch.ones((3, *dims), device=sizes.device, dtype=sizes.dtype)
    obj_indices = torch.prod(sizes, 1).sort(descending=True)[1]
    for j in obj_indices:
        size = sizes[j]
        if torch.count_nonzero(size) == 0:
            break
        translation = translations[j]
        R = get_rotation_matrix(angles[j])
        xp = torch.linspace(- size[0], size[0], num_vertices, device=sizes.device)
        yp = torch.linspace(- size[1], size[1], num_vertices, device=sizes.device)
        zp = torch.linspace(- size[2], size[2], num_vertices, device=sizes.device)
        coords = torch.stack(torch.meshgrid(xp,yp,zp, indexing='ij')).view(3, -1)
        coords = torch.mm(R[0].T, coords) + translation.unsqueeze(-1)
        coords[1] *= -1
        coords = coords.clamp(-0.5, 0.5)
        occ_grid = voxelize(coords.transpose(1,0).unsqueeze(0), dims[0]).long()
        voxel_mask = torch.logical_and(voxel_masks == 0, occ_grid != 0)
        voxel_masks += voxel_mask
        voxel_mask_col[:, voxel_mask[0]] = torch.rand((3,1,1,1), device=voxel_mask.device).expand(-1,*dims)[:, voxel_mask[0]]*out_scale/255
    if colored:
        return voxel_mask_col
    else:
        return voxel_masks

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

def voxelize2d(pc: torch.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
    pc = pc[:,:,[0, 2]]
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
    indices = torch.reshape(indices, (-1, 3))

    r = pc_grid - indices_floor
    rr = (1. - r, r)
    if filter_outlier:
        valid = valid.flatten()
        indices = indices[valid]
    
    if valid.sum() == 0:
        return torch.zeros((b, grid_size, grid_size), device=pc.device, dtype=torch.bool)

    def interpolate_scatter2d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1]
        updates = updates_raw.flatten()

        if filter_outlier:
            updates = updates[valid]

        indices_shift = torch.tensor([[0] + pos], device=pc.device)
        indices_loc = indices + indices_shift
        out_shape = (b,) + (voxel_size,) * 2
        out = torch.zeros(*out_shape, device=pc.device).flatten()
        rav_ind = ravel_index(indices_loc.t(), out_shape, pc.device).long()
        rav_ind = rav_ind.clamp(0, voxel_size**2 - 1)
        voxels = out.scatter_add_(-1, rav_ind, updates).view(*out_shape)
        return voxels

    voxels = [interpolate_scatter2d([k, j]) for k in range(2) for j in range(2)]
    voxels = sum(voxels)
    voxels = torch.clamp(voxels, 0., 1.)
    voxels = voxels > 0.5
    return voxels.unsqueeze(2)

def voxelize2(pc: torch.Tensor, voxel_size: int, grid_size=1., filter_outlier=True):
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
        rav_ind = rav_ind.view(b, -1).clamp((torch.arange(0, b, device=pc.device)*voxel_size**3).unsqueeze(1) (torch.arange(1, b+1, device=pc.device)*voxel_size**3 - 1).unsqueeze(1)).view(-1)
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

def tensor_linspace(start, end, steps, device):
    """
    Source: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps, device=device).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps, device=device).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out