import random
from typing import Dict


import torch.nn.functional as F
import numpy as np

#----------------------------------------------------------------------------

def sample_frames(cfg: Dict, total_video_len: int, **kwargs) -> np.ndarray:
    if cfg['type'] == 'random':
        return random_frame_sampling(cfg, total_video_len, **kwargs)
    elif cfg['type'] == 'uniform':
        return uniform_frame_sampling(cfg, total_video_len, **kwargs)
    else:
        raise NotImplementedError

#----------------------------------------------------------------------------

def random_frame_sampling(cfg: Dict, total_video_len: int, use_fractional_t: bool=False) -> np.ndarray:
    min_time_diff = cfg["num_frames_per_video"] - 1
    max_time_diff = min(total_video_len - 1, cfg.get('max_dist', float('inf')))

    if type(cfg.get('total_dists')) in (list, tuple):
        time_diff_range = [d for d in cfg['total_dists'] if min_time_diff <= d <= max_time_diff]
    else:
        time_diff_range = range(min_time_diff, max_time_diff)

    time_diff: int = random.choice(time_diff_range)
    if use_fractional_t:
        offset = random.random() * (total_video_len - time_diff - 1)
    else:
        offset = random.randint(0, total_video_len - time_diff - 1)
    frames_idx = [offset]

    if cfg["num_frames_per_video"] > 1:
        frames_idx.append(offset + time_diff)

    if cfg["num_frames_per_video"] > 2:
        frames_idx.extend([(offset + t) for t in random.sample(range(1, time_diff), k=cfg["num_frames_per_video"] - 2)])

    frames_idx = sorted(frames_idx)

    return np.array(frames_idx)

#----------------------------------------------------------------------------

def uniform_frame_sampling(cfg: Dict, total_video_len: int, use_fractional_t: bool=False) -> np.ndarray:
    # Step 1: Select the distance between frames
    if type(cfg.get('dists_between_frames')) in (list, tuple):
        valid_dists = [d for d in cfg['dists_between_frames'] if d <= ['max_dist_between_frames']]
        valid_dists = [d for d in valid_dists if (d * cfg['num_frames_per_video'] - d + 1) <= total_video_len]
        d = random.choice(valid_dists)
    else:
        max_dist = min(cfg.get('max_dist', float('inf')), total_video_len // cfg['num_frames_per_video'])
        d = random.randint(1, max_dist)

    d_total = d * cfg['num_frames_per_video'] - d + 1

    # Step 2: Sample.
    if use_fractional_t:
        offset = random.random() * (total_video_len - d_total)
    else:
        offset = random.randint(0, total_video_len - d_total)

    frames_idx = offset + np.arange(cfg['num_frames_per_video']) * d

    return frames_idx

#----------------------------------------------------------------------------