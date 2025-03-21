import torch
import functools
import numpy as np
from typing import List

def masked_mean(x, mask):
    return (x * mask).sum() / max(mask.sum(), 1)

def masked_sum(x, mask):
    return (x * mask).sum()

def masked_var(x, mask):
    N = max(mask.sum(), 1)
    mean = masked_mean(x, mask)
    return ((x - mean) ** 2 * mask).sum() / N

def masked_std(x, mask):
    return torch.sqrt(masked_var(x, mask) + 1e-8)

def normalize_advantage(advantages, valid_mask):
    assert advantages.shape[0] > 1
    mean = masked_mean(advantages, valid_mask)
    std = masked_std(advantages, valid_mask)
    return (advantages - mean) * valid_mask / (std + 1e-8)

# input: (batch_size, seq_len)
@functools.lru_cache(maxsize=8)
def mask_right_shift(mask):
    return torch.cat([torch.zeros_like(mask[:, 0:1]), mask[:, :-1]], dim=1)


def downsample_list_image_to_video_array(images:List[np.array], factor:int):
    # output: (T, C, H, W)
    assert len(images) > 0 and isinstance(images[0], np.ndarray)
    assert len(images[0].shape) == 3

    images = np.stack(images[::factor], axis=0)
    images = np.transpose(images, (0, 3, 1, 2))
    return images
    