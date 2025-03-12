import torch
    
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
