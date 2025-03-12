import pytest
import os
import torch
import numpy as np
from genRL.utils import masked_mean, masked_sum, masked_var, masked_std, normalize_advantage

def test_masked_mean():
    # Basic test
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    result = masked_mean(x, mask)
    assert torch.isclose(result, torch.tensor(1.5))
    
    # All masked out
    mask = torch.zeros_like(x)
    result = masked_mean(x, mask)
    assert torch.isclose(result, torch.tensor(0.0))
    
    # None masked out
    mask = torch.ones_like(x)
    result = masked_mean(x, mask)
    assert torch.isclose(result, torch.tensor(2.5))
    
    # 2D tensors
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    result = masked_mean(x, mask)
    assert torch.isclose(result, torch.tensor(8.0 / 3.0))

def test_masked_sum():
    # Basic test
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    result = masked_sum(x, mask)
    assert torch.isclose(result, torch.tensor(3.0))
    
    # All masked out
    mask = torch.zeros_like(x)
    result = masked_sum(x, mask)
    assert torch.isclose(result, torch.tensor(0.0))
    
    # None masked out
    mask = torch.ones_like(x)
    result = masked_sum(x, mask)
    assert torch.isclose(result, torch.tensor(10.0))
    
    # 2D tensors
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    result = masked_sum(x, mask)
    assert torch.isclose(result, torch.tensor(8.0))

def test_masked_var():
    # Basic test
    x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    result = masked_var(x, mask)
    # Mean of masked values is 2.0, variance is ((1-2)^2 + (3-2)^2)/2 = 1.0
    assert torch.isclose(result, torch.tensor(1.0))
    
    # All masked out
    mask = torch.zeros_like(x)
    result = masked_var(x, mask)
    assert torch.isclose(result, torch.tensor(0.0))
    
    # None masked out
    mask = torch.ones_like(x)
    result = masked_var(x, mask)
    # Mean is 4.0, variance is ((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2)/4 = 5.0
    assert torch.isclose(result, torch.tensor(5.0))

def test_masked_std():
    # Basic test
    x = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    var_result = masked_var(x, mask)
    std_result = masked_std(x, mask)
    assert torch.isclose(std_result, torch.sqrt(var_result + 1e-8))
    
    # All masked out
    mask = torch.zeros_like(x)
    var_result = masked_var(x, mask)
    std_result = masked_std(x, mask)
    assert torch.isclose(std_result, torch.sqrt(var_result + 1e-8))
    
    # None masked out
    mask = torch.ones_like(x)
    var_result = masked_var(x, mask)
    std_result = masked_std(x, mask)
    assert torch.isclose(std_result, torch.sqrt(var_result + 1e-8))
    assert torch.isclose(std_result, torch.tensor(np.sqrt(5.0 + 1e-8), dtype=std_result.dtype))

def test_normalize_advantage():
    # Basic test
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    result = normalize_advantage(advantages, mask)
    
    # Calculate expected result
    mean_val = advantages.mean()
    std_val = advantages.std(unbiased=False)
    expected = (advantages - mean_val) / (std_val + 1e-8)
    assert torch.allclose(result, expected)
    
    # Test with partial mask
    advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    result = normalize_advantage(advantages, mask)
    
    # Only values [1.0, 2.0, 4.0, 6.0] should be considered for mean/std
    masked_values = torch.tensor([1.0, 2.0, 4.0, 6.0])
    mean_val = masked_values.mean()
    std_val = masked_values.std(unbiased=False)
    
    expected = (advantages - mean_val) * mask / (std_val + 1e-8)
    assert torch.allclose(result, expected)
    
    # Test with multi-dimensional advantages
    advantages = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], 
                              [[5.0, 6.0], [7.0, 8.0]]])
    mask = torch.tensor([[[1.0, 1.0], [1.0, 0.0]], 
                        [[0.0, 1.0], [1.0, 1.0]]])
    result = normalize_advantage(advantages, mask)
    
    # Calculate expected result for masked values
    masked_adv = advantages * mask
    masked_sum = masked_adv.sum()
    n_elements = mask.sum()
    masked_mean = masked_sum / n_elements
    
    masked_var = (((advantages - masked_mean) ** 2) * mask).sum() / n_elements
    masked_std = torch.sqrt(masked_var + 1e-8)
    
    expected = (advantages - masked_mean) * mask / (masked_std + 1e-8)
    assert torch.allclose(result, expected)

if __name__ == "__main__":
    pytest.main(["-s", "-v", os.path.abspath(__file__)])