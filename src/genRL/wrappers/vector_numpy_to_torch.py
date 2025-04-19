import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorWrapper
from gymnasium.spaces import Box
from gymnasium.vector import VectorEnv

class AutoTorchWrapper(VectorWrapper):
    """Wraps a vectorized environment to ensure outputs are PyTorch tensors on the specified device.

    Converts observations, rewards, terminated, and truncated flags from NumPy arrays
    to PyTorch tensors if they are not already tensors. Ensures all output tensors
    reside on the specified device.
    Converts input actions tensor to NumPy for the underlying environment if needed.
    """

    # __init__ remains largely the same, but it will only be called
    # if __new__ returns an instance of VectorNumpyToTorch.
    
    def __init__(self, env, device):
        super().__init__(env)
        # No longer strictly requires Box spaces, but they are common.
        # if not isinstance(env.observation_space, Box): ...
        # if not isinstance(env.action_space, Box): ...
        self.device = device
        # Spaces remain unchanged conceptually.

    def step(self, actions):
        """Steps the environment. Ensures actions are NumPy for env, converts outputs to tensors."""
        # Ensure actions are NumPy for the underlying environment
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions_np = actions # Assume it's already NumPy
        else:
            # Attempt conversion, might fail for unsupported types
            try:
                 actions_np = np.array(actions)
            except Exception as e:
                 raise TypeError(f"Actions must be convertible to NumPy array, got {type(actions)}. Error: {e}")

        # Step the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(actions_np)

        # Convert outputs to tensors on the target device if they aren't already
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        # Ensure boolean tensors for terminated/truncated
        terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

        # Ensure tensors have the correct shape (especially reward, term, trunc)
        # Example: reward might be (num_envs,) but needs to be (num_envs, 1)
        # This depends on the algorithm's expectations, adjust if needed.
        # reward_tensor = reward_tensor.view(-1, 1)
        # terminated_tensor = terminated_tensor.view(-1, 1)
        # truncated_tensor = truncated_tensor.view(-1, 1)

        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment, converts observation to tensor."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert observation to tensor on the target device if it isn't already
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        return obs_tensor, info