import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorWrapper
from gymnasium.spaces import Box

class VectorNumpyToTorch(VectorWrapper):
    """Wraps a vectorized environment to convert NumPy outputs to PyTorch tensors.

    Specifically converts observations, rewards, terminated, and truncated flags
    from NumPy arrays to PyTorch tensors residing on the specified device.
    Assumes the input actions to `step` are already PyTorch tensors.
    """
    def __init__(self, env, device):
        super().__init__(env)
        if not isinstance(env.observation_space, Box):
            raise TypeError(
                f"Expected observation space to be gymnasium.spaces.Box, actual type: {type(env.observation_space)} "
            )
        if not isinstance(env.action_space, Box):
             raise TypeError(
                f"Expected action space to be gymnasium.spaces.Box, actual type: {type(env.action_space)} "
            )
        self.device = device
        # Observation space remains the same shape/dtype conceptually,
        # but reset() will return tensors.
        # Action space also remains the same, but step() expects tensors.

    def step(self, actions):
        """Steps the environment with PyTorch actions, converts outputs to tensors."""
        # Convert actions tensor to numpy for the underlying vector env
        actions_np = actions.cpu().numpy()

        # Step the underlying environment
        obs, reward, terminated, truncated, info = self.env.step(actions_np)

        # Convert outputs to tensors on the target device
        obs_tensor = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)
        reward_tensor = torch.from_numpy(reward).to(dtype=torch.float32, device=self.device)
        terminated_tensor = torch.from_numpy(terminated).to(dtype=torch.bool, device=self.device)
        truncated_tensor = torch.from_numpy(truncated).to(dtype=torch.bool, device=self.device)

        # Keep info as is (often contains non-numeric data)
        # If info needs conversion, it requires more complex handling

        return obs_tensor, reward_tensor, terminated_tensor, truncated_tensor, info

    def reset(self, *, seed=None, options=None):
        """Resets the environment, converts observation to tensor."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Convert observation to tensor on the target device
        obs_tensor = torch.from_numpy(obs).to(dtype=torch.float32, device=self.device)

        return obs_tensor, info 