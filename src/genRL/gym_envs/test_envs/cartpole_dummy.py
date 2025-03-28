import numpy as np
import genesis as gs
import sys
import gymnasium as gym
from gymnasium import spaces
import torch

class GenCartPoleDummyBase(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}

    def __init__(self,
                 num_envs=1,
                 **kwargs,
                 ):
        self.num_envs = num_envs

        self.x_threshold = 2.4
        self.theta_threshold_degrees = 12
        self.theta_threshold_radians = self.theta_threshold_degrees * 2 * np.pi / 360 # 0.209
        self.current_steps_count = 0

        self.done = torch.zeros((self.num_envs, 1), dtype=torch.bool)

        # [Cart Position, Cart Velocity, Pole Angle (rad), Pole Velocity]
        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -np.inf, -4.1887903e-01, -np.inf]),
                                            np.array([4.8000002e+00, np.inf, 4.1887903e-01, np.inf]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)
        
        self.current_timestep = 0

        self.rewards = None
        self.cart_position = None
        self.cart_velocity = None
        self.pole_angles = None
        self.pole_angular_velocity = None
        
        self._init_dummy_trajectory()
        
        self.reset()
    
    def _init_dummy_trajectory(self):
        raise NotImplementedError

    def _get_observation_tuple(self):
        # Returns
        # Cart Position -4.8 ~ 4.8        
        # Cart Velocity -inf ~ inf
        # Pole Angle -0.418 ~ 0.418
        # Pole Velocity At Tip -inf ~ inf
        
        self.cart_position, self.cart_velocity, self.pole_angles, self.pole_angular_velocity
        
        cart_position = [self.cart_position[self.current_timestep]] * self.num_envs
        cart_velocity = [self.cart_velocity[self.current_timestep]] * self.num_envs
        pole_angles = [self.pole_angles[self.current_timestep]] * self.num_envs
        pole_angular_velocity = [self.pole_angular_velocity[self.current_timestep]] * self.num_envs
        
        cart_position = torch.stack(cart_position, dim=0)
        cart_velocity = torch.stack(cart_velocity, dim=0)
        pole_angles = torch.stack(pole_angles, dim=0)
        pole_angular_velocity = torch.stack(pole_angular_velocity, dim=0)
    
        return cart_position, cart_velocity, pole_angles, pole_angular_velocity

    # return observation for external viewer
    def observation(self, observation=None):
        observation = observation if observation else torch.stack(self._get_observation_tuple(), dim=-1)
        observation = observation if self.num_envs > 1 or self.return_tensor else observation.squeeze(0).cpu().numpy()
        return observation
    

    def _get_info(self):
        return {}

    @torch.no_grad()
    def reset(self, seed=None, options=None):
        self.current_timestep = 0

        super().reset(seed=seed)
        self._seed = seed

        return self.observation(), self._get_info()
    

    @torch.no_grad()
    # action shape: (num_envs, action_dim)
    def step(self, action):
        assert action.shape == (self.num_envs, 1), f"action shape {action.shape} is not valid"
        assert action.dtype == torch.int64, f"action dtype {action.dtype} is not valid"
        assert action.min() == 0 and action.max() == 1, f"action {action} is not valid"

        done = torch.zeros((self.num_envs), 1) if self.current_timestep < self.rewards.shape[0] - 1 else torch.ones((self.num_envs), 1)
        done = done.to(torch.bool)
        reward = self.rewards[self.current_timestep] if self.current_timestep < self.rewards.shape[0] else torch.tensor(0.0, dtype=self.rewards.dtype)
        reward = torch.stack([reward] * self.num_envs, dim=0)
        
        obs = self.observation()
        info = self._get_info()
        
        self.current_timestep += 1

        # observation, reward, done, truncated, info
        return obs, reward, done, False, info
    
    def render(self):
        pass
    
    def close(self):
        pass



class GenCartPoleDummyOnes(GenCartPoleDummyBase):
    def _init_dummy_trajectory(self):
        reward_length = 5
        self.rewards = torch.ones((reward_length), dtype=torch.float32)
        self.cart_position = torch.ones((reward_length), dtype=torch.float32) * 0.5
        self.cart_velocity = torch.ones((reward_length), dtype=torch.float32) * 0.5
        self.pole_angles = torch.ones((reward_length), dtype=torch.float32) * 0.1
        self.pole_angular_velocity = torch.ones((reward_length), dtype=torch.float32) * 0.5


custom_environment_spec = gym.register(id='GenCartPole-v0-dummy-ones', 
                                        entry_point=GenCartPoleDummyOnes,
                                        reward_threshold=2000, 
                                        max_episode_steps=2000,
                                        vector_entry_point=GenCartPoleDummyOnes,
                                        )