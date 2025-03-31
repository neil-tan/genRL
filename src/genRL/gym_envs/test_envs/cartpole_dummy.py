import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class GenCartPoleDummyBase(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 60}
    episode_length = 5

    def __init__(self,
                 num_envs=1,
                 **kwargs,
                 ):
        self.num_envs = num_envs
        self.return_tensor = True
        self.device = "cpu"

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
    
    def _init_dummy_trajectory(self):
        raise NotImplementedError
    
    def _get_observation_tuple(self):
        raise NotImplementedError
    
    def _is_done(self):
        done = torch.zeros((self.num_envs), 1) if self.current_timestep < self.episode_length -1 else torch.ones((self.num_envs), 1)
        done = done.to(torch.bool)
        return done
    
    def _is_terminal(self):
        raise NotImplementedError

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
        obs, info = self.observation(), self._get_info()
        self.current_timestep = 1

        return obs, info
    

    @torch.no_grad()
    # action shape: (num_envs, action_dim)
    def step(self, action):
        assert action.shape == (self.num_envs, 1), f"action shape {action.shape} is not valid"
        assert action.dtype == torch.int64, f"action dtype {action.dtype} is not valid"

        done = self._is_done()
        assert done.shape == (self.num_envs, 1), f"done shape {done.shape} is not valid"
        assert done.dtype == torch.bool, f"done dtype {done.dtype} is not valid"
        # reward is 1 timestep behind because it is not returned during reset() unlike states
        reward_timestep = self.current_timestep - 1
        if reward_timestep < 0 or reward_timestep >= self.rewards.shape[0]:
            # if the reward is out of bounds, set it to 0
            reward = torch.tensor(0.0, dtype=self.rewards.dtype)
        else:
            reward = self.rewards[reward_timestep]
        reward = torch.stack([reward] * self.num_envs, dim=0)
        reward[1:] = reward[1:] * (~done.squeeze(-1))[0:-1]
        
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
    def __init__(self, num_envs=1, **kwargs):
        super().__init__(num_envs, **kwargs)
    
    def _init_dummy_trajectory(self):
        self.rewards = torch.ones((self.episode_length), dtype=torch.float32)
        self.cart_position = torch.ones((self.episode_length), dtype=torch.float32) * 0.5
        self.cart_velocity = torch.ones((self.episode_length), dtype=torch.float32) * 0.5
        self.pole_angles = torch.ones((self.episode_length), dtype=torch.float32) * 0.1
        self.pole_angular_velocity = torch.ones((self.episode_length), dtype=torch.float32) * 0.5

    @torch.no_grad()
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
        
        cart_position = torch.stack(cart_position, dim=0) # (num_envs)
        cart_velocity = torch.stack(cart_velocity, dim=0) # (num_envs)
        pole_angles = torch.stack(pole_angles, dim=0) # (num_envs)
        pole_angular_velocity = torch.stack(pole_angular_velocity, dim=0) # (num_envs)
    
        return cart_position, cart_velocity, pole_angles, pole_angular_velocity

class GenCartPoleDummyInverseTrig(GenCartPoleDummyBase):
    def __init__(self, num_envs=1, **kwargs):
        super().__init__(num_envs, **kwargs)
        self.episode_length = 5
        
    def _init_dummy_trajectory(self):
        self.rewards = torch.ones((self.episode_length-1), dtype=torch.float32)
        self.cart_position = torch.ones((self.episode_length), dtype=torch.float32) * 0.5
        self.cart_velocity = torch.ones((self.episode_length), dtype=torch.float32) * 0.5
        self.pole_angles = torch.ones((self.episode_length), dtype=torch.float32) * 0.1
        self.pole_angular_velocity = torch.ones((self.episode_length), dtype=torch.float32) * 0.5
        
        self.inverse_trig_mask = torch.ones((self.num_envs, self.episode_length), dtype=torch.bool)
        for n in range(self.num_envs):
            zero_start_index = self.episode_length - n
            self.inverse_trig_mask[n, zero_start_index:] = False
    
    def _is_done(self):
        if self.current_timestep >= self.episode_length - 1:
            return torch.ones((self.num_envs, 1), dtype=torch.bool)

        current_mask = self.inverse_trig_mask[:, self.current_timestep]
        next_mask = self.inverse_trig_mask[:, self.current_timestep + 1] if self.current_timestep < self.episode_length - 1 else torch.zeros_like(current_mask)
        done = torch.zeros_like(current_mask, dtype=torch.bool)
        for i in range(self.num_envs):
            if current_mask[i] and not next_mask[i]:
                done[i:] = True
        done = done.unsqueeze(-1)
        return done
    
    @torch.no_grad()
    def _get_observation_tuple(self):
        cart_position = self.cart_position[self.current_timestep]
        cart_velocity = self.cart_velocity[self.current_timestep]
        pole_angles = self.pole_angles[self.current_timestep]
        pole_angular_velocity = self.pole_angular_velocity[self.current_timestep]
        
        cart_position = [cart_position] * self.num_envs
        cart_velocity = [cart_velocity] * self.num_envs
        pole_angles = [pole_angles] * self.num_envs
        pole_angular_velocity = [pole_angular_velocity] * self.num_envs

        cart_position = torch.stack(cart_position, dim=0)
        cart_velocity = torch.stack(cart_velocity, dim=0)
        pole_angles = torch.stack(pole_angles, dim=0)
        pole_angular_velocity = torch.stack(pole_angular_velocity, dim=0)

        mask_slice = self.inverse_trig_mask[:, self.current_timestep]
        
        cart_position = cart_position * mask_slice
        cart_velocity = cart_velocity * mask_slice
        pole_angles = pole_angles * mask_slice
        pole_angular_velocity = pole_angular_velocity * mask_slice
        
        return cart_position, cart_velocity, pole_angles, pole_angular_velocity
        
custom_environment_dummy_one_spec = gym.register(id='GenCartPole-v0-dummy-ones', 
                                            entry_point=GenCartPoleDummyOnes,
                                            reward_threshold=2000, 
                                            max_episode_steps=2000,
                                            vector_entry_point=GenCartPoleDummyOnes,
                                            )

custom_environment_inverse_trig_spec = gym.register(id='GenCartPole-dummy_inverse_trig-v0',
                                            entry_point=GenCartPoleDummyInverseTrig,
                                            reward_threshold=2000, 
                                            max_episode_steps=2000,
                                            vector_entry_point=GenCartPoleDummyInverseTrig,
                                            )