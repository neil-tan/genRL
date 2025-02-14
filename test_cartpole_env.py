# %%
import numpy as np
import torch
import gymnasium as gym
import custom_envs.cartpole as gen_cartpole

# %%
# set random seed
np.random.seed(98765)
torch.manual_seed(98765)

# %%
custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/gen_cartpole-v1', 
                                                   entry_point=gen_cartpole.GenCartPoleEnv,
                                                   reward_threshold=2000, 
                                                   max_episode_steps=2000,
                                                   )
# %%
env = gym.make(custom_environment_spec, render_mode="rgb_array", max_force=1000, targetVelocity=1)

# %%