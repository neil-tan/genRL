# %%
import numpy as np
import torch
import gymnasium as gym
import custom_envs.cartpole as gen_cartpole
import genesis as gs
import sys
import time

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
env = gym.make(custom_environment_spec, render_mode="ansi", max_force=1000, targetVelocity=5)

# %%
def training_loop(env, max_steps=300):
    env.reset()
    for i in range(max_steps):
        action = env.action_space.sample()
        # observation, reward, done, truncated, info
        obs, reward, done, _, info = env.step(action)
    
        if done:
            break

    env.close() # stop the viewer and save the video

# %%
if not sys.platform == "linux":
    gs.tools.run_in_another_thread(fn=training_loop, args=(env, 300))
else:
    training_loop(env, 300)

# Render every second
for _ in range(8):  # Render 10 times
    env.render()
    time.sleep(1)
# %%
print("finished")